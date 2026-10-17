"""
Data Management and CSV Ingestion module for Q-Rescue.
Supports CSV upload, schema validation, data quality reports, and database re-seeding.
"""

import streamlit as st
import pandas as pd
from src.database import get_all_observations, get_resource_inventory, seed_initial_data, get_connection
from src.authentication import render_auth_sidebar, require_auth
from src.data_loader import validate_and_load_csv, REQUIRED_COLUMNS, OPTIONAL_COLUMNS

st.set_page_config(page_title="Data Management | Q-Rescue", page_icon="📁", layout="wide")
render_auth_sidebar()

st.title("📁 Data Management & CSV Telemetry Ingestion")
st.caption("Upload external flood telemetry datasets, inspect database tables, and run schema validation.")

tab_upload, tab_inspect, tab_seed = st.tabs(["📤 CSV Ingestion & Quality Report", "🗄️ Database Table Inspector", "🔄 Re-Seed Demonstration Data"])

with tab_upload:
    st.subheader("Upload Telemetry CSV")
    st.markdown("""
    Upload a CSV file containing flood station observations. The system will perform strict validation,
    range boundary checks, deduplication, and compile a **Data Quality Report**.
    """)
    
    with st.expander("ℹ️ Expected CSV Columns & Schema Specification", expanded=False):
        st.markdown(f"**Required Columns:** `{', '.join(REQUIRED_COLUMNS)}`")
        st.markdown(f"**Optional Columns:** `{', '.join(OPTIONAL_COLUMNS)}`")
        st.caption("Rainfall: 0-1500 mm | River Gauge: 0-35 m | Elevation: -50 to 9000 m | Population: > 0")

    uploaded_file = st.file_uploader("Select Flood Telemetry CSV", type=["csv"])
    
    if uploaded_file is not None:
        cleaned_df, report = validate_and_load_csv(uploaded_file)
        
        st.markdown("---")
        st.subheader("📋 Data Quality Report")
        
        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1:
            st.metric("Total Rows Processed", report["total_rows"])
        with qc2:
            st.metric("Valid Retained Rows", report["valid_rows"])
        with qc3:
            st.metric("Duplicates Removed", report["duplicates_removed"])
        with qc4:
            val_status = "✅ PASSED" if report["is_valid"] else "❌ FAILED"
            st.metric("Schema Validity", val_status)

        if report["warnings"]:
            st.warning("⚠️ **Data Quality Warnings:**\n" + "\n".join(f"- {w}" for w in report["warnings"]))
            
        if report["errors"]:
            st.error("🚨 **Validation Errors:**\n" + "\n".join(f"- {e}" for e in report["errors"]))
        else:
            st.success("🎉 CSV passed all schema and range validations!")
            st.markdown("##### Cleaned Preview:")
            st.dataframe(cleaned_df.head(10), use_container_width=True)
            
            if st.button("💾 Ingest Uploaded Records into SQLite Database", use_container_width=True):
                with get_connection() as conn:
                    cursor = conn.cursor()
                    ingested = 0
                    for _, row in cleaned_df.iterrows():
                        # Insert location
                        cursor.execute("""
                            INSERT OR IGNORE INTO locations 
                            (location_id, location_name, taluk, latitude, longitude, elevation_m, population, distance_to_rescue_base_km, historical_flood_indicator)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            str(row["location_id"]), str(row["location_name"]), str(row["taluk"]),
                            float(row.get("latitude", 16.0)), float(row.get("longitude", 74.5)),
                            float(row["elevation_m"]), int(row["population"]), float(row.get("distance_to_rescue_base_km", 50.0)),
                            int(row.get("historical_flood_indicator", 0))
                        ))
                        # Insert observation
                        cursor.execute("""
                            INSERT INTO observations 
                            (location_id, observation_date, rainfall_mm, temperature_c, humidity_percent, river_water_level_m,
                             available_rescue_teams, available_boats, shelter_capacity, flood_risk_label)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            str(row["location_id"]), str(row.get("observation_date", pd.Timestamp.now().strftime("%Y-%m-%d"))),
                            float(row["rainfall_mm"]), float(row.get("temperature_c", 25.0)),
                            float(row.get("humidity_percent", 80.0)), float(row["river_water_level_m"]),
                            int(row.get("available_rescue_teams", 0)), int(row.get("available_boats", 0)),
                            int(row.get("shelter_capacity", 500)), str(row.get("flood_risk_label", "Low"))
                        ))
                        ingested += 1
                    conn.commit()
                st.success(f"✅ Ingested {ingested} observation records successfully!")

with tab_inspect:
    st.subheader("Live SQLite Table Explorer")
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall() if not r[0].startswith("sqlite_")]
        
    sel_table = st.selectbox("Select SQLite Table to Inspect", tables)
    
    if sel_table:
        with get_connection() as conn:
            table_content = pd.read_sql_query(f"SELECT * FROM {sel_table} LIMIT 100;", conn)
        st.markdown(f"**Table: `{sel_table}` (Displaying up to 100 rows):**")
        st.dataframe(table_content, use_container_width=True)

with tab_seed:
    st.subheader("Reset & Seed Belagavi Demonstration Dataset")
    st.markdown("""
    Restore the default multi-station dataset for Belagavi District (144 observation records across 24 localities
    covering Krishna, Ghataprabha, Malaprabha, and Dudhganga basins) and reset default depot inventory.
    """)
    
    if st.button("🔄 Re-Seed Demonstration Dataset", type="primary", use_container_width=True):
        seed_initial_data()
        st.success("✅ Demonstration dataset re-seeded successfully into SQLite!")
        st.rerun()
