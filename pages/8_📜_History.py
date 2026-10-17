"""
Historical Observations and Allocation Runs Page for Q-Rescue.
Inspects past telemetry cycles and previous QAOA/Classical optimization runs stored in SQLite.
"""

import streamlit as st
import pandas as pd
from src.database import get_optimization_history, get_run_allocations, get_all_observations
from src.authentication import render_auth_sidebar

st.set_page_config(page_title="History | Q-Rescue", page_icon="📜", layout="wide")
render_auth_sidebar()

st.title("📜 Historical Observations & Dispatch Logs")
st.caption("Audit past telemetry monitoring cycles and historical resource allocation runs from SQLite.")

tab_runs, tab_obs = st.tabs(["⚛️ Past Optimization Runs", "📡 Multi-Date Observation Archive"])

with tab_runs:
    st.subheader("Historical Resource Allocation Runs")
    runs_df = get_optimization_history()
    
    if runs_df.empty:
        st.info("ℹ️ No previous optimization runs recorded in database yet. Run an allocation in the QAOA Optimization module.")
    else:
        st.markdown(f"**Found {len(runs_df)} historical dispatch runs:**")
        
        display_runs = runs_df[[
            "run_id", "run_timestamp", "algorithm", "total_locations_considered",
            "total_resources_allocated", "total_shortage", "constraint_satisfaction_rate",
            "execution_time_seconds"
        ]].copy()
        
        display_runs.columns = [
            "Run ID", "Timestamp", "Algorithm", "Locations",
            "Allocated Units", "Shortage", "Satisfaction (%)", "Time (s)"
        ]
        st.dataframe(display_runs, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Inspect Detailed Run Allocations")
        
        run_ids = runs_df["run_id"].tolist()
        sel_run_id = st.selectbox("Select Run ID to Inspect", run_ids)
        
        alloc_records = get_run_allocations(sel_run_id)
        if not alloc_records.empty:
            st.markdown(f"**Allocations for Run #{sel_run_id}:**")
            cols_show = [
                "priority_rank", "location_name", "taluk", "risk_level",
                "teams_allocated", "boats_allocated", "ambulances_allocated",
                "food_allocated", "medical_kits_allocated", "unmet_demand_flag", "notes"
            ]
            st.dataframe(alloc_records[cols_show], use_container_width=True)

with tab_obs:
    st.subheader("Multi-Date Belagavi Telemetry Archive")
    all_obs = get_all_observations()
    
    if all_obs.empty:
        st.warning("No observations found in database.")
    else:
        dates_available = sorted(list(all_obs["observation_date"].unique()), reverse=True)
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            sel_date = st.selectbox("Filter by Observation Date", ["All Dates"] + dates_available)
        with col_d2:
            st.metric("Total Archive Records", len(all_obs))
            
        view_obs = all_obs.copy()
        if sel_date != "All Dates":
            view_obs = view_obs[view_obs["observation_date"] == sel_date]
            
        show_cols = [
            "observation_date", "location_id", "location_name", "taluk", "flood_risk_label",
            "rainfall_mm", "river_water_level_m", "elevation_m", "population"
        ]
        st.dataframe(view_obs[show_cols], use_container_width=True, height=450)
