"""
Flood Risk Dashboard for Belagavi District.
Displays real-time monitoring telemetry, summary cards, risk distribution charts,
and location-wise hazard tables.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.database import get_latest_observations, get_resource_inventory, get_all_observations
from src.authentication import render_auth_sidebar, require_auth
from src.visualizations import (
    create_risk_distribution_pie, 
    create_rainfall_waterlevel_scatter, 
    RISK_COLORS
)

st.set_page_config(page_title="Flood Risk Dashboard | Q-Rescue", page_icon="📊", layout="wide")
render_auth_sidebar()

st.title("📊 Belagavi Flood Risk Command Dashboard")
st.caption("Live decision-support overview across Krishna, Ghataprabha, Malaprabha, and Dudhganga river basins.")

# Data Fetching
obs_df = get_latest_observations()
inv_df = get_resource_inventory()

if obs_df.empty:
    st.warning("⚠️ No observation telemetry currently found in database. Please seed or upload data in Data Management.")
    st.stop()

# Header Refresh & Timestamp
col_act1, col_act2 = st.columns([4, 1])
with col_act1:
    latest_date = obs_df["observation_date"].max() if "observation_date" in obs_df.columns else "N/A"
    st.info(f"🕒 **Observation Cycle:** Telemetry synchronized for date `{latest_date}` | Total Stations: `{len(obs_df)}`")
with col_act2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

st.markdown("---")

# Summary Metric Cards
total_locs = len(obs_df)
crit_locs = len(obs_df[obs_df["flood_risk_label"] == "Critical"])
high_locs = len(obs_df[obs_df["flood_risk_label"] == "High"])
med_locs = len(obs_df[obs_df["flood_risk_label"] == "Medium"])
low_locs = len(obs_df[obs_df["flood_risk_label"] == "Low"])

# Estimated exposed population in High & Critical areas
exposed_pop = obs_df[obs_df["flood_risk_label"].isin(["High", "Critical"])]["population"].sum()

# Resource inventory counts
total_teams = int(inv_df[inv_df["resource_id"] == "RES-001"]["total_available_units"].sum()) if not inv_df.empty else 0
total_boats = int(inv_df[inv_df["resource_id"] == "RES-002"]["total_available_units"].sum()) if not inv_df.empty else 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="Total Monitored Stations", value=total_locs, delta="Belagavi District")
with m2:
    st.metric(label="Critical Risk Areas", value=crit_locs, delta="Immediate Action", delta_color="inverse")
with m3:
    st.metric(label="High Risk Areas", value=high_locs, delta="Alert Level", delta_color="inverse")
with m4:
    st.metric(label="Estimated Exposed Population", value=f"{exposed_pop:,}", help="Population residing in High and Critical flood zones")

m5, m6, m7, m8 = st.columns(4)
with m5:
    st.metric(label="Medium Risk Areas", value=med_locs)
with m6:
    st.metric(label="Low Risk / Safe Areas", value=low_locs)
with m7:
    st.metric(label="Available Rescue Teams", value=total_teams, delta="Central Depot")
with m8:
    st.metric(label="Available Rescue Boats", value=total_boats, delta="Water Rescue Unit")

st.markdown("---")

# Visual Charts Section
c_chart1, c_chart2 = st.columns(2)
with c_chart1:
    fig_pie = create_risk_distribution_pie(obs_df, risk_col="flood_risk_label")
    st.plotly_chart(fig_pie, use_container_width=True)

with c_chart2:
    fig_scatter = create_rainfall_waterlevel_scatter(obs_df, risk_col="flood_risk_label")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# Interactive Table with Filters
st.subheader("📋 Locality Telemetry & Risk Table")

f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    all_taluks = ["All Taluks"] + sorted(list(obs_df["taluk"].unique()))
    selected_taluk = st.selectbox("Filter by Taluk", all_taluks)
with f_col2:
    all_risks = ["All Risks", "Critical", "High", "Medium", "Low"]
    selected_risk = st.selectbox("Filter by Risk Level", all_risks)
with f_col3:
    search_query = st.text_input("Search Locality Name", placeholder="e.g. Kallol, Gokak...")

filtered_df = obs_df.copy()
if selected_taluk != "All Taluks":
    filtered_df = filtered_df[filtered_df["taluk"] == selected_taluk]
if selected_risk != "All Risks":
    filtered_df = filtered_df[filtered_df["flood_risk_label"] == selected_risk]
if search_query.strip():
    filtered_df = filtered_df[filtered_df["location_name"].str.contains(search_query.strip(), case=False)]

# Display formatted table
display_cols = [
    "location_id", "location_name", "taluk", "flood_risk_label", 
    "rainfall_mm", "river_water_level_m", "elevation_m", "population", "distance_to_rescue_base_km"
]
table_df = filtered_df[display_cols].copy()
table_df.columns = [
    "ID", "Locality", "Taluk", "Risk Level", 
    "Rainfall (mm)", "River Level (m)", "Elevation (m)", "Population", "Dist to Base (km)"
]

# Color badge styling helper
def highlight_risk(val):
    color = RISK_COLORS.get(val, "transparent")
    return f"background-color: {color}33; color: {color}; font-weight: bold; border-radius: 4px; padding: 2px 8px;"

styled_table = table_df.style.applymap(highlight_risk, subset=["Risk Level"])
st.dataframe(styled_table, use_container_width=True, height=420)

st.caption(f"Displaying {len(filtered_df)} of {len(obs_df)} monitored stations in Belagavi district.")
