"""
Interactive Flood Risk Map for Belagavi District.
Renders verified GIS coordinates using Folium with risk color coding,
DEOC rescue base telemetry, popups, and recommended resource allocations.
"""

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from src.database import get_latest_observations
from src.authentication import render_auth_sidebar
from src.map_utils import generate_flood_risk_map, BELAGAVI_DEOC_BASE

st.set_page_config(page_title="Risk Map | Q-Rescue", page_icon="🗺️", layout="wide")
render_auth_sidebar()

st.title("🗺️ Interactive Belagavi Flood Risk & Dispatch Map")
st.caption("Geospatial monitoring across Krishna, Ghataprabha, Malaprabha, and Dudhganga river basins with live DEOC base.")

# Load observation telemetry
obs_df = get_latest_observations()

if obs_df.empty:
    st.warning("⚠️ No location coordinates available in the database. Please seed data in Data Management.")
    st.stop()

# Filters & Controls
col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
with col_f1:
    risk_options = ["Critical", "High", "Medium", "Low"]
    selected_risks = st.multiselect("Filter Markers by Hazard Level", risk_options, default=risk_options)

with col_f2:
    taluk_options = ["All Taluks"] + sorted(list(obs_df["taluk"].unique()))
    selected_taluk = st.selectbox("Focus Taluk", taluk_options)

with col_f3:
    st.write("")
    st.write("")
    include_alloc = st.checkbox("Overlay Latest QAOA Allocations", value=True)

# Prepare filtered dataframe
filtered_df = obs_df.copy()
if selected_taluk != "All Taluks":
    filtered_df = filtered_df[filtered_df["taluk"] == selected_taluk]

# Check if allocation results exist in session state
alloc_df = None
if include_alloc and "last_optimization_results" in st.session_state:
    alloc_df = st.session_state["last_optimization_results"]["allocation_df"]

# Map Metrics Overview
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Mapped Stations", len(filtered_df))
with m2:
    st.metric("Critical Red Zones", len(filtered_df[filtered_df["flood_risk_label"] == "Critical"]))
with m3:
    st.metric("High Alert Zones", len(filtered_df[filtered_df["flood_risk_label"] == "High"]))
with m4:
    st.metric("Dispatch Base", "Belagavi DEOC", delta="15.8497° N, 74.4977° E")

# Generate Folium Map
folium_map = generate_flood_risk_map(
    df=filtered_df,
    risk_col="flood_risk_label",
    selected_risks=selected_risks,
    allocations_df=alloc_df
)

# Render Map in Streamlit
st_folium(folium_map, width="100%", height=620)

st.markdown("""
<div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 12px; margin-top: 10px; font-size: 0.88rem; color: #CBD5E1;">
    💡 <b>Navigation Tips:</b> Click any colored circle marker to inspect real-time rainfall, river water gauge readings, elevation, population, 
    and recommended emergency resource allocations computed by QAOA. The blue shield icon marks the <b>Belagavi Central Emergency Operations Centre (DEOC)</b>.
</div>
""", unsafe_allow_html=True)
