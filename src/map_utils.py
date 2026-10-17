"""
Interactive mapping module for Q-Rescue using Folium and OpenStreetMap.
Renders verified Belagavi district flood monitoring locations, rescue base,
risk-colored markers, and allocation details.
"""

import folium
from folium import plugins
import pandas as pd
from typing import Optional, List, Dict, Any

# Map marker color mapping
FOLIUM_COLOR_MAP = {
    "Critical": "red",
    "High": "orange",
    "Medium": "beige",
    "Low": "green"
}

HEX_COLOR_MAP = {
    "Critical": "#EF4444",
    "High": "#F97316",
    "Medium": "#F59E0B",
    "Low": "#10B981"
}

# Belagavi District Center & Rescue Base
BELAGAVI_CENTER = [16.18, 74.78]
BELAGAVI_DEOC_BASE = {
    "name": "Belagavi Central Emergency Operations Centre (DEOC)",
    "lat": 15.8497,
    "lon": 74.4977
}

def generate_flood_risk_map(df: pd.DataFrame, 
                            risk_col: str = "flood_risk_label",
                            selected_risks: Optional[List[str]] = None,
                            allocations_df: Optional[pd.DataFrame] = None) -> folium.Map:
    """
    Generate an interactive Folium map centered on Belagavi district.
    Plots DEOC central depot and all monitoring locations with risk-coded markers and popups.
    """
    m = folium.Map(
        location=BELAGAVI_CENTER,
        zoom_start=9,
        tiles="OpenStreetMap",
        control_scale=True
    )
    
    # 1. Add Central Rescue Base Marker
    base_html = f"""
    <div style="font-family: sans-serif; min-width: 200px;">
        <h4 style="color: #0284C7; margin-bottom: 4px;">📍 {BELAGAVI_DEOC_BASE['name']}</h4>
        <p style="margin: 0; font-size: 13px;"><b>Role:</b> Central Disaster Command & Dispatch Depot</p>
        <p style="margin: 0; font-size: 12px; color: #64748B;">Coordinates: 15.8497° N, 74.4977° E</p>
    </div>
    """
    folium.Marker(
        location=[BELAGAVI_DEOC_BASE["lat"], BELAGAVI_DEOC_BASE["lon"]],
        tooltip="Belagavi Central Rescue Base (DEOC)",
        popup=folium.Popup(base_html, max_width=300),
        icon=folium.Icon(color="blue", icon="shield", prefix="fa")
    ).add_to(m)

    # Filter dataframe by selected risk levels if specified
    filtered_df = df.copy()
    if selected_risks:
        filtered_df = filtered_df[filtered_df[risk_col].isin(selected_risks)]

    # Merge allocation info if available
    alloc_dict = {}
    if allocations_df is not None and not allocations_df.empty:
        for _, a_row in allocations_df.iterrows():
            alloc_dict[a_row["location_id"]] = a_row.to_dict()

    # 2. Add Location Markers
    for _, row in filtered_df.iterrows():
        lat = float(row.get("latitude", 0))
        lon = float(row.get("longitude", 0))
        if lat == 0 or lon == 0:
            continue
            
        risk = str(row.get(risk_col, "Low"))
        marker_color = FOLIUM_COLOR_MAP.get(risk, "blue")
        hex_color = HEX_COLOR_MAP.get(risk, "#3B82F6")
        
        loc_id = row.get("location_id", "")
        loc_name = row.get("location_name", "Location")
        taluk = row.get("taluk", "Belagavi")
        rainfall = row.get("rainfall_mm", "N/A")
        water_level = row.get("river_water_level_m", "N/A")
        elevation = row.get("elevation_m", "N/A")
        population = row.get("population", "N/A")
        dist = row.get("distance_to_rescue_base_km", "N/A")
        
        # Check if allocation details exist for this location
        alloc_html = ""
        if loc_id in alloc_dict:
            a_info = alloc_dict[loc_id]
            alloc_html = f"""
            <hr style="margin: 6px 0;"/>
            <div style="background: #1E293B; color: #F8FAFC; padding: 6px; border-radius: 4px; font-size: 11px;">
                <b>🚀 Optimized Allocation (QAOA):</b><br/>
                Priority Rank: #{a_info.get('priority_rank', 'N/A')}<br/>
                Teams: {a_info.get('teams_allocated', 0)} | Boats: {a_info.get('boats_allocated', 0)}<br/>
                Ambulances: {a_info.get('ambulances_allocated', 0)} | Food Units: {a_info.get('food_allocated', 0)}
            </div>
            """
            
        pop_str = f"{int(population):,}" if isinstance(population, (int, float)) else str(population)
        
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 220px; font-size: 12px; line-height: 1.4;">
            <h4 style="margin: 0 0 4px 0; color: {hex_color};">⚠️ {loc_name} ({taluk})</h4>
            <div style="display: inline-block; background-color: {hex_color}; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; margin-bottom: 6px;">
                {risk.upper()} RISK
            </div>
            <p style="margin: 2px 0;"><b>Rainfall:</b> {rainfall} mm</p>
            <p style="margin: 2px 0;"><b>River Water Level:</b> {water_level} m</p>
            <p style="margin: 2px 0;"><b>Elevation:</b> {elevation} m</p>
            <p style="margin: 2px 0;"><b>Population:</b> {pop_str}</p>
            <p style="margin: 2px 0;"><b>Distance to Base:</b> {dist} km</p>
            {alloc_html}
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{loc_name} ({taluk}): {risk} Risk",
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.85,
            weight=2
        ).add_to(m)

    # 3. Add Legend
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 25px; right: 25px; width: 170px; height: 165px; 
        background-color: #1E293B; border: 2px solid #334155; 
        border-radius: 6px; z-index: 9999; font-size: 12px; color: #F8FAFC;
        padding: 10px; font-family: sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    ">
        <b>🚨 Flood Risk Legend</b><br/>
        <div style="margin-top: 6px;"><i class="fa fa-circle" style="color: #EF4444;"></i> Critical Risk</div>
        <div style="margin-top: 4px;"><i class="fa fa-circle" style="color: #F97316;"></i> High Risk</div>
        <div style="margin-top: 4px;"><i class="fa fa-circle" style="color: #F59E0B;"></i> Medium Risk</div>
        <div style="margin-top: 4px;"><i class="fa fa-circle" style="color: #10B981;"></i> Low Risk</div>
        <div style="margin-top: 4px;"><i class="fa fa-shield" style="color: #0284C7;"></i> Rescue Base (DEOC)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m
