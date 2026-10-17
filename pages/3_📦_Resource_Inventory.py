"""
Resource Inventory Management for Q-Rescue.
Allows emergency response coordinators to inspect, edit, and update the central depot
inventory for rescue teams, boats, ambulances, rations, and medical kits.
"""

import streamlit as st
import pandas as pd
from src.database import get_resource_inventory, update_resource_inventory
from src.authentication import render_auth_sidebar, require_auth

st.set_page_config(page_title="Resource Inventory | Q-Rescue", page_icon="📦", layout="wide")
render_auth_sidebar()

st.title("📦 Emergency Resource Inventory Management")
st.caption("Maintain central disaster depot stockpiles, mobilization status, and equipment readiness.")

# Fetch inventory
inv_df = get_resource_inventory()

if inv_df.empty:
    st.warning("⚠️ No inventory records found in database. Please seed initial data in Data Management.")
    st.stop()

# Overview Metric Cards
total_units = inv_df["total_available_units"].sum()
total_allocated = inv_df["allocated_units"].sum()
net_available = total_units - total_allocated

m_c1, m_c2, m_c3, m_c4 = st.columns(4)
with m_c1:
    st.metric("Total Equipment & Personnel", f"{total_units:,} Units")
with m_c2:
    st.metric("Currently Deployed", f"{total_allocated:,} Units")
with m_c3:
    st.metric("Net Ready for Dispatch", f"{net_available:,} Units", delta="Mobilizable")
with m_c4:
    st.metric("Resource Categories", len(inv_df), delta="Active Depots")

st.markdown("---")

st.subheader("📋 Central Depot Stock Ledger")

# Interactive inventory table editor
with st.form("inventory_edit_form"):
    st.markdown("Modify total available stockpile counts or assigned deployment units:")
    
    updated_records = []
    
    for idx, row in inv_df.iterrows():
        res_id = row["resource_id"]
        res_type = row["resource_type"]
        curr_total = int(row["total_available_units"])
        curr_alloc = int(row["allocated_units"])
        depot = row.get("depot_location", "Belagavi Central Depot")
        
        col_t1, col_t2, col_t3, col_t4 = st.columns([3, 2, 2, 3])
        with col_t1:
            st.markdown(f"**{res_type}** (`{res_id}`)")
            st.caption(f"Depot: {depot}")
        with col_t2:
            new_total = st.number_input(f"Total Stock ({res_id})", min_value=0, max_value=5000, value=curr_total, key=f"tot_{res_id}")
        with col_t3:
            new_alloc = st.number_input(f"Allocated Units ({res_id})", min_value=0, max_value=new_total, value=min(curr_alloc, new_total), key=f"alc_{res_id}")
        with col_t4:
            avail = new_total - new_alloc
            status_color = "#10B981" if avail > 10 else ("#F59E0B" if avail > 0 else "#EF4444")
            st.markdown(f"<span style='color: {status_color}; font-weight: bold;'>Net Ready: {avail} units</span>", unsafe_allow_html=True)
            
        updated_records.append((res_id, new_total, new_alloc))
        st.markdown("<hr style='margin: 8px 0; border-color: #334155;'/>", unsafe_allow_html=True)
        
    submit_inv = st.form_submit_button("💾 Save Inventory Updates to Database", use_container_width=True)
    
    if submit_inv:
        for r_id, n_tot, n_alc in updated_records:
            update_resource_inventory(r_id, n_tot, n_alc)
        st.success("✅ Resource inventory successfully updated in SQLite database!")
        st.rerun()

st.markdown("---")

st.subheader("📍 Designated Belagavi Rescue Depots")
depot_data = [
    {"Depot Name": "Belagavi Central DEOC Depot", "Role": "District Headquarters Command & Multi-Resource Reserve", "Lat": 15.8497, "Lon": 74.4977},
    {"Depot Name": "Chikkodi Water Rescue Depot", "Role": "Krishna-Dudhganga River Inflatable Boats & Diving Teams", "Lat": 16.4312, "Lon": 74.5982},
    {"Depot Name": "Gokak Civil Hospital Emergency Unit", "Role": "Ghataprabha Basin First Aid Kits & Medical Ambulances", "Lat": 16.1667, "Lon": 74.8333},
    {"Depot Name": "Raybag APMC Logistics Hub", "Role": "Food Supplies, Clean Water Tankers & Dry Rations", "Lat": 16.4833, "Lon": 74.7833},
    {"Depot Name": "Khanapur Western Ghats Fire Station", "Role": "High-Volume De-Watering Pumps & Heavy Clearing Units", "Lat": 15.6333, "Lon": 74.5167}
]
st.dataframe(pd.DataFrame(depot_data), use_container_width=True)
