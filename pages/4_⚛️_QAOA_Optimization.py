"""
Quantum Approximate Optimization Algorithm (QAOA) Resource Allocation Page for Q-Rescue.
Formulates multi-resource dispatch as a QUBO problem, executes simulated QAOA and Classical baselines,
reports constraint violations, and provides CSV export.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.database import get_resource_inventory, get_latest_observations, get_latest_predictions
from src.authentication import render_auth_sidebar, require_auth
from src.resource_allocation import execute_resource_allocation, format_allocation_csv
from src.visualizations import create_resource_comparison_bar, RISK_COLORS

st.set_page_config(page_title="QAOA Optimization | Q-Rescue", page_icon="⚛️", layout="wide")
render_auth_sidebar()

st.title("⚛️ Emergency Resource Optimization via QAOA")
st.caption("Quadratic Unconstrained Binary Optimization (QUBO) solved via simulated PennyLane QAOA and Classical Benchmark.")

# Retrieve current resource inventory
inv_df = get_resource_inventory()
if inv_df.empty:
    st.warning("⚠️ No resource inventory found. Please seed data in Data Management.")
    st.stop()

# Helper dict for inventory
curr_inv = {}
for _, r in inv_df.iterrows():
    rtype = r["resource_type"].lower()
    units = int(r["total_available_units"])
    if "team" in rtype:
        curr_inv["teams"] = units
    elif "boat" in rtype:
        curr_inv["boats"] = units
    elif "ambulance" in rtype:
        curr_inv["ambulances"] = units
    elif "food" in rtype:
        curr_inv["food"] = units
    elif "medical" in rtype:
        curr_inv["medical_kits"] = units

# Fallback defaults if missing
curr_inv.setdefault("teams", 35)
curr_inv.setdefault("boats", 24)
curr_inv.setdefault("ambulances", 20)
curr_inv.setdefault("food", 40)
curr_inv.setdefault("medical_kits", 50)

# Optimization Configuration Panel
st.subheader("⚙️ Dispatch & Optimization Parameters")

col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    algo_choice = st.selectbox(
        "Optimization Engine",
        ["QAOA (Quantum Approximate Optimization)", "Classical Priority Baseline (Greedy Knapsack)"],
        help="QAOA optimizes a QUBO cost Hamiltonian via parameterized quantum mixer and cost unitaries."
    )
with col_cfg2:
    qaoa_p_steps = st.slider("QAOA Circuit Depth (p-steps)", 1, 3, 1, help="Number of alternating cost and mixer Hamiltonian layers.")
with col_cfg3:
    max_qubits = st.slider("Quantum Candidate Set (Max Qubits)", 3, 8, 6, help="Restricted to top candidate priority clusters to ensure fast laptop simulator convergence.")

# Editable Available Stockpile for this Run
with st.expander("📦 Adjust Available Dispatch Stockpile for this Run", expanded=False):
    s_c1, s_c2, s_c3, s_c4, s_c5 = st.columns(5)
    with s_c1:
        run_teams = st.number_input("Rescue Teams", 1, 200, curr_inv["teams"])
    with s_c2:
        run_boats = st.number_input("Inflatable Boats", 1, 200, curr_inv["boats"])
    with s_c3:
        run_amb = st.number_input("Ambulances", 1, 200, curr_inv["ambulances"])
    with s_c4:
        run_food = st.number_input("Food Rations", 1, 500, curr_inv["food"])
    with s_c5:
        run_med = st.number_input("Medical Kits", 1, 500, curr_inv["medical_kits"])
        
    run_inventory = {
        "teams": run_teams,
        "boats": run_boats,
        "ambulances": run_amb,
        "food": run_food,
        "medical_kits": run_med
    }

st.write("")
run_opt_btn = st.button("🚀 Calculate Optimized Resource Allocation", type="primary", use_container_width=True)

if run_opt_btn:
    algo_key = "QAOA" if "QAOA" in algo_choice else "Classical"
    
    with st.spinner(f"Formulating QUBO problem and running {algo_key} optimization..."):
        results = execute_resource_allocation(
            inventory_counts=run_inventory,
            algorithm=algo_key,
            max_quantum_qubits=max_qubits,
            save_to_db=True
        )
        
    st.session_state["last_optimization_results"] = results
    st.success(f"✅ Optimization complete using {results['algorithm']}! Saved as Run #{results.get('run_id', 'N/A')} in SQLite.")

# Display results if available in session state
if "last_optimization_results" in st.session_state:
    res = st.session_state["last_optimization_results"]
    alloc_df = res["allocation_df"]
    
    st.markdown("---")
    st.subheader(f"📊 Allocation Summary — {res['algorithm']}")
    
    m_r1, m_r2, m_r3, m_r4 = st.columns(4)
    with m_r1:
        st.metric("Total Demanded Resources", f"{res['total_demand']} Units")
    with m_r2:
        st.metric("Total Resources Allocated", f"{res['total_resources_allocated']} Units", delta=f"{res['constraint_satisfaction_rate']}% Met")
    with m_r3:
        shortage = res["total_shortage"]
        delta_color = "normal" if shortage == 0 else "inverse"
        st.metric("Net Resource Shortage", f"{shortage} Units", delta=f"-{shortage} Unmet", delta_color=delta_color)
    with m_r4:
        st.metric("Execution Time", f"{res['execution_time_seconds']:.3f} s", delta=f"Objective: {res['objective_value']}")
        
    if "optimal_bitstring" in res:
        st.info(f"⚛️ **QAOA Optimal Quantum State:** Bitstring `|{res['optimal_bitstring']}⟩` (Max Probability: `{res.get('optimal_probability', 0.0):.4f}` across {len(res['optimal_bitstring'])} qubits)")

    # Grouped Bar Chart: Demand vs Availability vs Allocated
    st.markdown("##### 📈 Resource Demand vs Availability vs Allocation")
    demand_dict = {
        "Teams": int(alloc_df["teams_allocated"].sum() + (shortage // 5)),
        "Boats": int(alloc_df["boats_allocated"].sum() + (shortage // 5)),
        "Ambulances": int(alloc_df["ambulances_allocated"].sum() + (shortage // 5)),
        "Food": int(alloc_df["food_allocated"].sum() + (shortage // 5)),
        "Medical Kits": int(alloc_df["medical_kits_allocated"].sum() + (shortage // 5))
    }
    avail_dict = {
        "Teams": run_inventory["teams"],
        "Boats": run_inventory["boats"],
        "Ambulances": run_inventory["ambulances"],
        "Food": run_inventory["food"],
        "Medical Kits": run_inventory["medical_kits"]
    }
    allocated_dict = {
        "Teams": int(alloc_df["teams_allocated"].sum()),
        "Boats": int(alloc_df["boats_allocated"].sum()),
        "Ambulances": int(alloc_df["ambulances_allocated"].sum()),
        "Food": int(alloc_df["food_allocated"].sum()),
        "Medical Kits": int(alloc_df["medical_kits_allocated"].sum())
    }
    
    fig_comp = create_resource_comparison_bar(avail_dict, demand_dict, allocated_dict)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Detailed Location Allocation Table
    st.markdown("---")
    st.subheader("📋 Priority Location Allocation Plan")
    
    def highlight_shortage(row):
        if row["Unmet Demand?"] == 1:
            return ["background-color: #EF444422; color: #EF4444;"] * len(row)
        return [""] * len(row)
        
    view_df = alloc_df[[
        "priority_rank", "location_name", "taluk", "risk_level", "priority_score",
        "teams_allocated", "boats_allocated", "ambulances_allocated", "food_allocated", 
        "medical_kits_allocated", "unmet_demand_flag", "notes"
    ]].copy()
    
    view_df.columns = [
        "Rank", "Location", "Taluk", "Risk", "Priority Score",
        "Teams", "Boats", "Ambulances", "Food", "Medical Kits", "Unmet Demand?", "Dispatch Status"
    ]
    
    styled_alloc = view_df.style.apply(highlight_shortage, axis=1)
    st.dataframe(styled_alloc, use_container_width=True, height=450)
    
    # CSV Export Button
    csv_data = format_allocation_csv(alloc_df)
    st.download_button(
        label="📥 Download Allocation Plan as CSV",
        data=csv_data,
        file_name=f"Q_Rescue_Allocation_Plan_Run_{res.get('run_id', 'latest')}.csv",
        mime="text/csv",
        use_container_width=True
    )
