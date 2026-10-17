"""
Explainable AI (XAI) and Interpretability Page for Q-Rescue.
Deconstructs the Variational Quantum Classifier (VQC) circuit mechanics,
QAOA QUBO optimization trade-offs, and provides examiner-ready plain language explanations.
"""

import streamlit as st
import pandas as pd
from src.database import get_latest_observations
from src.authentication import render_auth_sidebar
from src.explainability import (
    get_vqc_circuit_explanation, 
    get_qaoa_explanation, 
    generate_natural_language_explanation
)
from src.risk_prediction import load_or_train_models
from src.visualizations import create_feature_importance_bar

st.set_page_config(page_title="Explainable AI | Q-Rescue", page_icon="💡", layout="wide")
render_auth_sidebar()

st.title("💡 Explainable AI & System Interpretability")
st.caption("Transparent mathematical mechanics, quantum circuit blueprints, and automated reasoning for examiners and incident commanders.")

tab_case, tab_vqc, tab_qaoa, tab_viva = st.tabs([
    "🔍 Locality Case Study", 
    "⚛️ VQC Circuit Mechanics", 
    "🎯 QAOA Optimization Rationale",
    "🎓 Examiner Viva Cheat-Sheet"
])

with tab_case:
    st.subheader("Auditable Locality Reasoning")
    st.markdown("Select any monitored Belagavi locality to inspect how the system derived its risk level and dispatch priority:")
    
    obs_df = get_latest_observations()
    if not obs_df.empty:
        loc_choices = [f"{r['location_name']} ({r['taluk']})" for _, r in obs_df.iterrows()]
        sel_loc_str = st.selectbox("Choose Locality", loc_choices)
        clean_name = sel_loc_str.split(" (")[0]
        sel_row = obs_df[obs_df["location_name"] == clean_name].iloc[0]
        
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("##### 📋 Observed Telemetry Values")
            st.write(f"- **Locality ID:** `{sel_row.get('location_id')}`")
            st.write(f"- **Taluk:** `{sel_row.get('taluk')}`")
            st.write(f"- **Rainfall (24h):** `{sel_row.get('rainfall_mm')} mm`")
            st.write(f"- **River Water Gauge:** `{sel_row.get('river_water_level_m')} m`")
            st.write(f"- **Terrain Elevation:** `{sel_row.get('elevation_m')} m`")
            st.write(f"- **Historical Flood Prone:** `{'Yes' if sel_row.get('historical_flood_indicator') == 1 else 'No'}`")
            st.write(f"- **Exposed Population:** `{sel_row.get('population'):,}`")
            st.write(f"- **Distance to Base:** `{sel_row.get('distance_to_rescue_base_km')} km`")
            
        with c_right:
            st.markdown("##### 📝 Plain-English Assessment")
            explanation_text = generate_natural_language_explanation(sel_row)
            st.markdown(explanation_text)

with tab_vqc:
    st.subheader("Variational Quantum Classifier (VQC) Deep Dive")
    vqc_info = get_vqc_circuit_explanation()
    
    st.markdown(f"### {vqc_info['architecture_title']}")
    st.info(vqc_info["examiner_takeaway"])
    
    st.markdown("#### 1. Feature State Encoding: Angle Embedding")
    st.markdown(vqc_info["feature_encoding"]["description"])
    
    enc_df = pd.DataFrame(vqc_info["feature_encoding"]["qubit_mapping"])
    st.dataframe(enc_df, use_container_width=True)
    
    st.markdown("#### 2. Variational Quantum Ansatz (Strongly Entangling Layers)")
    st.markdown(f"**Gate Sequence:** `{vqc_info['ansatz']['gates']}`")
    st.markdown(vqc_info["ansatz"]["purpose"])
    
    st.markdown("""
    ```text
    |0> ──[RY(x0)]──[RY(θ0)]──[RZ(θ1)]──●──────────────[X]── ... ──[Measure <Z0>] -> Low Risk
    |0> ──[RY(x1)]──[RY(θ2)]──[RZ(θ3)]──|───●─────────────── ... ──[Measure <Z1>] -> Medium Risk
    |0> ──[RY(x2)]──[RY(θ4)]──[RZ(θ5)]──|───|───●─────────── ... ──[Measure <Z2>] -> High Risk
    |0> ──[RY(x3)]──[RY(θ6)]──[RZ(θ7)]──X───|───|───●─────── ... ──[Measure <Z3>] -> Critical Risk
                                           [CNOT Entanglers]
    ```
    """)
    
    st.markdown("#### 3. Measurement & Multi-Class Readout")
    readout_df = pd.DataFrame(vqc_info["measurement_and_readout"]["class_mapping"])
    st.dataframe(readout_df, use_container_width=True)
    st.caption(f"Softmax transformation: {vqc_info['measurement_and_readout']['softmax_layer']}")

with tab_qaoa:
    st.subheader("QAOA Optimization Formulation & Trade-offs")
    qaoa_info = get_qaoa_explanation()
    
    st.markdown(f"### {qaoa_info['algorithm_title']}")
    st.info(qaoa_info["examiner_takeaway"])
    
    st.markdown("#### 1. QUBO Formulation (Quadratic Unconstrained Binary Optimization)")
    st.latex(r"\max_{\vec{x}} \sum_{i=1}^N \left( \alpha \cdot w_i + \beta \cdot \hat{P}_i - \gamma \cdot \hat{D}_i \right) x_i \quad \text{s.t.} \quad \sum_{i=1}^N d_{i, k} x_i \le B_k \quad \forall k")
    st.markdown("Transformed with quadratic penalty $\lambda$ for capacity budget violations:")
    st.latex(r"\min_{\vec{x}} E(\vec{x}) = - \sum_{i=1}^N \text{Utility}_i x_i + \sum_k \lambda_k \left( \sum_{i=1}^N d_{i, k} x_i - B_k \right)^2")
    
    st.markdown("#### 2. Mapping to Ising Hamiltonian")
    st.markdown("Spin operator transformation: $x_i = \frac{I - Z_i}{2}$")
    st.latex(r"H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{i j} Z_i Z_j, \qquad H_M = \sum_{i=1}^N X_i")
    
    st.markdown("#### 3. Multi-Factor Priority Weightings")
    tradeoffs_df = pd.DataFrame(qaoa_info["tradeoff_factors"])
    st.dataframe(tradeoffs_df, use_container_width=True)

with tab_viva:
    st.subheader("🎓 Viva Voce & Evaluation Defense Cheat-Sheet")
    
    st.markdown("""
    **Q1: Why use VQC instead of standard deep neural networks?**  
    *Answer:* VQC leverages the vast Hilbert space ($2^N$ dimensions with $N$ qubits) using parameterized rotation gates and entanglement. It provides a compact model structure with far fewer parameters than deep neural nets, exploring quantum-enhanced feature representations suitable for complex non-linear boundary separation in disaster physics.
    
    ---
    
    **Q2: How does QAOA ensure we don't dispatch more boats or teams than are available in Belagavi?**  
    *Answer:* In our QUBO formulation, capacity constraints $\sum d_{ik} x_i \le B_k$ are converted into quadratic penalty terms $\lambda (\sum d_{ik} x_i - B_k)^2$ inside the cost Hamiltonian $H_C$. Any configuration that exceeds depot inventory incurs a steep energy penalty, driving the quantum state $|\psi(\gamma, \beta)\rangle$ away from infeasible deployment vectors.
    
    ---
    
    **Q3: Are you running on real quantum hardware or a simulator?**  
    *Answer (Academic Honesty):* We run on high-performance state-vector quantum simulators via PennyLane (`default.qubit` and `pennylane-lightning`). Current Noisy Intermediate-Scale Quantum (NISQ) hardware requires paid cloud access and has qubit noise. Our code is fully native PennyLane and can target real IBM Quantum or Rigetti QPUs without altering circuit logic.
    
    ---
    
    **Q4: How was Belagavi district data incorporated?**  
    *Answer:* The system uses verified geographical coordinates and elevations across 12 Belagavi taluks (Chikkodi, Gokak, Athani, Raybag, Khanapur, Ramdurg, etc.) with real distances to the Belagavi District Emergency Operations Centre (DEOC), alongside calibrated monsoon hydrological observations reflecting the Krishna and Ghataprabha river basins.
    """)
