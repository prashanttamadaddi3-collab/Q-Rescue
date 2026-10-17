"""
Flood Risk Prediction module for Q-Rescue.
Executes Variational Quantum Classifier (VQC) and Classical Random Forest predictions,
displaying confidence scores, class probabilities, and raw quantum Pauli-Z state readouts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.database import get_all_locations, get_latest_observations
from src.authentication import render_auth_sidebar, require_auth
from src.risk_prediction import predict_single_location, predict_flood_risk_batch
from src.explainability import generate_natural_language_explanation
from src.visualizations import RISK_COLORS

st.set_page_config(page_title="Risk Prediction | Q-Rescue", page_icon="🔮", layout="wide")
render_auth_sidebar()

st.title("🔮 Intelligent Flood Risk Prediction")
st.caption("Variational Quantum Classifier (VQC) with 4-Qubit Parameterized Circuit & Classical Baseline Comparison.")

tab_single, tab_batch = st.tabs(["🎯 Single Location Inference", "⚡ Batch Prediction on Monitored Stations"])

with tab_single:
    st.subheader("Interactive Hazard Assessment")
    st.markdown("Select an existing Belagavi locality or adjust environmental telemetry sliders to test the models:")
    
    locations_df = get_all_locations()
    obs_df = get_latest_observations()
    
    col_pre, col_model = st.columns([2, 1])
    with col_pre:
        loc_options = ["Custom Input"] + [f"{r['location_name']} ({r['taluk']})" for _, r in locations_df.iterrows()]
        selected_loc_name = st.selectbox("Load Station Preset", loc_options)
        
    with col_model:
        model_choice = st.radio(
            "Prediction Engine",
            ["VQC (Quantum Machine Learning)", "Classical Random Forest Baseline"],
            help="VQC uses a 4-qubit parameterized quantum circuit with AngleEmbedding; Random Forest uses 8 classical features."
        )

    # Prepopulate values if preset chosen
    default_rainfall = 145.0
    default_wl = 7.8
    default_elev = 540.0
    default_hist = 1
    default_temp = 24.5
    default_hum = 92.0
    default_pop = 15000
    default_dist = 65.0
    selected_name_clean = "Custom Test Site"
    selected_taluk_clean = "Belagavi"
    
    if selected_loc_name != "Custom Input" and not obs_df.empty:
        clean_name = selected_loc_name.split(" (")[0]
        matched = obs_df[obs_df["location_name"] == clean_name]
        if not matched.empty:
            m_row = matched.iloc[0]
            default_rainfall = float(m_row.get("rainfall_mm", 120.0))
            default_wl = float(m_row.get("river_water_level_m", 6.5))
            default_elev = float(m_row.get("elevation_m", 550.0))
            default_hist = int(m_row.get("historical_flood_indicator", 1))
            default_temp = float(m_row.get("temperature_c", 25.0))
            default_hum = float(m_row.get("humidity_percent", 85.0))
            default_pop = int(m_row.get("population", 15000))
            default_dist = float(m_row.get("distance_to_rescue_base_km", 60.0))
            selected_name_clean = m_row.get("location_name")
            selected_taluk_clean = m_row.get("taluk")

    with st.form("single_prediction_form"):
        st.markdown("##### 📡 Environmental & Terrain Telemetry")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            inp_rf = st.slider("24-Hour Cumulative Rainfall (mm)", 0.0, 350.0, default_rainfall, 1.0)
            inp_wl = st.slider("River Gauge Water Level (m)", 0.0, 16.0, default_wl, 0.1)
            inp_elev = st.slider("Terrain Elevation (m AMSL)", 450.0, 900.0, default_elev, 5.0)
            inp_hist = st.radio("Historical Flood Prone Area?", [1, 0], index=0 if default_hist == 1 else 1, format_func=lambda x: "Yes (Recurrent Inundation)" if x == 1 else "No (Stable)")
        with p_c2:
            inp_temp = st.slider("Ambient Temperature (°C)", 15.0, 42.0, default_temp, 0.5)
            inp_hum = st.slider("Relative Humidity (%)", 30.0, 100.0, default_hum, 1.0)
            inp_pop = st.number_input("Exposed Locality Population", min_value=500, max_value=250000, value=default_pop, step=500)
            inp_dist = st.slider("Distance to Belagavi DEOC Base (km)", 1.0, 160.0, default_dist, 1.0)
            
        submit_pred = st.form_submit_button("⚡ Execute Model Inference", use_container_width=True)

    if submit_pred:
        model_type_key = "VQC" if "VQC" in model_choice else "Classical"
        
        sample_input = {
            "location_id": "PRED-SAMPLE",
            "location_name": selected_name_clean,
            "taluk": selected_taluk_clean,
            "rainfall_mm": inp_rf,
            "river_water_level_m": inp_wl,
            "elevation_m": inp_elev,
            "historical_flood_indicator": inp_hist,
            "temperature_c": inp_temp,
            "humidity_percent": inp_hum,
            "population": inp_pop,
            "distance_to_rescue_base_km": inp_dist
        }
        
        with st.spinner("Running quantum circuit simulation..." if model_type_key == "VQC" else "Computing classical model prediction..."):
            pred_result = predict_single_location(sample_input, model_type=model_type_key)

        pred_label = pred_result["predicted_risk_label"]
        conf = pred_result["confidence"]
        badge_color = RISK_COLORS.get(pred_label, "#3B82F6")

        st.markdown("---")
        st.subheader("🔍 Prediction Results")
        
        r_c1, r_c2 = st.columns([1, 2])
        with r_c1:
            st.markdown(f"""
            <div style="background-color: #1E293B; border: 2px solid {badge_color}; border-radius: 8px; padding: 1.5rem; text-align: center;">
                <h4 style="margin: 0; color: #94A3B8;">Predicted Flood Hazard</h4>
                <h1 style="color: {badge_color}; margin: 8px 0;">{pred_label.upper()}</h1>
                <p style="margin: 0; color: #F8FAFC; font-size: 1.1rem;">Model Confidence: <b>{conf * 100:.1f}%</b></p>
                <span style="font-size: 0.8rem; color: #94A3B8;">Engine: {pred_result.get('model_used')}</span>
            </div>
            """, unsafe_allow_html=True)
            
            if model_type_key == "VQC" and "raw_pauli_z_expectations" in pred_result:
                st.markdown("##### ⚛️ Raw Quantum Expectation Values")
                st.caption(r"Measurement $\langle Z_i \rangle \in [-1, +1]$ for each qubit observable:")
                for q_name, val in pred_result["raw_pauli_z_expectations"].items():
                    st.text(f"{q_name}: {val:+.4f}")

        with r_c2:
            st.markdown("##### 📊 Class Probability Distribution")
            probs_data = pred_result["class_probabilities"]
            df_prob = pd.DataFrame({
                "Risk Class": list(probs_data.keys()),
                "Probability": list(probs_data.values())
            })
            fig_prob = px.bar(
                df_prob, x="Risk Class", y="Probability",
                color="Risk Class", color_discrete_map=RISK_COLORS,
                range_y=[0, 1.0], text="Probability"
            )
            fig_prob.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            fig_prob.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1E293B",
                plot_bgcolor="#0F172A",
                font={"color": "#F8FAFC"},
                showlegend=False,
                margin={"l": 20, "r": 20, "t": 30, "b": 20}
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Automated Reasoning & Interpretability")
        explanation = generate_natural_language_explanation(pd.Series(sample_input | {"predicted_risk_level": pred_label}))
        st.markdown(explanation)

with tab_batch:
    st.subheader("Batch Model Inference on All Monitored Stations")
    st.markdown("Execute batch inference across all current Belagavi monitoring stations and synchronize predictions to the database:")
    
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        batch_model_type = st.radio("Select Model for Batch Inference", ["VQC (Variational Quantum Classifier)", "Classical Random Forest Baseline"], key="batch_model_radio")
    with col_b2:
        st.write("")
        st.write("")
        run_batch_btn = st.button("🚀 Run Full District Batch Prediction", use_container_width=True)
        
    if run_batch_btn:
        model_code = "VQC" if "VQC" in batch_model_type else "Classical"
        obs_current = get_latest_observations()
        
        with st.spinner(f"Computing batch predictions using {model_code}..."):
            batch_results = predict_flood_risk_batch(obs_current, model_type=model_code, save_to_db=True)
            
        st.success(f"✅ Successfully evaluated and saved {len(batch_results)} station predictions via {model_code}!")
        
        display_sub = batch_results[[
            "location_id", "location_name", "taluk", "predicted_risk_level", 
            "confidence_score", "rainfall_mm", "river_water_level_m", "population"
        ]]
        st.dataframe(display_sub, use_container_width=True)
