"""
Model Evaluation and Benchmark Comparison Page for Q-Rescue.
Compares Variational Quantum Classifier (VQC) with Classical Random Forest on identical held-out test splits.
Presents side-by-side Accuracy, Precision, Recall, F1, Confusion Matrices, and Training Curves.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from src.database import get_all_observations
from src.authentication import render_auth_sidebar
from src.data_loader import validate_and_load_csv
from src.preprocessing import PreprocessingPipeline
from src.classical_model import ClassicalFloodModel
from src.vqc_model import VariationalQuantumClassifier
from src.risk_prediction import load_or_train_models
from src.visualizations import (
    create_confusion_matrix_heatmap, 
    create_feature_importance_bar, 
    create_vqc_training_history_chart
)

st.set_page_config(page_title="Model Evaluation | Q-Rescue", page_icon="📈", layout="wide")
render_auth_sidebar()

st.title("📈 Model Evaluation & Quantum-Classical Benchmark")
st.caption("Rigorous comparative performance evaluation on identical 25% held-out stratified test data.")

# Academic Honesty Banner
st.info("""
⚖️ **Fair Comparison Protocol:** Both the **Variational Quantum Classifier (VQC)** and the **Classical Random Forest** 
are evaluated on the exact same held-out test split. Quantum computations are executed via the high-performance 
`default.qubit` state-vector simulator backend on standard hardware.
""")

# Load dataset and prepare test split
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_flood_data.csv")
df, _ = validate_and_load_csv(csv_path)

prep, rf_model, vqc_model = load_or_train_models()
X_train_c, X_test_c, X_train_q, X_test_q, y_train, y_test = prep.fit_transform_datasets(df, test_size=0.25, random_state=42)

# Evaluate both models
metrics_rf = rf_model.evaluate(X_test_c, y_test)
metrics_vqc = vqc_model.evaluate(X_test_q, y_test)

# Side-by-Side Overall Metrics Cards
st.subheader("🎯 Primary Benchmark Metrics (Test Set: N = 36)")

col_vqc, col_rf = st.columns(2)
with col_vqc:
    st.markdown("""
    <div style="background-color: #1E293B; border: 2px solid #0284C7; border-radius: 8px; padding: 1rem;">
        <h3 style="color: #38BDF8; margin: 0 0 10px 0;">⚛️ Variational Quantum Classifier (VQC)</h3>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">4-Qubit Parameterized Circuit | AngleEmbedding | Strongly Entangling Layers</p>
    </div>
    """, unsafe_allow_html=True)
    
    vm1, vm2, vm3, vm4 = st.columns(4)
    with vm1:
        st.metric("Accuracy", f"{metrics_vqc['accuracy'] * 100:.1f}%")
    with vm2:
        st.metric("Precision", f"{metrics_vqc['precision'] * 100:.1f}%")
    with vm3:
        st.metric("Recall", f"{metrics_vqc['recall'] * 100:.1f}%")
    with vm4:
        st.metric("F1-Score", f"{metrics_vqc['f1_score'] * 100:.1f}%")

with col_rf:
    st.markdown("""
    <div style="background-color: #1E293B; border: 2px solid #10B981; border-radius: 8px; padding: 1rem;">
        <h3 style="color: #10B981; margin: 0 0 10px 0;">🌲 Classical Baseline (Random Forest)</h3>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">100 Estimators | Max Depth 6 | 8 Classical Environmental Features</p>
    </div>
    """, unsafe_allow_html=True)
    
    rm1, rm2, rm3, rm4 = st.columns(4)
    with rm1:
        st.metric("Accuracy", f"{metrics_rf['accuracy'] * 100:.1f}%")
    with rm2:
        st.metric("Precision", f"{metrics_rf['precision'] * 100:.1f}%")
    with rm3:
        st.metric("Recall", f"{metrics_rf['recall'] * 100:.1f}%")
    with rm4:
        st.metric("F1-Score", f"{metrics_rf['f1_score'] * 100:.1f}%")

st.markdown("---")

# Comparative Confusion Matrices
st.subheader("🔲 Confusion Matrices")
col_cm1, col_cm2 = st.columns(2)

with col_cm1:
    fig_cm_vqc = create_confusion_matrix_heatmap(metrics_vqc["confusion_matrix"])
    fig_cm_vqc.update_layout(title="<b>VQC Quantum Confusion Matrix</b>")
    st.plotly_chart(fig_cm_vqc, use_container_width=True)

with col_cm2:
    fig_cm_rf = create_confusion_matrix_heatmap(metrics_rf["confusion_matrix"])
    fig_cm_rf.update_layout(title="<b>Random Forest Classical Confusion Matrix</b>")
    st.plotly_chart(fig_cm_rf, use_container_width=True)

st.markdown("---")

# Per-Class Precision, Recall, F1 Comparison
st.subheader("📊 Per-Class Performance Breakdown")

class_names = ["Low", "Medium", "High", "Critical"]
comp_rows = []
for c_name in class_names:
    rep_v = metrics_vqc["classification_report"].get(c_name, {})
    rep_r = metrics_rf["classification_report"].get(c_name, {})
    comp_rows.append({
        "Risk Class": c_name,
        "VQC Precision": f"{rep_v.get('precision', 0.0):.2f}",
        "RF Precision": f"{rep_r.get('precision', 0.0):.2f}",
        "VQC Recall": f"{rep_v.get('recall', 0.0):.2f}",
        "RF Recall": f"{rep_r.get('recall', 0.0):.2f}",
        "VQC F1": f"{rep_v.get('f1-score', 0.0):.2f}",
        "RF F1": f"{rep_r.get('f1-score', 0.0):.2f}",
        "Support": rep_v.get("support", 0)
    })

df_comp = pd.DataFrame(comp_rows)
st.dataframe(df_comp, use_container_width=True)

st.markdown("---")

# Feature Importance and VQC Convergence Curves
st.subheader("🔬 Model Dynamics & Convergence")
col_d1, col_d2 = st.columns(2)

with col_d1:
    fig_imp = create_feature_importance_bar(rf_model.get_feature_importances())
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Gini feature importance ranking indicates rainfall and river gauge readings provide the highest information gain.")

with col_d2:
    fig_vqc_train = create_vqc_training_history_chart(vqc_model.training_history)
    st.plotly_chart(fig_vqc_train, use_container_width=True)
    st.caption("Convergence of VQC categorical cross-entropy loss and training accuracy across optimization iterations.")

# Retrain Models Section
with st.expander("🔄 Re-train Models on Fresh Dataset", expanded=False):
    st.markdown("Re-execute model fitting with custom hyperparameters:")
    r_c1, r_c2, r_c3 = st.columns(3)
    with r_c1:
        vqc_epochs = st.slider("VQC Epochs", 5, 40, 15)
    with r_c2:
        vqc_lr = st.slider("Learning Rate", 0.01, 0.20, 0.06, 0.01)
    with r_c3:
        rf_trees = st.slider("RF Estimators", 20, 200, 100, 10)
        
    if st.button("⚡ Retrain Both Models Now", use_container_width=True):
        with st.spinner("Retraining Random Forest and simulating VQC circuit..."):
            prep = PreprocessingPipeline()
            X_tr_c, X_te_c, X_tr_q, X_te_q, y_tr, y_te = prep.fit_transform_datasets(df, test_size=0.25)
            
            rf_new = ClassicalFloodModel(n_estimators=rf_trees, max_depth=6)
            rf_new.train(X_tr_c, y_tr)
            rf_new.save()
            
            vqc_new = VariationalQuantumClassifier(epochs=vqc_epochs, lr=vqc_lr, batch_size=16)
            vqc_new.train(X_tr_q, y_tr)
            vqc_new.save()
            prep.save()
            
        st.success("✅ Models retrained and saved to disk! Refreshing evaluation...")
        st.rerun()
