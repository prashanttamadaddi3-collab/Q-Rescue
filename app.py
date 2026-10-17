"""
Q-Rescue: Hybrid Quantum–Classical Framework for Flood Risk Prediction
and Emergency Resource Allocation using VQC and QAOA.

Main Entry Point and Landing Page for the Streamlit Application.
APS College of Engineering, Bengaluru — Department of Information Science and Engineering.
"""

import os
import streamlit as st
import pandas as pd
from PIL import Image

# Initialize database on startup
from src.database import init_db, seed_initial_data, get_all_locations, get_latest_observations
from src.authentication import (
    init_auth_state, login_user_session, logout_user_session, 
    authenticate_user, register_user, get_current_user, render_auth_sidebar
)

# Page configuration
st.set_page_config(
    page_title="Q-Rescue | Hybrid Quantum-Classical Disaster Support",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure database and seeds
init_db()
seed_initial_data()
init_auth_state()

# Custom CSS for disaster-management aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #38BDF8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .feature-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover {
        border-color: #0284C7;
        transform: translateY(-2px);
    }
    .metric-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .academic-footer {
        margin-top: 3rem;
        padding: 1.5rem;
        background-color: #1E293B;
        border-top: 2px solid #0284C7;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

# Render User Profile / Logout in Sidebar
render_auth_sidebar()

# Header with College & Project Branding
col_logo, col_header = st.columns([1, 5])
with col_logo:
    aps_logo_path = os.path.join(os.path.dirname(__file__), "assets", "aps_logo.jpg")
    if os.path.exists(aps_logo_path):
        st.image(aps_logo_path, width=120)
    else:
        st.markdown("# 🌊")

with col_header:
    st.markdown('<div class="main-title">Q-RESCUE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Hybrid Quantum–Classical Framework for Flood Risk Prediction and Emergency Resource Allocation using VQC and QAOA</div>', 
        unsafe_allow_html=True
    )
    st.caption("🏫 **APS College of Engineering, Bengaluru** | Department of Information Science & Engineering")

st.markdown("---")

# Quick Status Banner
obs_df = get_latest_observations()
total_monitored = len(obs_df)
crit_count = len(obs_df[obs_df["flood_risk_label"] == "Critical"]) if "flood_risk_label" in obs_df.columns else 0
high_count = len(obs_df[obs_df["flood_risk_label"] == "High"]) if "flood_risk_label" in obs_df.columns else 0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
    <div class="metric-box">
        <h4 style="margin:0; color:#94A3B8;">Monitored Belagavi Stations</h4>
        <h2 style="margin:4px 0; color:#38BDF8;">{total_monitored}</h2>
        <span style="font-size:0.8rem; color:#64748B;">Across 12 Taluks</span>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-box">
        <h4 style="margin:0; color:#94A3B8;">Critical Emergency Areas</h4>
        <h2 style="margin:4px 0; color:#EF4444;">{crit_count}</h2>
        <span style="font-size:0.8rem; color:#EF4444;">Immediate Evacuation</span>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-box">
        <h4 style="margin:0; color:#94A3B8;">High Alert Locations</h4>
        <h2 style="margin:4px 0; color:#F97316;">{high_count}</h2>
        <span style="font-size:0.8rem; color:#F97316;">Pre-emptive Readiness</span>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    curr_user = get_current_user()
    status_text = f"Logged in as {curr_user['username']}" if curr_user else "Guest / Observer"
    st.markdown(f"""
    <div class="metric-box">
        <h4 style="margin:0; color:#94A3B8;">Operational Status</h4>
        <h2 style="margin:4px 0; color:#10B981;">Active</h2>
        <span style="font-size:0.8rem; color:#94A3B8;">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Layout: Main Overview (left) + Auth / Quick Access (right)
col_main, col_auth = st.columns([3, 2], gap="large")

with col_main:
    st.markdown("### 📌 About the System")
    st.markdown("""
    **Q-Rescue** addresses the computational bottlenecks of disaster management by coupling 
    **Variational Quantum Classification (VQC)** for high-dimensional environmental pattern recognition 
    with **Quantum Approximate Optimization (QAOA)** for multi-constrained emergency resource dispatch.
    
    The initial deployment focuses on **Belagavi District, Karnataka**, covering flood-prone stretches along 
    the **Krishna, Ghataprabha, Malaprabha, and Dudhganga** river basins (Chikkodi, Gokak, Athani, Raybag, Khanapur).
    """)
    
    st.markdown("### ⚡ Core Capabilities")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #38BDF8; margin-top:0;">🔮 Variational Quantum Classifier</h4>
            <p style="font-size:0.88rem; color:#CBD5E1; margin-bottom:0;">
                Encodes rainfall, water levels, elevation, and terrain into a 4-qubit parameterized circuit 
                to predict risk categories: <b>Low, Medium, High, or Critical</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #38BDF8; margin-top:0;">🗺️ Interactive GIS Mapping</h4>
            <p style="font-size:0.88rem; color:#CBD5E1; margin-bottom:0;">
                Real-world geospatial coordinates of Belagavi taluks and villages with color-coded risk markers, 
                warning levels, and central rescue base telemetry.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #38BDF8; margin-top:0;">⚛️ QAOA Resource Optimization</h4>
            <p style="font-size:0.88rem; color:#CBD5E1; margin-bottom:0;">
                Maps rescue teams, boats, ambulances, and food supplies into a Quadratic Unconstrained Binary 
                Optimization (QUBO) problem solved via simulated quantum annealing.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #38BDF8; margin-top:0;">💡 Transparent Explainable AI</h4>
            <p style="font-size:0.88rem; color:#CBD5E1; margin-bottom:0;">
                Quantum circuit state readout, classical feature importances, and plain-language reasoning 
                for college viva examination and responder verification.
            </p>
        </div>
        """, unsafe_allow_html=True)

with col_auth:
    st.markdown("### 🔐 Responder Access Portal")
    
    user = get_current_user()
    if user:
        st.success(f"✅ Active Session: **{user.get('full_name') or user.get('username')}**")
        st.info(f"**Assigned Role:** {user.get('role')} | **Email:** {user.get('email', 'N/A')}")
        st.markdown("""
        You have authorized access to all emergency dispatch modules. Use the sidebar or quick links below:
        """)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📊 Open Dashboard", use_container_width=True):
                st.switch_page("pages/1_📊_Dashboard.py")
            if st.button("🗺️ View Risk Map", use_container_width=True):
                st.switch_page("pages/5_🗺️_Risk_Map.py")
        with col_btn2:
            if st.button("⚛️ Run QAOA Allocation", use_container_width=True):
                st.switch_page("pages/4_⚛️_QAOA_Optimization.py")
            if st.button("🔮 Predict Flood Risk", use_container_width=True):
                st.switch_page("pages/2_🔮_Risk_Prediction.py")
                
        if st.button("🚪 Sign Out", key="main_logout", use_container_width=True):
            logout_user_session()
    else:
        tab_login, tab_register = st.tabs(["🔑 Sign In", "📝 Register New Account"])
        
        with tab_login:
            st.markdown("<span style='font-size:0.85rem; color:#94A3B8;'>Sign in with your responder credentials:</span>", unsafe_allow_html=True)
            with st.form("login_form"):
                login_user = st.text_input("Username", placeholder="e.g. responder1")
                login_pass = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("🚀 Log In", use_container_width=True)
                
                if submit_login:
                    success, u_data, msg = authenticate_user(login_user, login_pass)
                    if success and u_data:
                        login_user_session(u_data)
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error(msg)
                        
            st.caption("💡 **Demo Credentials:** Register any username & password (min 6 chars) or enter a registered account.")

        with tab_register:
            st.markdown("<span style='font-size:0.85rem; color:#94A3B8;'>Create a secure disaster responder account:</span>", unsafe_allow_html=True)
            with st.form("reg_form"):
                reg_name = st.text_input("Full Name", placeholder="e.g. Prashanth T")
                reg_user = st.text_input("Username (alphanumeric, min 3 chars)", placeholder="e.g. prashanth_t")
                reg_email = st.text_input("Email Address", placeholder="responder@apsce.edu.in")
                reg_pass = st.text_input("Password (min 6 chars)", type="password")
                reg_role = st.selectbox("Operational Role", ["Disaster Response Officer", "Incident Commander", "Field Rescue Lead", "Researcher"])
                submit_reg = st.form_submit_button("📝 Register", use_container_width=True)
                
                if submit_reg:
                    success, msg = register_user(reg_user, reg_pass, full_name=reg_name, email=reg_email, role=reg_role)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

# Academic Project Footer
st.markdown("""
<div class="academic-footer">
    <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
        <div>
            <b>Major Project:</b> Q-Rescue: Hybrid Quantum–Classical Framework for Flood Risk Prediction and Resource Allocation<br/>
            <b>Institution:</b> APS College of Engineering, Bengaluru | Dept. of Information Science & Engineering<br/>
            <b>Project Guide:</b> Dr. Krushitha Shetty, Associate Professor, Dept. of ISE, APSCE<br/>
            <b>HOD / Coordinator:</b> Dr. Mithun B N, Professor & Head, Dept. of ISE, APSCE
        </div>
        <div style="text-align: right; margin-top: 6px;">
            <b>Project Team (ISE):</b><br/>
            Prashanth T (1AP23IS039) | Rajesh Naik (1AP23IS042)<br/>
            Siddaram Masali (1AP23IS045) | Srujan Chandra K R (1AP23IS050)<br/>
            <span style="color: #38BDF8;">Academic Decision-Support Prototype</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
