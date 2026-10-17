"""
About Page for Q-Rescue Academic Major Project.
Presents institution details, guide, HOD, student team, architecture diagrams,
technology stack justification, and literature citations.
"""

import os
import streamlit as st
from PIL import Image
from src.authentication import render_auth_sidebar

st.set_page_config(page_title="About the Project | Q-Rescue", page_icon="ℹ️", layout="wide")
render_auth_sidebar()

st.title("ℹ️ Academic Project Details — Q-Rescue")
st.caption("College Major Project | Visvesvaraya Technological University (VTU) | APSCE Bengaluru")

col_l1, col_l2 = st.columns([1, 4])
with col_l1:
    aps_logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "aps_logo.jpg")
    vtu_logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "vtu_logo.png")
    if os.path.exists(aps_logo_path):
        st.image(aps_logo_path, width=130)
    if os.path.exists(vtu_logo_path):
        st.image(vtu_logo_path, width=110)

with col_l2:
    st.markdown("""
    ### **APS College of Engineering**
    *(Affiliated to Visvesvaraya Technological University, Belagavi | Approved by AICTE, New Delhi)*  
    **Anantha Gnana Gangothri Campus, Kanakapura Road, Somanahalli, Bengaluru – 560116**  
    #### **Department of Information Science and Engineering**
    """)

st.markdown("---")

col_team1, col_team2 = st.columns(2)

with col_team1:
    st.markdown("""
    #### 👨‍🏫 Project Guidance & Leadership
    - **Project Guide:**  
      **Dr. Krushitha Shetty**, Associate Professor, Dept. of Information Science & Engineering, APSCE.
    - **Coordinator & Head of Department:**  
      **Dr. Mithun B N**, Professor & Head, Dept. of Information Science & Engineering, APSCE.
    """)

with col_team2:
    st.markdown("""
    #### 👥 Student Project Team
    | Name | USN | Department |
    | :--- | :--- | :--- |
    | **Prashanth T** | `1AP23IS039` | Information Science & Engg |
    | **Rajesh Naik** | `1AP23IS042` | Information Science & Engg |
    | **Siddaram Masali** | `1AP23IS045` | Information Science & Engg |
    | **Srujan Chandra K R** | `1AP23IS050` | Information Science & Engg |
    """)

st.markdown("---")

# Architectural Diagrams from Attached Presentation
st.subheader("🏛️ System Architecture & Engineering Blueprints")

tab_diag1, tab_diag2, tab_diag3 = st.tabs(["System Architecture / Block Diagram", "Data Flow Diagram", "Sequence Diagram"])

assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

with tab_diag1:
    arch_img = os.path.join(assets_dir, "system_architecture.png")
    if os.path.exists(arch_img):
        st.image(arch_img, caption="System Architecture / Block Diagram (From Project Presentation)", use_container_width=True)
    else:
        st.info("System architecture diagram.")

with tab_diag2:
    dfd_img = os.path.join(assets_dir, "data_flow.png")
    if os.path.exists(dfd_img):
        st.image(dfd_img, caption="Data Flow Diagram (From Project Presentation)", use_container_width=True)
    else:
        st.info("Data flow diagram.")

with tab_diag3:
    seq_img = os.path.join(assets_dir, "sequence_diagram.png")
    if os.path.exists(seq_img):
        st.image(seq_img, caption="Sequence Diagram (From Project Presentation)", use_container_width=True)
    else:
        st.info("Sequence diagram.")

st.markdown("---")

st.subheader("📚 Literature Survey & Theoretical References")
st.markdown("""
1. **Havlíček et al. (2019)** – *Supervised learning with quantum-enhanced feature spaces*, Nature, 567(7747), 209-212.
2. **Farhi, Goldstone, & Gutmann (2014)** – *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028.
3. **Mosavi et al. (2018)** – *Flood Prediction Using Machine Learning Models: Literature Review*, Water, 10(11), 1536.
4. **Kratzert et al. (2018)** – *Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks*, HESS, 22(11), 6005-6022.
5. **Zhang et al. (2017)** – *Assignment optimization for emergency rescue teams under uncertain conditions*, Knowledge-Based Systems, 120, 170-182.
6. **Wang et al. (2022)** – *Emergency Resource Allocation and Route Planning under Natural Disasters*, IJERPH, 19(17), 10839.
7. **Grzesiak & Thakkar (2024)** – *Flood Prediction Using Classical and Quantum Machine Learning Models*, IEEE Access.
""")

st.markdown("---")

st.markdown("""
<div style="background-color: #1E293B; border-left: 4px solid #0284C7; padding: 14px; border-radius: 4px; font-size: 0.9rem; color: #CBD5E1;">
    ⚠️ <b>Academic Disclaimer:</b> Q-Rescue is an academic engineering decision-support prototype developed for final year Major Project evaluation 
    at APS College of Engineering. It is not an officially certified government flood-warning authority or real-time emergency service.
</div>
""", unsafe_allow_html=True)
