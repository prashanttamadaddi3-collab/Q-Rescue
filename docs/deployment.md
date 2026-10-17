# Deployment & Operation Guide — Q-Rescue

**Deployment onto Streamlit Community Cloud & Local Execution**

---

## 1. Local Development & Testing Instructions

### 1.1 Prerequisites
- Python 3.10 to 3.13 installed.
- Git installed.
- Internet connectivity for package installation and OpenStreetMap tile loading.

### 1.2 Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/Q-Rescue.git
cd Q-Rescue

# 2. Create and activate a clean virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run the automated test suite
pytest tests/ -v

# 5. Launch the Streamlit application
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`.

---

## 2. Streamlit Community Cloud Deployment Guide

Streamlit Community Cloud provides free hosting for Streamlit applications directly linked to a public GitHub repository.

### Step 1: Push Code to GitHub
1. Create a new public repository on GitHub named `Q-Rescue` (or your preferred name).
2. Initialize git and push:
```bash
git init
git add .
git commit -m "Initial release of Q-Rescue: Hybrid VQC and QAOA disaster support system"
git branch -M main
git remote add origin https://github.com/<your-username>/Q-Rescue.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<your-username>/Q-Rescue`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Deploy!"**.

Streamlit Cloud will read `requirements.txt`, install dependencies (including PennyLane, Folium, Plotly, Scikit-learn), and boot the application at a public URL (e.g. `https://q-rescue.streamlit.app`).

---

## 3. Free Hosting Platform Considerations & Limitations

| Aspect | Behavior on Streamlit Community Cloud | Q-Rescue Architectural Handling |
| :--- | :--- | :--- |
| **Ephemeral Storage** | Local container disk resets on app reboot or inactivity timeout. | `data/sample_flood_data.csv` and `data/sample_resource_inventory.csv` are tracked in Git. The app automatically re-initializes and seeds SQLite on startup if the database file is absent. |
| **Memory Constraints** | 1 GB RAM allocation per free container. | VQC uses 4 qubits (Hilbert space size $2^4 = 16$), and QAOA uses candidate clusters of 4–8 qubits ($2^8 = 256$ states). Memory consumption stays well below 350 MB. |
| **Cold Starts** | Inactivity causes the app to sleep. | Pre-trained model weights (`models/vqc_model.joblib` and `classical_rf_model.joblib`) are packaged in the repository so cold boots take $< 4$ seconds without re-training delay. |
| **Quantum Hardware** | No physical QPU connectivity on free tier. | Runs state-vector simulation via PennyLane's `default.qubit` CPU backend. |
| **User Persistence** | User logins and optimization history stored locally in SQLite. | Documented as an academic prototype; administrators can export allocation plans to CSV at any time. |
