# Q-RESCUE
### Hybrid Quantum–Classical Framework for Flood Risk Prediction and Emergency Resource Allocation using VQC and QAOA

**An Academic Engineering Major Project**  
**Institution:** APS College of Engineering, Bengaluru — 560116  
**Affiliation:** Visvesvaraya Technological University (VTU), Belagavi  
**Department:** Information Science and Engineering  

**Project Guide:** Dr. Krushitha Shetty, Associate Professor, Dept. of ISE, APSCE  
**Coordinator & HOD:** Dr. Mithun B N, Professor & Head, Dept. of ISE, APSCE  

**Student Project Team:**
- **Prashanth T** — `1AP23IS039`
- **Rajesh Naik** — `1AP23IS042`
- **Siddaram Masali** — `1AP23IS045`
- **Srujan Chandra K R** — `1AP23IS050`

---

## 1. Project Abstract
Floods inflict catastrophic socio-economic devastation and human casualties worldwide. Effective disaster mitigation requires two interlocked capabilities: early, granular **flood-risk hazard classification** and rapid, multi-constrained **emergency rescue resource dispatch**. Classical artificial intelligence techniques and heuristic dispatchers often struggle when scaling across combinatorial constraints and high-dimensional non-linear hydrology.

**Q-Rescue** introduces a novel hybrid quantum–classical decision-support architecture. Environmental telemetry (rainfall, river gauge water levels, elevation, and historical flood recurrence) is encoded into a **Variational Quantum Classifier (VQC)** utilizing a 4-qubit parameterized quantum circuit to classify hazard severity into *Low, Medium, High, or Critical* risk categories. Concurrently, emergency logistics are formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem solved via the **Quantum Approximate Optimization Algorithm (QAOA)** to calculate an optimal allocation of rescue teams, inflatable boats, ambulances, rations, and medical kits from central depots while strictly respecting capacity constraints. 

The primary operational focus is **Belagavi District, Karnataka, India**, analyzing vulnerable river stretches across the Krishna, Ghataprabha, Malaprabha, and Dudhganga basins.

---

## 2. Problem Statement & Objectives
- **Problem:** Given environmental, river gauge, and topographical data across diverse localities and finite, scarce emergency response inventory, traditional systems isolate flood forecasting from logistics dispatch. Rescue planning remains manual or rule-based, causing dispatch bottlenecks, travel delays, and uncoordinated resource shortages during life-critical initial flood surge windows.
- **Key Objectives:**
  1. Ingest and validate environmental and geographical telemetry across Belagavi district taluks.
  2. Implement a 4-qubit **Variational Quantum Classifier (VQC)** with AngleEmbedding and strongly entangling ansatz to predict multi-class flood hazard.
  3. Formulate and solve multi-resource allocation as a QUBO using **QAOA** on a quantum simulator, benchmarked against a classical priority knapsack solver.
  4. Build an interactive disaster response GIS map and command dashboard in Streamlit.
  5. Provide transparent Explainable AI (XAI) deconstructing quantum circuit readouts, classical feature importances, and multi-factor dispatch trade-offs.

---

## 3. Technology Stack & Justification

| Technology | Role in Q-Rescue | Justification |
| :--- | :--- | :--- |
| **Python 3.10–3.13** | Core Programming Language | Universal ecosystem for scientific computing, quantum SDKs, and machine learning. |
| **PennyLane (0.45.1)** | Quantum Circuit & Optimization Framework | Native autograd gradient support, parameter-shift differentiation, QAOA modules, and lightning simulators. |
| **Scikit-Learn (1.8.0)** | Classical Machine Learning Baseline | Random Forest Classifier, StandardScaler, and train/test cross-validation splits. |
| **Streamlit (1.66.0)** | Interactive Web Dashboard | Rapid, reactive UI rendering dataframes, interactive controls, and multi-page routing. |
| **Plotly (7.1.0)** | Analytical Visualizations | Interactive donut charts, scatter matrices, heatmaps, and grouped allocation comparisons. |
| **Folium & Streamlit-Folium** | Geospatial Interactive Mapping | OpenStreetMap integration with verified Belagavi latitude/longitude coordinates and custom SVG pins. |
| **SQLite3 (with WAL)** | Relational Persistence | Lightweight, zero-config relational database storing locations, telemetry, predictions, and dispatch runs. |
| **Bcrypt (5.0.0)** | Authentication & Security | Cryptographically secure salted password hashing preventing plaintext password vulnerabilities. |
| **Pytest (9.1.1)** | Automated Test Suite | Robust unit and integration testing covering database, preprocessing, VQC, and QAOA constraints. |

---

## 4. Geographical Focus: Belagavi District, Karnataka

The system models verified geographic coordinates and telemetry across 12 taluks in Belagavi district:
- **Chikkodi Taluk:** Kallol (Krishna–Dudhganga confluence), Yadur, Manjari, Chikkodi Town.
- **Athani Taluk:** Shirguppi, Hulagabali, Athani Town.
- **Gokak Taluk:** Gokak Falls (Ghataprabha River), Konnur, Lolasur.
- **Raybag Taluk:** Kudachi, Diggewadi, Raybag Town.
- **Khanapur Taluk:** Khanapur Town, Jamboti (heavy Western Ghats catchment), Londa.
- **Belagavi Urban/Rural:** Vadgaon, Kangrali Khurd, Peeranwadi.
- **Ramdurg Taluk:** Ramdurg Town, Sureban (Malaprabha River).
- **Bailhongal Taluk:** Bailhongal Town.
- **Saundatti Taluk:** Munavalli (Navilatirtha Dam reservoir).
- **Nippani Taluk:** Soundalga (Vedganga River).
- **Emergency Operations Centre (DEOC):** Belagavi DC Office Complex (`15.8497° N, 74.4977° E`).

*Data Provenance Note:* Geographic coordinates and administrative boundaries are verified official locations in Belagavi District, Karnataka. Hydrological observations (rainfall, water gauge readings) are calibrated illustrative records reflecting typical monsoon flood conditions.

---

## 5. System Architecture & Modules

```
parsh major/
├── app.py                      # Main landing page, executive overview, bcrypt login/register
├── requirements.txt            # Pinned dependency specifications
├── README.md                   # Comprehensive academic documentation
├── .gitignore                  # Git exclusions for python bytecode and temporary logs
├── .streamlit/
│   └── config.toml             # Custom disaster-management dark navy styling
├── assets/                     # Logos and architecture diagrams from presentation
│   ├── aps_logo.jpg
│   ├── vtu_logo.png
│   ├── system_architecture.png
│   ├── data_flow.png
│   └── sequence_diagram.png
├── data/                       # Telemetry data and SQLite database
│   ├── sample_flood_data.csv
│   ├── sample_resource_inventory.csv
│   └── q_rescue.db
├── models/                     # Serialized pre-trained model weights
│   ├── preprocessor.joblib
│   ├── classical_rf_model.joblib
│   └── vqc_model.joblib
├── src/                        # Modular backend engines
│   ├── __init__.py
│   ├── database.py             # SQLite WAL connection, migrations, transactions, CRUD
│   ├── authentication.py       # Bcrypt password hashing, session state management
│   ├── data_loader.py          # CSV ingestion, range bounds checking, data quality report
│   ├── preprocessing.py        # Imputation, quantum angle scaling [0, pi], leak-free split
│   ├── classical_model.py      # Random Forest baseline, feature importance, confusion matrix
│   ├── vqc_model.py            # 4-Qubit VQC circuit, Nesterov momentum, Pauli-Z expectations
│   ├── qaoa_optimizer.py       # QUBO formulation, Ising Hamiltonian, QAOA simulation
│   ├── resource_allocation.py  # Dispatch orchestration, shortage tracking, CSV export
│   ├── explainability.py       # Quantum XAI, QUBO weights, natural language explanations
│   ├── visualizations.py       # Plotly distribution donut, scatter matrix, grouped bar charts
│   └── map_utils.py            # Folium OpenStreetMap visualization and popups
├── pages/                      # Multi-page dashboard modules
│   ├── 1_📊_Dashboard.py       # Command dashboard with metrics, charts, and telemetry table
│   ├── 2_🔮_Risk_Prediction.py # Interactive inference with VQC and Classical toggle
│   ├── 3_📦_Resource_Inventory.py # Central depot stock ledger and equipment editor
│   ├── 4_⚛️_QAOA_Optimization.py # Parameterized QAOA dispatch, shortage alerts, CSV export
│   ├── 5_🗺️_Risk_Map.py        # Interactive Belagavi map with risk markers & DEOC base
│   ├── 6_📈_Model_Evaluation.py # Side-by-side benchmark, confusion matrices, loss curves
│   ├── 7_💡_Explainable_AI.py  # Quantum circuit blueprints, QUBO math, viva cheat-sheet
│   ├── 8_📜_History.py         # Multi-date observation archive and past optimization logs
│   ├── 9_📁_Data_Management.py # CSV uploader, SQLite table inspector, re-seed tool
│   └── 10_ℹ️_About.py          # Academic project details, team, guide, HOD, and references
├── tests/                      # Automated test suite
│   ├── __init__.py
│   ├── test_database.py        # Database schema, foreign keys, transactions
│   ├── test_preprocessing.py   # Range checking, angle scaling, zero leakage
│   ├── test_prediction.py      # VQC forward pass, probabilities, RF baseline
│   └── test_optimization.py    # QAOA constraints, budget checks, shortages
└── docs/                       # Technical & architectural specifications
    ├── architecture.md         # Component diagrams and ER relational schema
    ├── algorithms.md           # Mathematical formulation of VQC and QAOA QUBO
    └── deployment.md           # Local run commands and Streamlit Cloud setup
```

---

## 6. How VQC Works (Flood Risk Classification)
1. **Feature Selection:** 4 key continuous hydrologic features are extracted: Cumulative Rainfall ($x_1$), River Water Level ($x_2$), Elevation ($x_3$), and Historical Flood Indicator ($x_4$).
2. **Angle Embedding:** Features are MinMax-scaled into $[0, \pi]$ and encoded into quantum state $|\psi(\vec{x})\rangle = \bigotimes_{i=1}^4 R_Y(\tilde{x}_i) |0\rangle$.
3. **Variational Ansatz:** Strongly entangling layers apply parameterized $R_Y(\theta)$ and $R_Z(\theta)$ single-qubit rotations followed by circular CNOT entanglers to capture non-linear hydrological interactions across features.
4. **Readout & Softmax:** Expectation values of the Pauli-Z operator $\langle Z_i \rangle \in [-1, +1]$ are measured on each qubit. A temperature-scaled softmax layer ($\beta = 2.5$) converts expectations into normalized probabilities corresponding to:
   - Qubit 0: Low Risk
   - Qubit 1: Medium Risk
   - Qubit 2: High Risk
   - Qubit 3: Critical Risk
5. **Optimization:** Trained using Nesterov Momentum / Gradient Descent on Categorical Cross-Entropy loss via the parameter-shift rule.

---

## 7. How QAOA Works (Emergency Resource Allocation)
1. **Decision Variables:** Let binary variable $x_i \in \{0, 1\}$ represent whether flood candidate location $i$ is actively prioritized for full emergency resource mobilization.
2. **Constrained Objective:** Maximize composite utility based on Risk Weight ($60\%$), Population Exposure ($25\%$), and Travel Distance from Belagavi DEOC ($15\%$), subject to depot capacity limits for rescue teams, boats, ambulances, and rations.
3. **QUBO Formulation:** Capacity budget overshoots are penalized quadratically:
   $$\min_{\vec{x}} E(\vec{x}) = -\sum_i \text{Utility}_i x_i + \sum_k \lambda_k \left( \sum_i d_{i, k} x_i - B_k \right)^2$$
4. **Ising Cost Hamiltonian:** Transformed to quantum spin operators via $x_i = \frac{I - Z_i}{2}$ yielding $H_C = \sum h_i Z_i + \sum J_{ij} Z_i Z_j$.
5. **Quantum Evolution:** Starting from an equal superposition $|+\rangle^{\otimes N}$, $p$ alternating layers of cost unitary $e^{-i \gamma H_C}$ and mixer unitary $e^{-i \beta H_M}$ ($H_M = \sum X_i$) are applied.
6. **Sampling & Decoding:** Measurement in the computational basis yields the optimal bitstring $\vec{x}^*$ maximizing rescue coverage while respecting stock limits.

---

## 8. Installation & Local Execution

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Q-Rescue.git
cd Q-Rescue

# 2. Set up virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest tests/ -v

# 5. Launch application
streamlit run app.py
```

---

## 9. Automated Testing

Run all automated unit and integration tests:
```bash
pytest tests/ -v
```
**Test Coverage Includes:**
- `test_database.py`: Schema generation, WAL activation, transactions, prediction and allocation persistence.
- `test_preprocessing.py`: CSV schema validation, physical domain imputation, MinMax scaling bounds $[0, \pi]$, zero data leakage.
- `test_prediction.py`: VQC forward pass, probability normalization ($\sum p_i = 1.0$), single-location inference, and Random Forest baseline metrics.
- `test_optimization.py`: Strict enforcement of resource budget limits, shortage reporting, QAOA simulator convergence, and CSV formatting.

---

## 10. Free Deployment onto Streamlit Community Cloud

1. Push this repository to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Select the repository, set the branch to `main`, and specify `app.py` as the main entry point.
4. Click **Deploy!**. Streamlit Cloud installs `requirements.txt` and launches the application publicly without requiring paid servers or quantum cloud subscriptions.

---

## 11. Academic Disclaimer
*Q-Rescue is an academic engineering decision-support prototype developed for final year Major Project evaluation at APS College of Engineering, Bengaluru. It is designed to demonstrate hybrid quantum–classical algorithmic feasibility and is not an officially certified government flood warning service.*
