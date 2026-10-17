# System Architecture — Q-Rescue

**Hybrid Quantum–Classical Framework for Flood Risk Prediction and Emergency Resource Allocation using VQC and QAOA**

APS College of Engineering, Bengaluru  
Department of Information Science and Engineering

---

## 1. High-Level Architectural Overview

Q-Rescue bridges physical flood hazard estimation and rapid logistical dispatch through a hybrid quantum–classical pipeline:

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data Ingestion & Preprocessing"]
        A1["Belagavi Monitored Stations Telemetry"] --> B["Data Loader & Validator"]
        A2["Depot Stockpiles & Logistics"] --> B
        B --> C1["Classical Scaler (StandardScaler)"]
        B --> C2["Quantum Scaler (MinMaxScaler to [0, π])"]
    end

    subgraph Quantum_ML ["2. Flood Risk Classification"]
        C1 --> D1["Classical Baseline (Random Forest)"]
        C2 --> D2["4-Qubit VQC (AngleEmbedding + Parameterized Layers)"]
        D1 --> E["Risk Classification: Low, Medium, High, Critical"]
        D2 --> E
    end

    subgraph Optimization ["3. Emergency Resource Optimization"]
        E --> F["QUBO Problem Formulation"]
        A2 --> F
        F --> G1["Classical Priority Greedy Benchmark"]
        F --> G2["QAOA Simulator (Cost + Mixer Hamiltonians)"]
        G1 --> H["Optimal Allocation Vector"]
        G2 --> H
    end

    subgraph Decision_Support ["4. UI & Emergency Operations"]
        H --> I1["Interactive Folium GIS Map"]
        H --> I2["Streamlit Command Dashboard"]
        H --> I3["Explainable AI & Audit Reports"]
        H --> I4["SQLite Persistent Storage"]
    end
```

---

## 2. Component Deconstruction

### 2.1 Web Presentation Layer (`Streamlit`)
- **`app.py`**: Entry landing page, session-aware authentication, executive metrics summary, and academic details.
- **`pages/1_📊_Dashboard.py`**: Real-time hazard monitoring, summary metric cards, Plotly risk donut & scatter matrix, searchable station table.
- **`pages/2_🔮_Risk_Prediction.py`**: Interactive inference for custom inputs or station presets, VQC vs Random Forest toggle, confidence and Pauli-Z expectation readout.
- **`pages/3_📦_Resource_Inventory.py`**: Central depot equipment ledger (Rescue teams, boats, ambulances, rations, medical kits).
- **`pages/4_⚛️_QAOA_Optimization.py`**: Parameterized QUBO optimization, execution of QAOA and Classical baselines, shortage alerts, and CSV export.
- **`pages/5_🗺️_Risk_Map.py`**: Folium OpenStreetMap visualization centered on Belagavi district with risk markers and DEOC dispatch base.
- **`pages/6_📈_Model_Evaluation.py`**: Fair benchmark comparison on identical test split, confusion matrices, and VQC training curves.
- **`pages/7_💡_Explainable_AI.py`**: Quantum circuit blueprints, QAOA trade-off weightings, and plain-English natural language case studies.
- **`pages/8_📜_History.py`**: SQLite historical observation archive and past optimization run records.
- **`pages/9_📁_Data_Management.py`**: Telemetry CSV uploader, range validator, database table inspector, and re-seeding tool.
- **`pages/10_ℹ️_About.py`**: VTU academic project details, faculty guide, HOD, student team, and literature survey.

### 2.2 Business Logic & Machine Learning (`src/`)
- **`src/database.py`**: SQLite connection pooling (WAL mode, foreign keys), schema migrations, transactions, and CRUD operations.
- **`src/authentication.py`**: Bcrypt salted password hashing, credential validation, session state management.
- **`src/data_loader.py`**: Strict schema checking, physical range validation, missing value diagnosis, and data quality reports.
- **`src/preprocessing.py`**: Imputation of physical defaults, MinMax angle scaling to $[0, \pi]$ for quantum circuits, leak-free train/test split.
- **`src/classical_model.py`**: 100-estimator Random Forest classifier with balanced class weighting, Gini feature importance, and full classification metrics.
- **`src/vqc_model.py`**: 4-qubit Variational Quantum Classifier using PennyLane, AngleEmbedding, parameterized rotation layers, circular CNOT entanglement, and multi-qubit Pauli-Z expectation readout.
- **`src/qaoa_optimizer.py`**: Multi-resource allocation formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem, mapped to an Ising Hamiltonian, and solved via PennyLane QAOA.
- **`src/resource_allocation.py`**: Dispatch orchestration, shortage calculations, CSV formatting, and database persistence.
- **`src/explainability.py`**: Circuit deconstruction, XAI explanations, and plain-English narrative generation.
- **`src/visualizations.py`**: Plotly charts adhering to a dark disaster-management color palette.
- **`src/map_utils.py`**: Verified Belagavi district GIS mapping with Folium.

---

## 3. Database Schema

The SQLite schema (`data/q_rescue.db`) manages 7 core relational tables:

```mermaid
erDiagram
    USERS {
        int user_id PK
        string username UK
        string password_hash
        string email
        string full_name
        string role
        timestamp created_at
        timestamp last_login
    }

    LOCATIONS {
        string location_id PK
        string location_name
        string taluk
        float latitude
        float longitude
        float elevation_m
        int population
        float distance_to_rescue_base_km
        int historical_flood_indicator
    }

    OBSERVATIONS {
        int observation_id PK
        string location_id FK
        string observation_date
        float rainfall_mm
        float temperature_c
        float humidity_percent
        float river_water_level_m
        int available_rescue_teams
        int available_boats
        int shelter_capacity
        string flood_risk_label
    }

    PREDICTIONS {
        int prediction_id PK
        string location_id FK
        string observation_date
        string model_type
        string predicted_risk_level
        float confidence_score
        string raw_quantum_output
        timestamp created_at
    }

    RESOURCE_INVENTORY {
        string resource_id PK
        string resource_type
        int total_available_units
        int allocated_units
        float unit_cost_or_weight
        string depot_location
        timestamp last_updated
    }

    OPTIMIZATION_RUNS {
        int run_id PK
        timestamp run_timestamp
        string algorithm
        int total_locations_considered
        int total_resources_allocated
        int total_demand
        int total_shortage
        float objective_cost
        float constraint_satisfaction_rate
        float execution_time_seconds
        string notes
    }

    ALLOCATION_RESULTS {
        int result_id PK
        int run_id FK
        string location_id FK
        string risk_level
        int priority_rank
        int teams_allocated
        int boats_allocated
        int ambulances_allocated
        int food_allocated
        int medical_kits_allocated
        int unmet_demand_flag
        string notes
    }

    LOCATIONS ||--o{ OBSERVATIONS : has
    LOCATIONS ||--o{ PREDICTIONS : receives
    LOCATIONS ||--o{ ALLOCATION_RESULTS : allocated_to
    OPTIMIZATION_RUNS ||--o{ ALLOCATION_RESULTS : produces
```
