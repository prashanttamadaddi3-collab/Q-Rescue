"""
Explainable AI (XAI) and Model Interpretability module for Q-Rescue.
Separates classical feature importance, quantum circuit encoding mechanics,
and multi-factor resource optimization trade-offs for clear academic presentation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.preprocessing import QUANTUM_FEATURES, CLASSICAL_FEATURES, LABEL_MAPPING, REVERSE_LABEL_MAPPING

def get_vqc_circuit_explanation() -> Dict[str, Any]:
    """Provide clear academic explanation of the Variational Quantum Classifier."""
    return {
        "architecture_title": "Variational Quantum Classifier (VQC) with 4-Qubit Parameterized Circuit",
        "feature_encoding": {
            "method": "Angle Embedding (RY Rotations)",
            "description": "Each normalized continuous feature x_i in [0, pi] is encoded as the rotation angle of a single-qubit RY gate: |psi(x)> = tensor_i RY(x_i) |0>",
            "qubit_mapping": [
                {"qubit": 0, "feature": "rainfall_mm", "physical_meaning": "Direct precipitation load in locality"},
                {"qubit": 1, "feature": "river_water_level_m", "physical_meaning": "Proximity to river flood warning / danger levels"},
                {"qubit": 2, "feature": "elevation_m", "physical_meaning": "Topographical vulnerability (low-lying basins vs high plateaus)"},
                {"qubit": 3, "feature": "historical_flood_indicator", "physical_meaning": "Geomorphic memory and recurrent inundation vulnerability"}
            ]
        },
        "ansatz": {
            "type": "Strongly Entangling Parameterized Layers",
            "gates": "RY(theta_1) -> RZ(theta_2) single-qubit rotations per wire followed by circular CNOT entanglers",
            "purpose": "Creates non-local quantum correlations (entanglement) across features to capture non-linear hydrological interactions"
        },
        "measurement_and_readout": {
            "observable": "Pauli-Z expectation values <Z_i> on all 4 qubits",
            "class_mapping": [
                {"qubit": 0, "class": "Low Risk", "meaning": "<Z_0> dominant indicates safe hydrologic parameters"},
                {"qubit": 1, "class": "Medium Risk", "meaning": "<Z_1> dominant indicates elevated watch level"},
                {"qubit": 2, "class": "High Risk", "meaning": "<Z_2> dominant indicates warning threshold crossed"},
                {"qubit": 3, "class": "Critical Risk", "meaning": "<Z_3> dominant indicates severe inundation emergency"}
            ],
            "softmax_layer": "Probabilities computed as p_i = exp(beta * <Z_i>) / sum_j exp(beta * <Z_j>) where beta=2.5"
        },
        "examiner_takeaway": (
            "The VQC transforms classical hydrologic features into a quantum Hilbert space of dimension 2^4 = 16. "
            "By tuning variational rotation angles during supervised training, the decision boundary is optimized "
            "to classify multi-class flood hazard states without relying on a black-box classical neural network."
        )
    }

def get_qaoa_explanation() -> Dict[str, Any]:
    """Explain QAOA resource optimization mechanics, QUBO formulation, and trade-offs."""
    return {
        "algorithm_title": "Quantum Approximate Optimization Algorithm (QAOA) for Emergency Dispatch",
        "qubo_formulation": {
            "objective_maximization": "Max sum_i [alpha * Risk_i + beta * Population_i - gamma * Distance_i] * x_i",
            "budget_constraints": "sum_i Demand_{i, k} * x_i <= Budget_k for all resources k",
            "penalty_transformation": "Unconstrained QUBO min E(x) = - Utility(x) + lambda * (Resource_Demand(x) - Budget)^2"
        },
        "ising_hamiltonian": {
            "transformation": "Binary variable x_i in {0, 1} mapped to spin operator x_i = (I - Z_i)/2",
            "cost_hamiltonian": "H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j (diagonal in computational basis)",
            "mixer_hamiltonian": "H_M = sum_i X_i (drives quantum tunneling between allocation states)"
        },
        "circuit_evolution": {
            "state_preparation": "Equal superposition |+>^n via Hadamard gates on all n qubits",
            "alternating_layers": "|psi(gamma, beta)> = [exp(-i beta H_M) exp(-i gamma H_C)]^p |+>^n",
            "optimization": "Variational parameters gamma, beta tuned to maximize expectation <psi| -H_C |psi>"
        },
        "tradeoff_factors": [
            {"factor": "Risk Severity Weight", "weight": "60%", "rationale": "Critical and High risk localities must receive life-saving boats and teams first."},
            {"factor": "Population Exposure", "weight": "25%", "rationale": "Densely populated settlements (e.g. Belagavi, Chikkodi, Gokak) receive priority to minimize human casualties."},
            {"factor": "Travel Distance Penalty", "weight": "15%", "rationale": "Farther taluks have higher travel delays and logistical friction from the Belagavi Central Rescue Base."}
        ],
        "examiner_takeaway": (
            "QAOA solves an NP-hard combinatorial knapsack problem by exploring the superposition of 2^N potential deployment "
            "configurations simultaneously. The quantum mixer drives state exploration while the cost Hamiltonian penalizes budget "
            "violations and rewards high-priority rescue coverage."
        )
    }

def generate_natural_language_explanation(row: pd.Series) -> str:
    """Generate a readable, examiner-ready plain English explanation for a specific location's risk prediction."""
    loc_name = row.get("location_name", "Location")
    taluk = row.get("taluk", "Belagavi")
    risk = row.get("predicted_risk_level", row.get("flood_risk_label", "Medium"))
    rf = float(row.get("rainfall_mm", 0.0))
    wl = float(row.get("river_water_level_m", 2.0))
    elev = float(row.get("elevation_m", 600.0))
    hist = int(row.get("historical_flood_indicator", 0))
    pop = int(row.get("population", 10000))
    dist = float(row.get("distance_to_rescue_base_km", 50.0))
    
    explanation_parts = [
        f"**Location Assessment for {loc_name} ({taluk} Taluk):**",
        f"The system designated this locality as **{risk.upper()} RISK** based on a combination of real-time environmental telemetry and physical terrain factors:"
    ]
    
    # Rainfall factor
    if rf >= 120.0:
        explanation_parts.append(f"• **Severe Rainfall:** Recorded precipitation of **{rf:.1f} mm** significantly exceeds the district flash flood warning threshold.")
    elif rf >= 60.0:
        explanation_parts.append(f"• **Moderate Rainfall:** Recorded precipitation of **{rf:.1f} mm** indicates active monsoon downpours.")
    else:
        explanation_parts.append(f"• **Mild Rainfall:** Recorded precipitation of **{rf:.1f} mm** is within baseline seasonal tolerances.")
        
    # River water level
    if wl >= 8.0:
        explanation_parts.append(f"• **Critical Water Inundation:** River gauge reading of **{wl:.2f} m** is above the danger level, indicating active overtopping of riverbanks.")
    elif wl >= 5.0:
        explanation_parts.append(f"• **Elevated River Level:** River gauge reading of **{wl:.2f} m** is approaching warning levels.")
    else:
        explanation_parts.append(f"• **Normal River Stage:** River gauge reading of **{wl:.2f} m** is safe within river banks.")
        
    # Elevation and historical indicator
    if hist == 1 and elev < 560:
        explanation_parts.append(f"• **Topographical Hazard:** Low elevation of **{elev:.0f} m** situated in a historically recurring flood zone along the river basin.")
    elif hist == 1:
        explanation_parts.append(f"• **Historical Vulnerability:** The area has documented historical flood events in prior monsoon seasons.")
    elif elev > 700:
        explanation_parts.append(f"• **Elevation Defense:** Higher elevation of **{elev:.0f} m** provides natural gravitational drainage reducing standing water risk.")

    # Rescue priority impact
    explanation_parts.append(
        f"• **Emergency Response Factor:** With a population of **{pop:,}** situated **{dist:.1f} km** from the Belagavi Central Rescue Base, "
        f"this area receives {'immediate emergency mobilization priority' if risk in ['Critical', 'High'] else 'continuous monitoring status'}."
    )
    
    return "\n\n".join(explanation_parts)
