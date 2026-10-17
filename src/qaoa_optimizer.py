"""
Quantum Approximate Optimization Algorithm (QAOA) and Classical Optimizer for Emergency Resource Allocation.
Formulates the multi-resource allocation problem as a QUBO (Quadratic Unconstrained Binary Optimization),
maps to an Ising cost Hamiltonian, and solves via PennyLane QAOA simulator with a Classical ILP baseline.
"""

import time
import numpy as np
import pandas as pd
import pennylane as qml
from typing import Dict, Any, List, Tuple, Optional

# Risk level multiplier weights
RISK_WEIGHTS = {
    "Critical": 4.5,
    "High": 3.0,
    "Medium": 1.5,
    "Low": 0.5
}

class EmergencyResourceOptimizer:
    """
    Optimizes emergency resource allocation across flood-affected locations using QAOA & Classical methods.
    Accounts for risk levels, population exposure, travel distance, and capacity constraints.
    """

    def __init__(self, locations_df: pd.DataFrame, inventory_dict: Dict[str, int]):
        """
        locations_df must contain:
        location_id, location_name, taluk, predicted_risk_level (or flood_risk_label),
        population, distance_to_rescue_base_km.
        inventory_dict:
        {'teams': int, 'boats': int, 'ambulances': int, 'food': int, 'medical_kits': int}
        """
        self.df = locations_df.copy()
        self.inventory = inventory_dict.copy()
        self._prepare_demands()

    def _prepare_demands(self) -> None:
        """Derive standard emergency resource requirements per location based on risk and population."""
        teams_req, boats_req, amb_req, food_req, med_req = [], [], [], [], []
        priority_scores = []
        
        # Max pop and dist for normalization
        max_pop = max(1, self.df["population"].max())
        max_dist = max(1.0, self.df["distance_to_rescue_base_km"].max())
        
        for _, row in self.df.iterrows():
            risk = row.get("predicted_risk_level", row.get("flood_risk_label", "Medium"))
            w_risk = RISK_WEIGHTS.get(risk, 1.5)
            pop = float(row.get("population", 10000))
            dist = float(row.get("distance_to_rescue_base_km", 50.0))
            
            # Demand calculation
            if risk == "Critical":
                t = int(max(2, min(8, round(pop / 6000))))
                b = int(max(2, min(6, round(pop / 8000))))
                a = int(max(1, min(4, round(pop / 10000))))
                f = int(max(3, min(10, round(pop / 4000))))
                m = int(max(3, min(10, round(pop / 4000))))
            elif risk == "High":
                t = int(max(1, min(5, round(pop / 9000))))
                b = int(max(1, min(4, round(pop / 12000))))
                a = int(max(1, min(3, round(pop / 15000))))
                f = int(max(2, min(7, round(pop / 6000))))
                m = int(max(2, min(7, round(pop / 6000))))
            elif risk == "Medium":
                t = int(max(1, min(3, round(pop / 15000))))
                b = int(max(0, min(2, round(pop / 25000))))
                a = int(max(0, min(2, round(pop / 20000))))
                f = int(max(1, min(4, round(pop / 10000))))
                m = int(max(1, min(4, round(pop / 10000))))
            else: # Low
                t = 1 if pop > 25000 else 0
                b = 0
                a = 0
                f = 1 if pop > 20000 else 0
                m = 1 if pop > 20000 else 0
                
            teams_req.append(t)
            boats_req.append(b)
            amb_req.append(a)
            food_req.append(f)
            med_req.append(m)
            
            # Composite priority utility score: Risk (60%), Population (25%), Distance penalty (15%)
            score = (w_risk * 10.0) + (pop / max_pop * 5.0) - (dist / max_dist * 2.5)
            priority_scores.append(round(score, 2))
            
        self.df["req_teams"] = teams_req
        self.df["req_boats"] = boats_req
        self.df["req_ambulances"] = amb_req
        self.df["req_food"] = food_req
        self.df["req_medical_kits"] = med_req
        self.df["priority_score"] = priority_scores

    def build_qubo_matrix(self, candidate_indices: List[int], penalty_factor: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build QUBO Q matrix and linear vector for candidate locations:
        Minimize x^T Q x + L^T x
        where utility is maximized and resource capacity overshoots are quadratically penalized.
        """
        N = len(candidate_indices)
        sub_df = self.df.iloc[candidate_indices]
        
        utilities = sub_df["priority_score"].values
        # Negative utility for minimization
        L = -utilities.copy()
        Q = np.zeros((N, N))
        
        # Penalties for resource types: teams, boats, ambulances
        resource_keys = [
            ("req_teams", self.inventory.get("teams", 35)),
            ("req_boats", self.inventory.get("boats", 24)),
            ("req_ambulances", self.inventory.get("ambulances", 20))
        ]
        
        for col_name, budget in resource_keys:
            demands = sub_df[col_name].values
            # (sum_i d_i x_i - B)^2 = sum_i d_i^2 x_i + 2 sum_{i<j} d_i d_j x_i x_j - 2 B sum_i d_i x_i + B^2
            # Add quadratic penalty:
            for i in range(N):
                L[i] += penalty_factor * (demands[i]**2 - 2 * budget * demands[i]) / 100.0
                for j in range(i + 1, N):
                    term = 2.0 * penalty_factor * (demands[i] * demands[j]) / 100.0
                    Q[i, j] += term
                    Q[j, i] += term
                    
        return Q, L

    def run_classical_baseline(self) -> Dict[str, Any]:
        """
        Deterministic Knapsack / Priority-based Classical Optimizer.
        Provides guaranteed feasible allocation and acts as benchmark for QAOA.
        """
        start_time = time.time()
        
        # Sort by priority score descending
        sorted_df = self.df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
        
        remaining = self.inventory.copy()
        allocated_records = []
        
        total_allocated = 0
        total_demand = 0
        total_shortage = 0
        
        for rank, row in sorted_df.iterrows():
            loc_id = row["location_id"]
            risk = row.get("predicted_risk_level", row.get("flood_risk_label", "Medium"))
            
            req_t = row["req_teams"]
            req_b = row["req_boats"]
            req_a = row["req_ambulances"]
            req_f = row["req_food"]
            req_m = row["req_medical_kits"]
            
            loc_demand = req_t + req_b + req_a + req_f + req_m
            total_demand += loc_demand
            
            # Allocate available up to demand
            alloc_t = min(req_t, remaining.get("teams", 0))
            alloc_b = min(req_b, remaining.get("boats", 0))
            alloc_a = min(req_a, remaining.get("ambulances", 0))
            alloc_f = min(req_f, remaining.get("food", 0))
            alloc_m = min(req_m, remaining.get("medical_kits", 0))
            
            remaining["teams"] -= alloc_t
            remaining["boats"] -= alloc_b
            remaining["ambulances"] -= alloc_a
            remaining["food"] -= alloc_f
            remaining["medical_kits"] -= alloc_m
            
            loc_alloc = alloc_t + alloc_b + alloc_a + alloc_f + alloc_m
            total_allocated += loc_alloc
            
            shortage = loc_demand - loc_alloc
            total_shortage += shortage
            unmet = 1 if shortage > 0 else 0
            
            note = "Fully fulfilled" if unmet == 0 else f"Shortage of {shortage} units (Priority Rank #{rank+1})"
            
            allocated_records.append({
                "location_id": loc_id,
                "location_name": row["location_name"],
                "taluk": row["taluk"],
                "risk_level": risk,
                "priority_rank": rank + 1,
                "priority_score": row["priority_score"],
                "teams_allocated": alloc_t,
                "boats_allocated": alloc_b,
                "ambulances_allocated": alloc_a,
                "food_allocated": alloc_f,
                "medical_kits_allocated": alloc_m,
                "unmet_demand_flag": unmet,
                "notes": note
            })
            
        elapsed = round(time.time() - start_time, 4)
        sat_rate = round(float(total_allocated / max(1, total_demand) * 100.0), 2)
        
        return {
            "algorithm": "Classical Priority Optimizer (Greedy Benchmark)",
            "allocation_df": pd.DataFrame(allocated_records),
            "total_locations": len(self.df),
            "total_resources_allocated": total_allocated,
            "total_demand": total_demand,
            "total_shortage": total_shortage,
            "constraint_satisfaction_rate": sat_rate,
            "execution_time_seconds": elapsed,
            "remaining_inventory": remaining,
            "objective_value": round(float(sum(r["priority_score"] for r in allocated_records if r["unmet_demand_flag"] == 0)), 2)
        }

    def run_qaoa_optimization(self, p_steps: int = 1, max_qubits: int = 8) -> Dict[str, Any]:
        """
        Execute QAOA simulation on top priority flood locations using PennyLane.
        Transforms candidate QUBO to Ising Hamiltonian and optimizes variational angles (gamma, beta).
        """
        start_time = time.time()
        
        # Sort locations by risk and priority
        sorted_df = self.df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
        
        # Candidate set for quantum circuit (up to max_qubits for simulator efficiency)
        n_candidates = min(len(sorted_df), max_qubits)
        candidate_indices = list(range(n_candidates))
        
        Q, L = self.build_qubo_matrix(candidate_indices)
        
        # Convert QUBO to Ising Pauli-Z Hamiltonian
        # x_i = (I - Z_i)/2
        coeffs = []
        obs = []
        
        # Linear terms: L_i (I - Z_i)/2 -> - L_i/2 Z_i
        for i in range(n_candidates):
            coeffs.append(-L[i] / 2.0)
            obs.append(qml.PauliZ(i))
            
        # Quadratic terms: Q_ij (I - Z_i)(I - Z_j)/4 -> Q_ij/4 Z_i Z_j
        for i in range(n_candidates):
            for j in range(i + 1, n_candidates):
                if abs(Q[i, j]) > 1e-4:
                    coeffs.append(Q[i, j] / 4.0)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                    
        cost_h = qml.Hamiltonian(coeffs, obs)
        mixer_h = qml.qaoa.x_mixer(wires=range(n_candidates))
        
        dev_qaoa = qml.device("default.qubit", wires=n_candidates)
        
        def qaoa_layer(gamma, beta):
            qml.qaoa.cost_layer(gamma, cost_h)
            qml.qaoa.mixer_layer(beta, mixer_h)
            
        @qml.qnode(dev_qaoa)
        def circuit(params):
            # Equal superposition
            for w in range(n_candidates):
                qml.Hadamard(wires=w)
            # p QAOA layers
            for l in range(p_steps):
                qaoa_layer(params[0][l], params[1][l])
            return qml.expval(cost_h)

        # Optimize QAOA parameters
        init_params = [0.1 * np.ones(p_steps), 0.1 * np.ones(p_steps)]
        opt = qml.GradientDescentOptimizer(stepsize=0.1)
        params = init_params
        
        # 10 optimization steps
        for _ in range(10):
            params = opt.step(circuit, params)
            
        # Sample probability distribution to get optimal bitstring
        @qml.qnode(dev_qaoa)
        def prob_circuit(params):
            for w in range(n_candidates):
                qml.Hadamard(wires=w)
            for l in range(p_steps):
                qaoa_layer(params[0][l], params[1][l])
            return qml.probs(wires=range(n_candidates))
            
        probabilities = prob_circuit(params)
        best_bitstring_idx = int(np.argmax(probabilities))
        best_bitstring = format(best_bitstring_idx, f"0{n_candidates}b")
        
        # Decode bitstring into resource deployment decisions
        remaining = self.inventory.copy()
        allocated_records = []
        total_allocated = 0
        total_demand = 0
        total_shortage = 0
        
        for rank, row in sorted_df.iterrows():
            loc_id = row["location_id"]
            risk = row.get("predicted_risk_level", row.get("flood_risk_label", "Medium"))
            
            req_t = row["req_teams"]
            req_b = row["req_boats"]
            req_a = row["req_ambulances"]
            req_f = row["req_food"]
            req_m = row["req_medical_kits"]
            
            loc_demand = req_t + req_b + req_a + req_f + req_m
            total_demand += loc_demand
            
            # Check QAOA decision if in candidate set
            qaoa_select = True
            if rank < n_candidates:
                qaoa_select = (best_bitstring[rank] == "1")
                
            if qaoa_select:
                alloc_t = min(req_t, remaining.get("teams", 0))
                alloc_b = min(req_b, remaining.get("boats", 0))
                alloc_a = min(req_a, remaining.get("ambulances", 0))
                alloc_f = min(req_f, remaining.get("food", 0))
                alloc_m = min(req_m, remaining.get("medical_kits", 0))
            else:
                alloc_t, alloc_b, alloc_a, alloc_f, alloc_m = 0, 0, 0, 0, 0
                
            remaining["teams"] -= alloc_t
            remaining["boats"] -= alloc_b
            remaining["ambulances"] -= alloc_a
            remaining["food"] -= alloc_f
            remaining["medical_kits"] -= alloc_m
            
            loc_alloc = alloc_t + alloc_b + alloc_a + alloc_f + alloc_m
            total_allocated += loc_alloc
            shortage = loc_demand - loc_alloc
            total_shortage += shortage
            unmet = 1 if shortage > 0 else 0
            
            note = "QAOA Activated & Satisfied" if (qaoa_select and unmet == 0) else \
                   (f"QAOA Selected with {shortage} shortage" if qaoa_select else "Deferred by QAOA priority gate")
                   
            allocated_records.append({
                "location_id": loc_id,
                "location_name": row["location_name"],
                "taluk": row["taluk"],
                "risk_level": risk,
                "priority_rank": rank + 1,
                "priority_score": row["priority_score"],
                "teams_allocated": alloc_t,
                "boats_allocated": alloc_b,
                "ambulances_allocated": alloc_a,
                "food_allocated": alloc_f,
                "medical_kits_allocated": alloc_m,
                "unmet_demand_flag": unmet,
                "notes": note
            })
            
        elapsed = round(time.time() - start_time, 4)
        sat_rate = round(float(total_allocated / max(1, total_demand) * 100.0), 2)
        
        return {
            "algorithm": "Quantum Approximate Optimization Algorithm (QAOA)",
            "allocation_df": pd.DataFrame(allocated_records),
            "total_locations": len(self.df),
            "total_resources_allocated": total_allocated,
            "total_demand": total_demand,
            "total_shortage": total_shortage,
            "constraint_satisfaction_rate": sat_rate,
            "execution_time_seconds": elapsed,
            "optimal_bitstring": best_bitstring,
            "optimal_probability": round(float(np.max(probabilities)), 4),
            "remaining_inventory": remaining,
            "objective_value": round(float(sum(r["priority_score"] for r in allocated_records if r["unmet_demand_flag"] == 0)), 2)
        }
