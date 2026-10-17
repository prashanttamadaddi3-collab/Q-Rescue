"""
Resource Allocation orchestration module for Q-Rescue.
Executes QAOA and Classical optimization, saves runs into SQLite,
and formats data for dashboard visualizations and CSV export.
"""

import pandas as pd
from typing import Dict, Any, Tuple
from src.database import save_optimization_run, get_latest_observations, get_latest_predictions
from src.qaoa_optimizer import EmergencyResourceOptimizer

def execute_resource_allocation(inventory_counts: Dict[str, int], 
                                algorithm: str = "QAOA", 
                                max_quantum_qubits: int = 8,
                                save_to_db: bool = True) -> Dict[str, Any]:
    """
    Run emergency resource optimization on latest monitored flood locations.
    Returns full run summary, allocation dataframe, shortages, and DB run_id.
    """
    # Fetch locations with their latest risk predictions or observations
    preds_df = get_latest_predictions()
    if len(preds_df) == 0:
        # Fallback to observations
        obs_df = get_latest_observations()
        loc_df = obs_df.copy()
        if "flood_risk_label" in loc_df.columns:
            loc_df["predicted_risk_level"] = loc_df["flood_risk_label"]
    else:
        loc_df = preds_df.copy()
        
    optimizer = EmergencyResourceOptimizer(loc_df, inventory_counts)
    
    if algorithm == "QAOA":
        results = optimizer.run_qaoa_optimization(p_steps=1, max_qubits=max_quantum_qubits)
    else:
        results = optimizer.run_classical_baseline()
        
    alloc_df = results["allocation_df"]
    
    # Save to SQLite database
    run_id = None
    if save_to_db:
        items_to_save = alloc_df.to_dict(orient="records")
        run_id = save_optimization_run(
            algorithm=results["algorithm"],
            total_locations=results["total_locations"],
            total_allocated=results["total_resources_allocated"],
            total_demand=results["total_demand"],
            total_shortage=results["total_shortage"],
            objective_cost=results["objective_value"],
            satisfaction_rate=results["constraint_satisfaction_rate"],
            exec_time=results["execution_time_seconds"],
            allocation_items=items_to_save,
            notes=f"Allocation computed via {algorithm} optimizer."
        )
        
    results["run_id"] = run_id
    return results

def format_allocation_csv(alloc_df: pd.DataFrame) -> str:
    """Format allocation dataframe as CSV string for downloading."""
    cols_to_export = [
        "priority_rank", "location_id", "location_name", "taluk", "risk_level",
        "teams_allocated", "boats_allocated", "ambulances_allocated",
        "food_allocated", "medical_kits_allocated", "unmet_demand_flag", "notes"
    ]
    avail_cols = [c for c in cols_to_export if c in alloc_df.columns]
    return alloc_df[avail_cols].to_csv(index=False)
