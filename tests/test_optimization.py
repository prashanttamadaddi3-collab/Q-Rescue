"""
Unit tests for QAOA and Classical Resource Optimization modules.
"""

import os
import pytest
import pandas as pd
from src.data_loader import validate_and_load_csv
from src.qaoa_optimizer import EmergencyResourceOptimizer
from src.resource_allocation import execute_resource_allocation, format_allocation_csv

@pytest.fixture
def sample_locations():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "sample_flood_data.csv")
    df, _ = validate_and_load_csv(csv_path)
    return df.drop_duplicates(subset=["location_id"]).head(10)

@pytest.fixture
def test_inventory():
    return {
        "teams": 20,
        "boats": 15,
        "ambulances": 12,
        "food": 25,
        "medical_kits": 30
    }

def test_classical_optimizer_budget_constraints(sample_locations, test_inventory):
    """Test that classical optimization strictly obeys inventory budget constraints."""
    opt = EmergencyResourceOptimizer(sample_locations, test_inventory)
    res = opt.run_classical_baseline()
    
    alloc_df = res["allocation_df"]
    assert len(alloc_df) == len(sample_locations)
    
    # Assert total allocated does not exceed available budget
    assert alloc_df["teams_allocated"].sum() <= test_inventory["teams"]
    assert alloc_df["boats_allocated"].sum() <= test_inventory["boats"]
    assert alloc_df["ambulances_allocated"].sum() <= test_inventory["ambulances"]
    assert alloc_df["food_allocated"].sum() <= test_inventory["food"]
    assert alloc_df["medical_kits_allocated"].sum() <= test_inventory["medical_kits"]
    
    assert res["total_resources_allocated"] <= sum(test_inventory.values())
    assert res["total_shortage"] >= 0

def test_qaoa_optimizer_execution(sample_locations, test_inventory):
    """Test QAOA optimization on top candidates using PennyLane simulator."""
    opt = EmergencyResourceOptimizer(sample_locations, test_inventory)
    res = opt.run_qaoa_optimization(p_steps=1, max_qubits=4)
    
    assert "optimal_bitstring" in res
    assert len(res["optimal_bitstring"]) == 4
    assert res["algorithm"] == "Quantum Approximate Optimization Algorithm (QAOA)"
    
    alloc_df = res["allocation_df"]
    # Check that allocations respect remaining inventory
    assert alloc_df["teams_allocated"].sum() <= test_inventory["teams"]
    assert alloc_df["boats_allocated"].sum() <= test_inventory["boats"]

def test_format_allocation_csv(sample_locations, test_inventory):
    """Test CSV formatting of allocation results."""
    opt = EmergencyResourceOptimizer(sample_locations, test_inventory)
    res = opt.run_classical_baseline()
    csv_text = format_allocation_csv(res["allocation_df"])
    
    assert "location_id" in csv_text
    assert "teams_allocated" in csv_text
    assert "priority_rank" in csv_text
