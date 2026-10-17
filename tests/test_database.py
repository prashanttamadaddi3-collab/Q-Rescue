"""
Unit and integration tests for SQLite database operations in Q-Rescue.
"""

import os
import tempfile
import pytest
import sqlite3
import pandas as pd
from src.database import (
    init_db, seed_initial_data, get_connection, 
    get_all_locations, get_latest_observations, 
    save_prediction, get_latest_predictions,
    get_resource_inventory, update_resource_inventory,
    save_optimization_run, get_optimization_history, get_run_allocations
)

@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    import gc
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass

def test_database_initialization(temp_db):
    """Test that all required tables are properly created."""
    with get_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
    expected_tables = [
        "users", "locations", "observations", "predictions", 
        "resource_inventory", "optimization_runs", "allocation_results"
    ]
    for table in expected_tables:
        assert table in tables, f"Table {table} was not created in database."

def test_seed_initial_data(temp_db):
    """Test seeding database with sample datasets."""
    seed_initial_data(temp_db)
    
    locs = get_all_locations(temp_db)
    assert len(locs) > 0, "Locations table should contain seeded locations."
    assert "location_name" in locs.columns
    assert "taluk" in locs.columns

    inv = get_resource_inventory(temp_db)
    assert len(inv) > 0, "Resource inventory should contain seeded inventory."
    assert "resource_type" in inv.columns

def test_save_and_retrieve_prediction(temp_db):
    """Test saving a risk prediction and retrieving it."""
    seed_initial_data(temp_db)
    locs = get_all_locations(temp_db)
    sample_loc_id = locs.iloc[0]["location_id"]
    
    pred_id = save_prediction(
        location_id=sample_loc_id,
        observation_date="2024-08-01",
        model_type="VQC",
        predicted_risk_level="Critical",
        confidence_score=0.92,
        raw_quantum_output="{'Critical': 0.92}",
        db_path=temp_db
    )
    assert pred_id > 0
    
    latest = get_latest_predictions(model_type="VQC", db_path=temp_db)
    assert len(latest) > 0
    assert latest.iloc[0]["predicted_risk_level"] == "Critical"

def test_save_and_retrieve_optimization_run(temp_db):
    """Test saving an emergency resource optimization run and allocation rows."""
    seed_initial_data(temp_db)
    locs = get_all_locations(temp_db)
    sample_loc_id = locs.iloc[0]["location_id"]
    
    items = [{
        "location_id": sample_loc_id,
        "risk_level": "High",
        "priority_rank": 1,
        "teams_allocated": 2,
        "boats_allocated": 1,
        "ambulances_allocated": 1,
        "food_allocated": 4,
        "medical_kits_allocated": 4,
        "unmet_demand_flag": 0,
        "notes": "Allocated via QAOA"
    }]
    
    run_id = save_optimization_run(
        algorithm="QAOA",
        total_locations=1,
        total_allocated=12,
        total_demand=12,
        total_shortage=0,
        objective_cost=45.2,
        satisfaction_rate=100.0,
        exec_time=0.35,
        allocation_items=items,
        notes="Test QAOA Run",
        db_path=temp_db
    )
    assert run_id > 0
    
    history = get_optimization_history(temp_db)
    assert len(history) >= 1
    assert history.iloc[0]["algorithm"] == "QAOA"
    
    alloc_rows = get_run_allocations(run_id, temp_db)
    assert len(alloc_rows) == 1
    assert alloc_rows.iloc[0]["location_id"] == sample_loc_id
    assert alloc_rows.iloc[0]["teams_allocated"] == 2
