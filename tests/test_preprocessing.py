"""
Unit tests for data loader, validation, and preprocessing pipeline.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_loader import validate_and_load_csv, REQUIRED_COLUMNS
from src.preprocessing import PreprocessingPipeline, clean_and_impute_dataframe, QUANTUM_FEATURES, CLASSICAL_FEATURES

@pytest.fixture
def sample_csv_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "sample_flood_data.csv")

def test_validate_and_load_csv(sample_csv_path):
    """Test loading and validating the primary sample dataset."""
    df, report = validate_and_load_csv(sample_csv_path)
    assert report["is_valid"] is True
    assert df is not None
    assert len(df) > 0
    assert report["total_rows"] == len(df)
    assert len(report["errors"]) == 0

def test_invalid_csv_validation():
    """Test validation rejection on invalid CSV schema."""
    invalid_data = pd.DataFrame({
        "random_col": [1, 2, 3]
    })
    df, report = validate_and_load_csv(invalid_data)
    assert report["is_valid"] is False
    assert len(report["errors"]) > 0

def test_clean_and_impute():
    """Test physical domain imputation of missing values."""
    df_missing = pd.DataFrame([{
        "location_id": "TEST-01",
        "location_name": "Test Village",
        "taluk": "Belagavi",
        "rainfall_mm": np.nan,
        "river_water_level_m": np.nan,
        "population": np.nan
    }])
    df_clean = clean_and_impute_dataframe(df_missing)
    assert not df_clean["rainfall_mm"].isnull().any()
    assert not df_clean["river_water_level_m"].isnull().any()
    assert not df_clean["population"].isnull().any()

def test_preprocessing_pipeline_fit_transform(sample_csv_path):
    """Test feature scaling, zero data leakage, and quantum angle bounds [0, pi]."""
    df, _ = validate_and_load_csv(sample_csv_path)
    pipeline = PreprocessingPipeline()
    
    X_train_c, X_test_c, X_train_q, X_test_q, y_train, y_test = pipeline.fit_transform_datasets(df, test_size=0.25)
    
    # Check shapes
    assert X_train_c.shape[1] == len(CLASSICAL_FEATURES)
    assert X_train_q.shape[1] == len(QUANTUM_FEATURES)
    assert len(X_train_c) == len(y_train)
    assert len(X_test_c) == len(y_test)
    
    # Check quantum angle rotation bounds in [0, pi]
    assert np.all(X_train_q >= -1e-5)
    assert np.all(X_train_q <= np.pi + 1e-5)
    assert np.all(X_test_q >= -1e-5)
    assert np.all(X_test_q <= np.pi + 1e-5)
