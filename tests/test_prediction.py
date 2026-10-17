"""
Unit tests for Classical ML and Variational Quantum Classifier (VQC) models.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_loader import validate_and_load_csv
from src.preprocessing import PreprocessingPipeline
from src.classical_model import ClassicalFloodModel
from src.vqc_model import VariationalQuantumClassifier, NUM_QUBITS
from src.risk_prediction import predict_flood_risk_batch, predict_single_location

@pytest.fixture
def sample_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "sample_flood_data.csv")
    df, _ = validate_and_load_csv(csv_path)
    return df

def test_classical_model_training_and_eval(sample_dataset):
    """Test training Random Forest baseline and checking output metrics."""
    prep = PreprocessingPipeline()
    X_train_c, X_test_c, _, _, y_train, y_test = prep.fit_transform_datasets(sample_dataset, test_size=0.25)
    
    model = ClassicalFloodModel(n_estimators=30, max_depth=5)
    model.train(X_train_c, y_train)
    
    assert model.is_fitted is True
    
    metrics = model.evaluate(X_test_c, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 4

    importances = model.get_feature_importances()
    assert len(importances) == len(prep.transform_classical(sample_dataset.iloc[:2])[0])

def test_vqc_circuit_forward_and_probabilities(sample_dataset):
    """Test 4-qubit VQC forward pass and probability normalization."""
    prep = PreprocessingPipeline()
    _, _, X_train_q, _, y_train, _ = prep.fit_transform_datasets(sample_dataset, test_size=0.25)
    
    vqc = VariationalQuantumClassifier(num_qubits=NUM_QUBITS, epochs=2, batch_size=16)
    vqc.train(X_train_q[:16], y_train[:16])
    
    probs = vqc.predict_proba(X_train_q[:5])
    assert probs.shape == (5, 4)
    # Probabilities should sum to 1.0 per sample
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)

    preds = vqc.predict(X_train_q[:5])
    assert len(preds) == 5
    assert all(0 <= p <= 3 for p in preds)

def test_predict_single_location():
    """Test single location inference."""
    sample_loc = {
        "location_id": "BLG-TEST-01",
        "location_name": "Kallol Confluence",
        "taluk": "Chikkodi",
        "rainfall_mm": 185.0,
        "river_water_level_m": 9.4,
        "elevation_m": 538.0,
        "historical_flood_indicator": 1,
        "temperature_c": 24.0,
        "humidity_percent": 95.0,
        "population": 12500,
        "distance_to_rescue_base_km": 84.0
    }
    
    res_c = predict_single_location(sample_loc, model_type="Classical")
    assert "predicted_risk_label" in res_c
    assert res_c["predicted_risk_label"] in ["Low", "Medium", "High", "Critical"]
    assert 0.0 <= res_c["confidence"] <= 1.0
