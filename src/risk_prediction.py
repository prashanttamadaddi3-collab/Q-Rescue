"""
Risk Prediction orchestration module for Q-Rescue.
Executes batch and single-location predictions using VQC or Classical Random Forest,
saving inferences to SQLite and formatting results for the UI.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from src.database import save_prediction, get_connection
from src.preprocessing import PreprocessingPipeline, clean_and_impute_dataframe, REVERSE_LABEL_MAPPING
from src.classical_model import ClassicalFloodModel
from src.vqc_model import VariationalQuantumClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def load_or_train_models() -> Tuple[PreprocessingPipeline, ClassicalFloodModel, VariationalQuantumClassifier]:
    """Load fitted models from disk, or train on sample flood data if not present."""
    prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    rf_path = os.path.join(MODELS_DIR, "classical_rf_model.joblib")
    vqc_path = os.path.join(MODELS_DIR, "vqc_model.joblib")
    
    if os.path.exists(prep_path) and os.path.exists(rf_path) and os.path.exists(vqc_path):
        prep = PreprocessingPipeline.load(prep_path)
        rf_model = ClassicalFloodModel.load(rf_path)
        vqc_model = VariationalQuantumClassifier.load(vqc_path)
        return prep, rf_model, vqc_model
        
    # Otherwise train fresh on sample data
    data_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_flood_data.csv")
    df = pd.read_csv(data_csv)
    
    prep = PreprocessingPipeline()
    X_train_c, X_test_c, X_train_q, X_test_q, y_train, y_test = prep.fit_transform_datasets(df, test_size=0.25)
    
    rf_model = ClassicalFloodModel(n_estimators=100, max_depth=6)
    rf_model.train(X_train_c, y_train)
    
    # Fast lightweight epochs for clean VQC initialization
    vqc_model = VariationalQuantumClassifier(epochs=12, lr=0.08, batch_size=16)
    vqc_model.train(X_train_q, y_train)
    
    prep.save(prep_path)
    rf_model.save(rf_path)
    vqc_model.save(vqc_path)
    
    return prep, rf_model, vqc_model

def predict_flood_risk_batch(df: pd.DataFrame, model_type: str = "VQC", save_to_db: bool = True) -> pd.DataFrame:
    """Run batch prediction on input dataframe and return enriched results."""
    prep, rf_model, vqc_model = load_or_train_models()
    df_clean = clean_and_impute_dataframe(df)
    
    results = df_clean.copy()
    
    if model_type == "VQC":
        X_q = prep.transform_quantum(df_clean)
        probs = vqc_model.predict_proba(X_q)
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        results["predicted_risk_level"] = [REVERSE_LABEL_MAPPING[idx] for idx in pred_indices]
        results["confidence_score"] = [round(float(c), 4) for c in confidences]
        results["model_used"] = "VQC (Variational Quantum Classifier)"
        
        # Save raw quantum output strings
        raw_outputs = []
        for i in range(len(df_clean)):
            raw_dict = {
                REVERSE_LABEL_MAPPING[c]: round(float(probs[i][c]), 3) for c in range(4)
            }
            raw_outputs.append(json.dumps(raw_dict))
        results["raw_output"] = raw_outputs
    else:
        X_c = prep.transform_classical(df_clean)
        probs = rf_model.predict_proba(X_c)
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        results["predicted_risk_level"] = [REVERSE_LABEL_MAPPING[idx] for idx in pred_indices]
        results["confidence_score"] = [round(float(c), 4) for c in confidences]
        results["model_used"] = "Classical Random Forest Baseline"
        results["raw_output"] = [json.dumps({REVERSE_LABEL_MAPPING[c]: round(float(probs[i][c]), 3) for c in range(4)}) for i in range(len(df_clean))]

    # Save to SQLite database if requested
    if save_to_db:
        for _, row in results.iterrows():
            obs_date = str(row.get("observation_date", pd.Timestamp.now().strftime("%Y-%m-%d")))
            save_prediction(
                location_id=str(row["location_id"]),
                observation_date=obs_date,
                model_type=model_type,
                predicted_risk_level=str(row["predicted_risk_level"]),
                confidence_score=float(row["confidence_score"]),
                raw_quantum_output=str(row["raw_output"])
            )
            
    return results

def predict_single_location(location_data: Dict[str, Any], model_type: str = "VQC") -> Dict[str, Any]:
    """Run prediction on a single location input dict and return detailed breakdown."""
    df_single = pd.DataFrame([location_data])
    prep, rf_model, vqc_model = load_or_train_models()
    df_clean = clean_and_impute_dataframe(df_single)
    
    if model_type == "VQC":
        X_q = prep.transform_quantum(df_clean)
        detail = vqc_model.predict_single_with_quantum_state(X_q[0])
        detail["model_used"] = "Variational Quantum Classifier (4-Qubit Circuit)"
        return detail
    else:
        X_c = prep.transform_classical(df_clean)
        probs = rf_model.predict_proba(X_c)[0]
        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class_idx": pred_idx,
            "predicted_risk_label": REVERSE_LABEL_MAPPING[pred_idx],
            "confidence": float(probs[pred_idx]),
            "class_probabilities": {
                REVERSE_LABEL_MAPPING[i]: round(float(probs[i]), 4) for i in range(4)
            },
            "model_used": "Classical Random Forest Baseline",
            "raw_pauli_z_expectations": "N/A (Classical Model)"
        }
