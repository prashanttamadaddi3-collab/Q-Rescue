"""
Classical Machine Learning Baseline model for flood risk prediction.
Uses Random Forest Classifier and Logistic Regression with full metrics evaluation,
feature importance extraction, and model persistence.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.preprocessing import CLASSICAL_FEATURES, LABEL_MAPPING, REVERSE_LABEL_MAPPING

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

class ClassicalFloodModel:
    """Random Forest classical baseline classifier for 4-class flood risk."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced"
        )
        self.is_fitted = False
        self.feature_names = CLASSICAL_FEATURES

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "ClassicalFloodModel":
        """Train Random Forest model."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk class indices (0 to 3)."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics on held-out test data."""
        y_pred = self.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]).tolist()
        
        # Per class metrics
        report_dict = classification_report(
            y_test, y_pred, 
            labels=[0, 1, 2, 3],
            target_names=["Low", "Medium", "High", "Critical"],
            output_dict=True,
            zero_division=0
        )
        
        return {
            "model_name": "Random Forest Baseline",
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist()
        }

    def get_feature_importances(self) -> Dict[str, float]:
        """Return Gini feature importances."""
        if not self.is_fitted:
            return {}
        importances = self.model.feature_importances_
        return {name: round(float(val), 4) for name, val in zip(self.feature_names, importances)}

    def save(self, filepath: Optional[str] = None) -> str:
        """Save trained model to disk."""
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, "classical_rf_model.joblib")
        joblib.dump(self, filepath)
        return filepath

    @staticmethod
    def load(filepath: Optional[str] = None) -> "ClassicalFloodModel":
        """Load model from disk."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "classical_rf_model.joblib")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        return joblib.load(filepath)
