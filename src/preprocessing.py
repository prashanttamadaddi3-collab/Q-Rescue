"""
Preprocessing module for Q-Rescue.
Prepares features for Classical Machine Learning and Quantum VQC circuits.
Ensures zero data leakage and handles feature scaling, imputation, and encoding.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Key feature sets
CLASSICAL_FEATURES = [
    "rainfall_mm", "river_water_level_m", "elevation_m", 
    "historical_flood_indicator", "temperature_c", "humidity_percent", 
    "population", "distance_to_rescue_base_km"
]

# 4 features for 4-qubit Quantum Circuit
QUANTUM_FEATURES = [
    "rainfall_mm", "river_water_level_m", "elevation_m", "historical_flood_indicator"
]

LABEL_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

def clean_and_impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using physically realistic domain defaults."""
    df_clean = df.copy()
    
    defaults = {
        "rainfall_mm": 0.0,
        "river_water_level_m": 2.5,
        "elevation_m": 600.0,
        "historical_flood_indicator": 0,
        "temperature_c": 25.0,
        "humidity_percent": 70.0,
        "population": 15000,
        "distance_to_rescue_base_km": 50.0,
        "available_rescue_teams": 0,
        "available_boats": 0,
        "shelter_capacity": 500
    }
    
    for col, default_val in defaults.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(default_val)
        else:
            df_clean[col] = default_val
            
    return df_clean

class PreprocessingPipeline:
    """Manages scaling and feature transformation for Classical and Quantum pipelines."""
    
    def __init__(self):
        self.classical_scaler = StandardScaler()
        # Scale quantum features to [0, pi] for angle embedding rotations
        self.quantum_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """Fit scalers on training data to avoid data leakage."""
        df_clean = clean_and_impute_dataframe(df)
        X_classical = df_clean[CLASSICAL_FEATURES].values
        X_quantum = df_clean[QUANTUM_FEATURES].values
        
        self.classical_scaler.fit(X_classical)
        self.quantum_scaler.fit(X_quantum)
        self.is_fitted = True
        return self

    def transform_classical(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features for classical models."""
        if not self.is_fitted:
            raise ValueError("PreprocessingPipeline is not fitted yet.")
        df_clean = clean_and_impute_dataframe(df)
        return self.classical_scaler.transform(df_clean[CLASSICAL_FEATURES].values)

    def transform_quantum(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features for 4-qubit Quantum VQC circuit."""
        if not self.is_fitted:
            raise ValueError("PreprocessingPipeline is not fitted yet.")
        df_clean = clean_and_impute_dataframe(df)
        return self.quantum_scaler.transform(df_clean[QUANTUM_FEATURES].values)

    def fit_transform_datasets(self, df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into train/test, fit on train only, return:
        (X_train_c, X_test_c, X_train_q, X_test_q, y_train, y_test)
        """
        df_clean = clean_and_impute_dataframe(df)
        
        if "flood_risk_label" not in df_clean.columns:
            raise ValueError("DataFrame must contain 'flood_risk_label' for supervised training.")
            
        y = df_clean["flood_risk_label"].map(LABEL_MAPPING).values
        
        # Check if all 4 classes exist
        stratify_opt = y if len(np.unique(y)) > 1 else None
        
        train_df, test_df, y_train, y_test = train_test_split(
            df_clean, y, test_size=test_size, random_state=random_state, stratify=stratify_opt
        )
        
        self.fit(train_df)
        
        X_train_c = self.transform_classical(train_df)
        X_test_c = self.transform_classical(test_df)
        X_train_q = self.transform_quantum(train_df)
        X_test_q = self.transform_quantum(test_df)
        
        return X_train_c, X_test_c, X_train_q, X_test_q, y_train, y_test

    def save(self, filepath: Optional[str] = None) -> str:
        """Save fitted pipeline to disk."""
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, "preprocessor.joblib")
        joblib.dump(self, filepath)
        return filepath

    @staticmethod
    def load(filepath: Optional[str] = None) -> "PreprocessingPipeline":
        """Load pipeline from disk."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "preprocessor.joblib")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessing pipeline file not found at {filepath}")
        return joblib.load(filepath)
