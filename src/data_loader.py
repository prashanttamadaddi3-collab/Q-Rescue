"""
Data loader and validation module for Q-Rescue.
Validates incoming CSV data, performs range checks, and produces data-quality reports.
"""

import io
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

REQUIRED_COLUMNS = [
    "location_id", "location_name", "taluk", "rainfall_mm", 
    "river_water_level_m", "elevation_m", "population"
]

OPTIONAL_COLUMNS = [
    "latitude", "longitude", "temperature_c", "humidity_percent", 
    "historical_flood_indicator", "distance_to_rescue_base_km",
    "available_rescue_teams", "available_boats", "shelter_capacity",
    "observation_date", "flood_risk_label"
]

VALID_RISK_LABELS = ["Low", "Medium", "High", "Critical"]

NUMERIC_RANGES = {
    "rainfall_mm": (0.0, 1500.0),
    "river_water_level_m": (0.0, 35.0),
    "elevation_m": (-50.0, 9000.0),
    "population": (1, 50000000),
    "latitude": (8.0, 38.0),       # India bounds
    "longitude": (68.0, 98.0),     # India bounds
    "temperature_c": (-10.0, 60.0),
    "humidity_percent": (0.0, 100.0),
    "distance_to_rescue_base_km": (0.0, 1000.0),
    "available_rescue_teams": (0, 500),
    "available_boats": (0, 1000),
    "shelter_capacity": (0, 1000000)
}

def validate_and_load_csv(file_or_path: Any) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Validate CSV schema, ranges, missing values, duplicates, and produce a data quality report.
    Returns (cleaned_df, quality_report).
    """
    report: Dict[str, Any] = {
        "is_valid": False,
        "errors": [],
        "warnings": [],
        "total_rows": 0,
        "valid_rows": 0,
        "missing_values": {},
        "duplicates_removed": 0,
        "range_violations": [],
        "columns_present": [],
        "missing_optional_cols": []
    }
    
    try:
        if isinstance(file_or_path, (str, io.StringIO, io.BytesIO)):
            df = pd.read_csv(file_or_path)
        elif hasattr(file_or_path, "read"):
            # Streamlit UploadedFile
            df = pd.read_csv(file_or_path)
        else:
            df = pd.DataFrame(file_or_path)
    except Exception as e:
        report["errors"].append(f"Failed to parse CSV file: {str(e)}")
        return None, report
        
    report["total_rows"] = len(df)
    report["columns_present"] = list(df.columns)
    
    if len(df) == 0:
        report["errors"].append("The uploaded file contains no data rows.")
        return None, report

    # Strip column names whitespace
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Check required columns
    missing_req = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_req:
        report["errors"].append(f"Missing mandatory columns: {', '.join(missing_req)}")
        return None, report
        
    # Check optional columns
    missing_opt = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    report["missing_optional_cols"] = missing_opt
    if missing_opt:
        report["warnings"].append(f"Optional columns absent: {', '.join(missing_opt)}. Defaults will be imputed.")

    # Deduplicate rows based on location_id and observation_date if present
    initial_len = len(df)
    subset_keys = ["location_id"]
    if "observation_date" in df.columns:
        subset_keys.append("observation_date")
    df = df.drop_duplicates(subset=subset_keys, keep="last")
    report["duplicates_removed"] = initial_len - len(df)

    # Missing values check
    missing_counts = df.isnull().sum().to_dict()
    report["missing_values"] = {k: v for k, v in missing_counts.items() if v > 0}

    # Range and type validation
    valid_mask = pd.Series([True] * len(df), index=df.index)
    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        if col in df.columns:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")
            out_of_bounds = df[col].notnull() & ((df[col] < min_val) | (df[col] > max_val))
            violations_count = int(out_of_bounds.sum())
            if violations_count > 0:
                report["warnings"].append(f"Column '{col}' has {violations_count} values outside expected bounds [{min_val}, {max_val}].")
                # Clip to physical bounds to prevent numerical instability
                df[col] = df[col].clip(lower=min_val, upper=max_val)

    # Validate labels if present
    if "flood_risk_label" in df.columns:
        invalid_labels = df[~df["flood_risk_label"].isin(VALID_RISK_LABELS) & df["flood_risk_label"].notnull()]["flood_risk_label"].unique()
        if len(invalid_labels) > 0:
            report["warnings"].append(f"Unrecognized flood risk labels found: {list(invalid_labels)}. Expected one of: {VALID_RISK_LABELS}")

    report["valid_rows"] = len(df)
    report["is_valid"] = True
    return df, report
