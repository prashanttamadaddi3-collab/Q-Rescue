"""
Visualization utilities for Q-Rescue using Plotly and Streamlit native components.
Standardizes dark disaster-management theme, risk-level color palettes, and analytical charts.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Standardized color scheme
RISK_COLORS = {
    "Low": "#10B981",       # Emerald
    "Medium": "#F59E0B",    # Amber
    "High": "#F97316",      # Vibrant Orange
    "Critical": "#EF4444"   # Crimson
}

CHART_LAYOUT_DEFAULTS = {
    "template": "plotly_dark",
    "paper_bgcolor": "#1E293B",
    "plot_bgcolor": "#0F172A",
    "font": {"color": "#F8FAFC", "family": "sans-serif"},
    "margin": {"l": 40, "r": 40, "t": 50, "b": 40}
}

def create_risk_distribution_pie(df: pd.DataFrame, risk_col: str = "flood_risk_label") -> go.Figure:
    """Donut chart showing risk category proportions."""
    counts = df[risk_col].value_counts().reset_index()
    counts.columns = ["Risk_Level", "Count"]
    
    colors = [RISK_COLORS.get(r, "#94A3B8") for r in counts["Risk_Level"]]
    
    fig = go.Figure(data=[go.Pie(
        labels=counts["Risk_Level"],
        values=counts["Count"],
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#0F172A", width=2)),
        textinfo="label+percent+value",
        hovertemplate="<b>%{label}</b>: %{value} locations (%{percent})<extra></extra>"
    )])
    
    fig.update_layout(
        title="<b>Flood Risk Level Distribution</b>",
        **CHART_LAYOUT_DEFAULTS
    )
    return fig

def create_rainfall_waterlevel_scatter(df: pd.DataFrame, risk_col: str = "flood_risk_label") -> go.Figure:
    """Scatter plot of Rainfall vs River Water Level colored by risk level."""
    fig = px.scatter(
        df,
        x="rainfall_mm",
        y="river_water_level_m",
        color=risk_col,
        color_discrete_map=RISK_COLORS,
        size="population",
        hover_name="location_name",
        hover_data=["taluk", "elevation_m", "distance_to_rescue_base_km"],
        labels={
            "rainfall_mm": "Rainfall (mm)",
            "river_water_level_m": "River Water Level (m)",
            risk_col: "Risk Level"
        },
        title="<b>Hydrological Hazard Matrix: Rainfall vs River Level</b>"
    )
    
    fig.update_layout(**CHART_LAYOUT_DEFAULTS)
    return fig

def create_resource_comparison_bar(inventory_dict: Dict[str, int], demand_dict: Dict[str, int], 
                                  allocated_dict: Dict[str, int]) -> go.Figure:
    """Grouped bar chart comparing Demand vs Available vs Allocated resources."""
    categories = list(demand_dict.keys())
    
    demands = [demand_dict.get(c, 0) for c in categories]
    availables = [inventory_dict.get(c, 0) for c in categories]
    allocateds = [allocated_dict.get(c, 0) for c in categories]
    
    fig = go.Figure(data=[
        go.Bar(name="Total Demand", x=categories, y=demands, marker_color="#EF4444"),
        go.Bar(name="Total Available", x=categories, y=availables, marker_color="#0284C7"),
        go.Bar(name="Actual Allocated", x=categories, y=allocateds, marker_color="#10B981")
    ])
    
    fig.update_layout(
        barmode="group",
        title="<b>Emergency Resources: Demand vs Availability vs Allocation</b>",
        yaxis_title="Units / Personnel",
        **CHART_LAYOUT_DEFAULTS
    )
    return fig

def create_confusion_matrix_heatmap(cm: List[List[int]], classes: Optional[List[str]] = None) -> go.Figure:
    """Confusion matrix heatmap."""
    if classes is None:
        classes = ["Low", "Medium", "High", "Critical"]
        
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"}
    ))
    
    fig.update_layout(
        title="<b>Model Confusion Matrix (Held-out Test Split)</b>",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        **CHART_LAYOUT_DEFAULTS
    )
    return fig

def create_feature_importance_bar(importances: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart for Classical feature importances."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features = [k.replace("_", " ").title() for k, v in sorted_items]
    values = [v for k, v in sorted_items]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color="#0284C7"
    ))
    
    fig.update_layout(
        title="<b>Classical Model Feature Importance (Gini Impurity)</b>",
        xaxis_title="Relative Importance Weight",
        yaxis=dict(autorange="reversed"),
        **CHART_LAYOUT_DEFAULTS
    )
    return fig

def create_vqc_training_history_chart(history: List[Dict[str, Any]]) -> go.Figure:
    """Line chart showing VQC Loss and Training Accuracy per epoch."""
    if not history:
        fig = go.Figure()
        fig.update_layout(title="No Training History Available", **CHART_LAYOUT_DEFAULTS)
        return fig
        
    df_h = pd.DataFrame(history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_h["epoch"], y=df_h["loss"],
        mode="lines+markers", name="Categorical Loss",
        line=dict(color="#EF4444", width=3)
    ))
    
    if "train_accuracy" in df_h.columns:
        fig.add_trace(go.Scatter(
            x=df_h["epoch"], y=df_h["train_accuracy"],
            mode="lines+markers", name="Train Accuracy",
            line=dict(color="#10B981", width=3), yaxis="y2"
        ))
        
    fig.update_layout(
        title="<b>VQC Optimization Convergence (PennyLane Simulation)</b>",
        xaxis_title="Epoch",
        yaxis=dict(title="Cross-Entropy Loss"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1.05]),
        **CHART_LAYOUT_DEFAULTS
    )
    return fig
