"""Explainability module for Cerebro-X."""
from .shap_temporal import compute_temporal_shap, TemporalModelWrapper
from .visualizations import plot_temporal_shap_summary, plot_temporal_shap_waterfall
from .gradcam import GradCAM3D

__all__ = [
    "compute_temporal_shap",
    "TemporalModelWrapper",
    "plot_temporal_shap_summary",
    "plot_temporal_shap_waterfall",
    "GradCAM3D",
]
