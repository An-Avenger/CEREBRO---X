"""Visualizations for explainability.

Includes temporal SHAP waterfall/summary plots and MRI Grad-CAM heatmaps.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def plot_temporal_shap_summary(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    max_display: int = 15,
) -> None:
    """Plots a summary of temporal SHAP values.
    
    Since sequence SHAP returns (num_test, seq_len, num_features), we flatten
    the time dimension into features (e.g. "Age_t-2", "Age_t-1", "Age_t") 
    to see which features at which timesteps matter most.
    
    Args:
        shap_values: Array of shape (num_test, seq_len, num_features).
        features: Array of shape (num_test, seq_len, num_features).
        feature_names: List of strings for the raw features.
        output_path: Where to save the plot.
        max_display: Maximum number of features to show.
    """
    num_test, seq_len, num_features = shap_values.shape
    
    # Flatten the sequence and feature dimensions
    flat_shap = shap_values.reshape(num_test, seq_len * num_features)
    flat_features = features.reshape(num_test, seq_len * num_features)
    
    # Create temporal feature names
    temporal_names = []
    for t in range(seq_len):
        time_suffix = f" (Visit -{seq_len - t - 1})" if (seq_len - t - 1) > 0 else " (Current Visit)"
        for name in feature_names:
            temporal_names.append(f"{name}{time_suffix}")
            
    plt.figure(figsize=(10, 8))
    
    # Generate the SHAP summary plot
    shap.summary_plot(
        flat_shap,
        flat_features,
        feature_names=temporal_names,
        max_display=max_display,
        show=False,
    )
    
    plt.title("Longitudinal Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved temporal SHAP summary to %s", output_path)


def plot_temporal_shap_waterfall(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    expected_value: float,
    sample_idx: int,
    output_path: str | Path,
) -> None:
    """Plots a waterfall plot for a single temporal prediction.
    
    Args:
        shap_values: Array of shape (num_test, seq_len, num_features).
        features: Array of shape (num_test, seq_len, num_features).
        feature_names: List of strings for the raw features.
        expected_value: The base value (e.g. mean model output) from the explainer.
        sample_idx: Which sample in the test set to plot.
        output_path: Where to save the plot.
    """
    num_test, seq_len, num_features = shap_values.shape
    
    flat_shap = shap_values[sample_idx].reshape(seq_len * num_features)
    flat_features = features[sample_idx].reshape(seq_len * num_features)
    
    temporal_names = []
    for t in range(seq_len):
        time_suffix = f" (t-{seq_len - t - 1})" if (seq_len - t - 1) > 0 else " (t)"
        for name in feature_names:
            temporal_names.append(f"{name}{time_suffix}")
            
    # Create an Explanation object required for new SHAP waterfall plots
    explanation = shap.Explanation(
        values=flat_shap,
        base_values=expected_value,
        data=flat_features,
        feature_names=temporal_names,
    )
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title(f"Local Explanation for Patient {sample_idx}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP waterfall plot to %s", output_path)
