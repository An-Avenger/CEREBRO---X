"""Evaluation metrics for Cerebro X.

Provides specialized metrics for ordinal classification and imbalanced datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_predictions(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """
    Compute all standard metrics for the baseline models.
    
    Args:
        y_true: Ground truth ordinal labels (e.g. 0.0, 0.5, 1.0, 2.0).
        y_pred: Predicted labels.
        
    Returns:
        Dictionary of computed metric values.
    """
    y_true_float = np.asarray(y_true, dtype=float)
    y_pred_float = np.asarray(y_pred, dtype=float)
    
    y_true_str = np.asarray(y_true, dtype=str)
    y_pred_str = np.asarray(y_pred, dtype=str)
    
    # Standard classification metrics
    acc = accuracy_score(y_true_str, y_pred_str)
    b_acc = balanced_accuracy_score(y_true_str, y_pred_str)
    f1_macro = f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)
    
    # Ordinal metric (MAE treats CDR classes as continuous distances)
    mae = mean_absolute_error(y_true_float, y_pred_float)
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(b_acc),
        "f1_macro": float(f1_macro),
        "mae": float(mae),
    }


def plot_confusion_matrix_custom(
    y_true: np.ndarray | pd.Series, 
    y_pred: np.ndarray | pd.Series, 
    classes: list[float],
    title: str,
    output_path: str | Path,
) -> None:
    """
    Generate and save a customized confusion matrix plot.
    """
    y_true_str = np.asarray(y_true, dtype=str)
    y_pred_str = np.asarray(y_pred, dtype=str)
    
    cm = confusion_matrix(y_true_str, y_pred_str, labels=[str(c) for c in classes])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=[str(c) for c in classes],
        yticklabels=[str(c) for c in classes],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True CDR")
    ax.set_xlabel("Predicted CDR")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
