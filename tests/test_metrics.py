"""Tests for evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cerebro_x.evaluation.metrics import evaluate_predictions

def test_evaluate_predictions_perfect():
    """Metrics should reflect perfect prediction."""
    y_true = [0.0, 0.5, 1.0, 2.0]
    y_pred = [0.0, 0.5, 1.0, 2.0]
    
    metrics = evaluate_predictions(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["mae"] == 0.0

def test_evaluate_predictions_imperfect():
    """Metrics should accurately compute error for imperfect prediction."""
    y_true = [0.0, 0.0, 1.0, 1.0]
    y_pred = [0.0, 0.5, 1.0, 2.0]
    
    metrics = evaluate_predictions(y_true, y_pred)
    
    assert metrics["accuracy"] == 0.5
    # balanced accuracy:
    # class 0.0: 1/2 correct (0.5)
    # class 1.0: 1/2 correct (0.5)
    # macro avg: 0.5
    assert metrics["balanced_accuracy"] == 0.5
    
    # MAE
    # |0.0 - 0.0| + |0.0 - 0.5| + |1.0 - 1.0| + |1.0 - 2.0|
    # = 0 + 0.5 + 0 + 1.0 = 1.5 / 4 = 0.375
    assert metrics["mae"] == 0.375

def test_evaluate_predictions_handles_pandas():
    """Should seamlessly handle pandas Series."""
    y_true = pd.Series([0.0, 0.5, 1.0, 2.0])
    y_pred = pd.Series([0.0, 0.5, 1.0, 2.0])
    
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
