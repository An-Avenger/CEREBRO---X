"""Tests for baseline models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cerebro_x.models.baselines import get_all_baselines

def test_models_initialize():
    """All models from the factory should initialize as Pipelines."""
    models = get_all_baselines()
    
    assert "Dummy (Majority)" in models
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    
    for name, model in models.items():
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

def test_models_fit_predict():
    """Models should fit and predict on synthetic data without errors."""
    models = get_all_baselines()
    
    # Synthetic data with missing values to test imputer
    X = pd.DataFrame({
        "feature1": [1, 2, np.nan, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "curr_cdr": [0.0, 0.5, 1.0, 0.0, 2.0]  # Required by LastVisitClassifier
    })
    y = np.array(["0.0", "0.0", "0.5", "1.0", "2.0"])
    
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
