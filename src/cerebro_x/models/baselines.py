"""Baseline models for predicting next-visit CDR.

Provides factory functions to initialize standard scikit-learn models
with appropriate hyperparameters for our imbalanced, ordinal task.
"""
from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


def get_dummy_classifier(strategy: str = "most_frequent", seed: int = 42) -> Pipeline:
    """
    Returns a naive baseline model.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy=strategy, random_state=seed)),
    ])


def get_logistic_regression(seed: int = 42, max_iter: int = 1000) -> Pipeline:
    """
    Returns a logistic regression baseline.
    
    Includes StandardScaler (essential for regularized linear models)
    and uses class_weight="balanced" to handle severe CDR class imbalance.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            multi_class="multinomial",
            max_iter=max_iter,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def get_random_forest(seed: int = 42, n_estimators: int = 100) -> Pipeline:
    """
    Returns a Random Forest baseline.
    
    Uses class_weight="balanced_subsample" to handle class imbalance
    across bootstrap samples.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # RF doesn't strictly need scaling, but we include it for consistency
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )),
    ])


class LastVisitClassifier(BaseEstimator, ClassifierMixin):
    """
    A baseline model that simply predicts the future state will be
    identical to the current state.
    """
    def __init__(self, cdr_column: str = "curr_cdr"):
        self.cdr_column = cdr_column
        self.classes_ = np.array(["0.0", "0.5", "1.0", "2.0"])
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        if self.cdr_column in X.columns:
            # Assumes pandas DataFrame
            return X[self.cdr_column].astype(str).values
        else:
            # Fallback if passed array and we know the index (approx)
            # Not robust but handles edge cases if passed raw numpy array
            raise ValueError(f"LastVisitClassifier requires DataFrame with '{self.cdr_column}'")


def get_all_baselines(seed: int = 42) -> dict[str, Pipeline]:
    """Returns a dictionary of all initialized baseline models."""
    return {
        "Dummy (Majority)": get_dummy_classifier(seed=seed),
        "Logistic Regression": get_logistic_regression(seed=seed),
        "Random Forest": get_random_forest(seed=seed),
        "Last Visit Baseline": LastVisitClassifier(cdr_column="curr_cdr"),
    }
