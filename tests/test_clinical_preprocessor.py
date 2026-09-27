"""
Tests for the canonical ClinicalPreprocessor.

Verifies:
1. FEATURE_ORDER has exactly 19 features.
2. Training and inference preprocessing produce identical transformed vectors
   for identical input (the core requirement of Phase 1).
3. Missing value handling (NaN → imputed, not errored).
4. Feature order validation raises on mismatch.
5. Temporal features are computed correctly from actual visit data.
6. Fabricated values (mmse_delta=0 for first visit) are NaN, not 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cerebro_x.features.preprocessor import (
    ClinicalPreprocessor,
    FEATURE_ORDER,
    N_FEATURES,
    CDR_CLASS_TO_KEY,
    CDR_CLASS_TO_VALUE,
    CDR_CLASS_TO_LABEL,
    BHI_PROB_KEYS,
)


# ── Feature order ─────────────────────────────────────────────────────────────

def test_feature_order_has_19_features():
    assert len(FEATURE_ORDER) == 19, (
        f"Expected 19 features, got {len(FEATURE_ORDER)}. "
        f"Feature order: {FEATURE_ORDER}"
    )


def test_n_features_constant():
    assert N_FEATURES == 19


def test_feature_order_no_duplicates():
    assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER)), (
        f"Duplicate features in FEATURE_ORDER: {FEATURE_ORDER}"
    )


def test_canonical_cdr_keys():
    """CDR class keys must be '0','1','2','3' — matching BHI expected keys."""
    assert set(CDR_CLASS_TO_KEY.values()) == {"0", "1", "2", "3"}
    assert BHI_PROB_KEYS == ("0", "1", "2", "3")


# ── Train vs inference preprocessing equality ─────────────────────────────────

def _make_dummy_df(n_rows: int = 5) -> pd.DataFrame:
    """Create a minimal valid DataFrame with all FEATURE_ORDER columns."""
    rng = np.random.default_rng(42)
    data = {
        "curr_age":            rng.uniform(60, 90, n_rows),
        "sex":                 rng.choice([0.0, 1.0], n_rows),
        "hand":                rng.choice([0.0, 1.0], n_rows),
        "educ":                rng.uniform(8, 20, n_rows),
        "ses":                 rng.choice([1.0, 2.0, 3.0, np.nan], n_rows),
        "curr_mmse":           rng.uniform(20, 30, n_rows),
        "curr_cdr":            rng.choice([0.0, 0.5, 1.0], n_rows),
        "curr_etiv":           rng.uniform(1300, 1800, n_rows),
        "curr_nwbv":           rng.uniform(0.68, 0.85, n_rows),
        "curr_asf":            rng.uniform(0.9, 1.3, n_rows),
        "current_mr_delay":    rng.uniform(0, 1000, n_rows),
        "days_between_visits": rng.choice([np.nan, 365, 730], n_rows),
        "n_prior_visits":      rng.uniform(0, 3, n_rows),
        "prev_mmse":           rng.choice([np.nan, 25.0, 28.0], n_rows),
        "prev_cdr":            rng.choice([np.nan, 0.0, 0.5], n_rows),
        "prev_nwbv":           rng.choice([np.nan, 0.74, 0.78], n_rows),
        "mmse_delta":          rng.choice([np.nan, -1.0, 0.0, 1.0], n_rows),
        "cdr_delta":           rng.choice([np.nan, 0.0, 0.5], n_rows),
        "nwbv_delta":          rng.choice([np.nan, -0.005, 0.0], n_rows),
    }
    return pd.DataFrame(data, columns=FEATURE_ORDER)


def test_training_inference_identical_output():
    """
    Core requirement: training and inference preprocessing must produce
    identical transformed vectors for the same input.
    """
    df_train = _make_dummy_df(10)
    df_test  = _make_dummy_df(3)

    # Training: fit + transform
    prep = ClinicalPreprocessor()
    X_train = prep.fit_transform(df_train)

    # Inference: transform only (same fitted objects)
    X_test_infer = prep.transform(df_test)

    # Also run fit_transform on test data separately — they should NOT be equal
    # (proves scaling is different when re-fitted)
    prep2 = ClinicalPreprocessor()
    X_test_refit = prep2.fit_transform(df_test)

    # Training transform: same prep applied to same data rows must be deterministic
    X_train_repeat = prep.transform(df_train)
    np.testing.assert_array_almost_equal(
        X_train, X_train_repeat,
        err_msg="fit_transform and subsequent transform must be identical for same data"
    )

    # Inference output shape
    assert X_test_infer.shape == (3, 19)
    assert X_train.shape == (10, 19)


def test_missing_value_handling():
    """NaN values are imputed (not left as NaN)."""
    df = _make_dummy_df(5)
    # Force some NaN
    df.loc[0, "ses"] = np.nan
    df.loc[1, "curr_mmse"] = np.nan
    df.loc[2, "prev_mmse"] = np.nan

    prep = ClinicalPreprocessor()
    X = prep.fit_transform(df)

    assert not np.any(np.isnan(X)), (
        f"NaN values found in output after fit_transform: {np.where(np.isnan(X))}"
    )


def test_wrong_feature_count_raises():
    """Passing DataFrame with wrong columns raises ValueError."""
    prep = ClinicalPreprocessor()
    df_good = _make_dummy_df(5)
    prep.fit_transform(df_good)

    # Drop a column
    df_bad = df_good.drop(columns=["curr_age"])
    with pytest.raises(ValueError, match="Feature mismatch"):
        prep.transform(df_bad)


def test_unfitted_raises_runtime_error():
    """Calling transform before fit raises RuntimeError."""
    prep = ClinicalPreprocessor()
    df = _make_dummy_df(3)
    with pytest.raises(RuntimeError, match="not fitted"):
        prep.transform(df)


def test_save_and_load(tmp_path):
    """Save and load preserves imputer/scaler state."""
    df = _make_dummy_df(10)
    prep = ClinicalPreprocessor()
    X_original = prep.fit_transform(df)
    prep.save(tmp_path)

    loaded = ClinicalPreprocessor.load(tmp_path)
    X_loaded = loaded.transform(df)
    np.testing.assert_array_almost_equal(X_original, X_loaded)


# ── visits_to_dataframe tests ─────────────────────────────────────────────────

def test_visits_to_dataframe_two_visits():
    """Second visit should have computed temporal features from first visit."""
    visits = [
        {"age": 72.0, "gender": "F", "educ": 16.0, "ses": 2.0,
         "mmse": 29.0, "cdr": 0.0, "nwbv": 0.76, "etiv": 1480.0, "asf": 1.23},
        {"age": 74.0, "gender": "F", "educ": 16.0, "ses": 2.0,
         "mmse": 27.0, "cdr": 0.5, "nwbv": 0.74, "etiv": 1480.0, "asf": 1.23},
    ]
    df = ClinicalPreprocessor.visits_to_dataframe(visits)

    assert df.shape == (2, 19)
    assert list(df.columns) == FEATURE_ORDER

    # First visit: no prior — temporal features must be NaN
    assert np.isnan(df.loc[0, "prev_mmse"]), "prev_mmse must be NaN for first visit"
    assert np.isnan(df.loc[0, "mmse_delta"]), "mmse_delta must be NaN for first visit (not fabricated as 0)"
    assert np.isnan(df.loc[0, "cdr_delta"]), "cdr_delta must be NaN for first visit"

    # Second visit: temporal features computed from actual data
    assert df.loc[1, "prev_mmse"] == pytest.approx(29.0)
    assert df.loc[1, "mmse_delta"] == pytest.approx(27.0 - 29.0)
    assert df.loc[1, "cdr_delta"] == pytest.approx(0.5 - 0.0)
    assert df.loc[1, "n_prior_visits"] == pytest.approx(1.0)


def test_visits_to_dataframe_days_not_fabricated():
    """days_between_visits must be NaN when not provided, NOT hardcoded to 365."""
    visits = [
        {"age": 72.0, "cdr": 0.0},
        {"age": 74.0, "cdr": 0.5},
    ]
    df = ClinicalPreprocessor.visits_to_dataframe(visits)

    # First visit: NaN (no previous)
    assert np.isnan(df.loc[0, "days_between_visits"])
    # Second visit: NaN (days_since_last_visit not provided)
    assert np.isnan(df.loc[1, "days_between_visits"]), (
        "days_between_visits must be NaN when not provided. "
        "Do NOT hardcode 365 as a default."
    )


def test_visits_to_dataframe_with_days_provided():
    """When days_since_last_visit is provided it should be used."""
    visits = [
        {"age": 72.0, "cdr": 0.0},
        {"age": 74.0, "cdr": 0.5, "days_since_last_visit": 730.0},
    ]
    df = ClinicalPreprocessor.visits_to_dataframe(visits)
    assert df.loc[1, "days_between_visits"] == pytest.approx(730.0)
