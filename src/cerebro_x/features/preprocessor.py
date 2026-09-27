"""
Canonical clinical preprocessing for Cerebro X.

This module defines the SINGLE source of truth for:
  - Feature column ordering (FEATURE_ORDER)
  - Categorical encoding (sex, hand)
  - Missing-value imputation (median)
  - Standard scaling

The same ClinicalPreprocessor instance MUST be used for:
  training, validation, test, /predict/clinical,
  /predict/bimodal, /brain-twin/extract, and /explain/clinical.

Usage:
    # Training
    prep = ClinicalPreprocessor()
    X = prep.fit_transform(visits_df)
    prep.save(artifact_dir)

    # Inference
    prep = ClinicalPreprocessor.load(artifact_dir)
    X = prep.transform(visits_df)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Canonical feature order ───────────────────────────────────────────────────
# THIS IS THE SINGLE SOURCE OF TRUTH.
# Do not change without retraining the model and updating the manifest.
FEATURE_ORDER: list[str] = [
    "curr_age",        # 0  — current age
    "sex",             # 1  — M=1, F=0
    "hand",            # 2  — R=1, L=0, ambidextrous=0.5
    "educ",            # 3  — years of education
    "ses",             # 4  — socioeconomic status (1-5); median-imputed
    "curr_mmse",       # 5  — current MMSE (0-30); median-imputed
    "curr_cdr",        # 6  — current CDR (input, NOT target)
    "curr_etiv",       # 7  — current eTIV (mm³)
    "curr_nwbv",       # 8  — current nWBV
    "curr_asf",        # 9  — current ASF
    "current_mr_delay",# 10 — MR delay
    "days_between_visits", # 11 — days since last visit
    "n_prior_visits",  # 12 — number of prior visits
    "prev_mmse",       # 13 — previous MMSE (NaN for first visit)
    "prev_cdr",        # 14 — previous CDR (NaN for first visit)
    "prev_nwbv",       # 15 — previous nWBV (NaN for first visit)
    "mmse_delta",      # 16 — MMSE change from previous visit
    "cdr_delta",       # 17 — CDR change from previous visit
    "nwbv_delta",      # 18 — nWBV change from previous visit
]

N_FEATURES = len(FEATURE_ORDER)  # must be 19

# ── API field → FEATURE_ORDER column mapping ──────────────────────────────────
# Maps the names used in the VisitInput schema to FEATURE_ORDER column names.
API_FIELD_MAP = {
    "age":    "curr_age",
    "gender": "sex",
    "educ":   "educ",
    "ses":    "ses",
    "mmse":   "curr_mmse",
    "cdr":    "curr_cdr",
    "etiv":   "curr_etiv",
    "nwbv":   "curr_nwbv",
    "asf":    "curr_asf",
}

# Canonical CDR class mapping — used by BHI and inference
# class_index → CDR float value
CDR_CLASS_TO_VALUE = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}
# class_index → short string key (used in class_probabilities dict)
CDR_CLASS_TO_KEY = {0: "0", 1: "1", 2: "2", 3: "3"}
# class_index → human label
CDR_CLASS_TO_LABEL = {
    0: "Normal (CDR 0.0)",
    1: "Very Mild Dementia (CDR 0.5)",
    2: "Mild Dementia (CDR 1.0)",
    3: "Moderate Dementia (CDR 2.0)",
}
# BHI expects these exact string keys in the probabilities dict:
BHI_PROB_KEYS = ("0", "1", "2", "3")

_PREPROCESSOR_FILENAME = "clinical_preprocessor.pkl"
_MANIFEST_FILENAME = "preprocessor_manifest.json"


class ClinicalPreprocessor:
    """
    Encapsulates imputation + scaling for the 19-feature clinical vector.

    Attributes:
        imputer:      Fitted SimpleImputer(strategy='median').
        scaler:       Fitted StandardScaler.
        feature_order: Ordered list of feature column names (== FEATURE_ORDER).
        is_fitted:    True once fit_transform() has been called.
    """

    def __init__(self) -> None:
        self.imputer: SimpleImputer = SimpleImputer(strategy="median")
        self.scaler: StandardScaler = StandardScaler()
        self.feature_order: list[str] = list(FEATURE_ORDER)
        self.is_fitted: bool = False

    # ── Training-time path ────────────────────────────────────────────────────

    def fit_transform(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Fit imputer and scaler on training data, then transform.

        Args:
            X_df: DataFrame with columns exactly matching FEATURE_ORDER.

        Returns:
            Transformed numpy array (n_samples, 19).

        Raises:
            ValueError: If feature count or order does not match.
        """
        self._validate_columns(X_df)
        X_raw = X_df[self.feature_order].values.astype(np.float64)
        X_imp = self.imputer.fit_transform(X_raw)
        X_scaled = self.scaler.fit_transform(X_imp)
        self.is_fitted = True
        logger.info(
            "ClinicalPreprocessor fitted on %d samples × %d features.",
            len(X_df), N_FEATURES,
        )
        return np.asarray(X_scaled, dtype=np.float32)

    # ── Inference-time path ───────────────────────────────────────────────────

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using already-fitted imputer and scaler.

        Args:
            X_df: DataFrame with columns matching FEATURE_ORDER.

        Returns:
            Transformed numpy array.

        Raises:
            RuntimeError: If preprocessor has not been fitted/loaded.
            ValueError:   If feature order or count mismatches.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ClinicalPreprocessor is not fitted. "
                "Either call fit_transform() on training data, "
                "or load a saved preprocessor via ClinicalPreprocessor.load()."
            )
        self._validate_columns(X_df)
        X_raw = X_df[self.feature_order].values.astype(np.float64)
        X_imp = self.imputer.transform(X_raw)
        X_scaled = self.scaler.transform(X_imp)
        return np.asarray(X_scaled, dtype=np.float32)

    # ── Visit-dict → feature DataFrame ───────────────────────────────────────

    @staticmethod
    def visits_to_dataframe(visits: list[dict]) -> pd.DataFrame:
        """
        Convert a list of API visit dicts into a DataFrame with FEATURE_ORDER columns.

        This method COMPUTES temporal features (delta columns, prev_* columns,
        days_between_visits, n_prior_visits) from the actual visit sequence.
        It does NOT fabricate values silently.

        Args:
            visits: List of dicts from VisitInput.model_dump(), in chronological order.
                    Required keys: age, cdr.
                    Optional: educ, ses, mmse, nwbv, etiv, asf, gender, hand,
                              mr_delay, days_since_last_visit.

        Returns:
            DataFrame with exactly the FEATURE_ORDER columns, one row per visit.
            Missing values are left as NaN (the imputer handles them).
        """
        rows = []
        for i, v in enumerate(visits):
            # Categorical encoding
            gender_raw = v.get("gender", None) or v.get("sex", None)
            sex_encoded: Optional[float]
            if gender_raw == "M":
                sex_encoded = 1.0
            elif gender_raw == "F":
                sex_encoded = 0.0
            else:
                sex_encoded = np.nan  # missing → imputed

            hand_raw = v.get("hand", None)
            if hand_raw == "R":
                hand_encoded: Optional[float] = 1.0
            elif hand_raw == "L":
                hand_encoded = 0.0
            elif hand_raw is not None:
                hand_encoded = 0.5  # ambidextrous
            else:
                hand_encoded = np.nan

            # Temporal features: only compute from actual prior visit, never fabricate
            if i > 0:
                prev = visits[i - 1]
                prev_mmse = prev.get("mmse", None)
                prev_cdr  = prev.get("cdr",  None)
                prev_nwbv = prev.get("nwbv", None)
                curr_mmse = v.get("mmse", None)
                curr_cdr  = v.get("cdr",  None)
                curr_nwbv = v.get("nwbv", None)

                mmse_delta = (curr_mmse - prev_mmse) if (curr_mmse is not None and prev_mmse is not None) else np.nan
                cdr_delta  = (curr_cdr  - prev_cdr)  if (curr_cdr  is not None and prev_cdr  is not None) else np.nan
                nwbv_delta = (curr_nwbv - prev_nwbv) if (curr_nwbv is not None and prev_nwbv is not None) else np.nan

                # days_between_visits: use provided value if available, else NaN
                days_between = v.get("days_since_last_visit", None)
                if days_between is None:
                    days_between = np.nan  # NOT fabricated as 365
            else:
                prev_mmse = np.nan
                prev_cdr  = np.nan
                prev_nwbv = np.nan
                mmse_delta = np.nan
                cdr_delta  = np.nan
                nwbv_delta = np.nan
                days_between = np.nan

            row = {
                "curr_age":            v.get("age",      np.nan),
                "sex":                 sex_encoded,
                "hand":                hand_encoded,
                "educ":                v.get("educ",     np.nan),
                "ses":                 v.get("ses",      np.nan),
                "curr_mmse":           v.get("mmse",     np.nan),
                "curr_cdr":            v.get("cdr",      np.nan),
                "curr_etiv":           v.get("etiv",     np.nan),
                "curr_nwbv":           v.get("nwbv",     np.nan),
                "curr_asf":            v.get("asf",      np.nan),
                "current_mr_delay":    v.get("mr_delay", 0.0),
                "days_between_visits": days_between,
                "n_prior_visits":      float(i),
                "prev_mmse":           prev_mmse,
                "prev_cdr":            prev_cdr,
                "prev_nwbv":           prev_nwbv,
                "mmse_delta":          mmse_delta,
                "cdr_delta":           cdr_delta,
                "nwbv_delta":          nwbv_delta,
            }
            rows.append(row)

        df = pd.DataFrame(rows, columns=FEATURE_ORDER)
        return df

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        """
        Save the fitted preprocessor to disk.

        Saves:
          {directory}/clinical_preprocessor.pkl   — imputer + scaler
          {directory}/preprocessor_manifest.json  — feature list, version

        Returns:
            Path to the saved .pkl file.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        pkl_path = directory / _PREPROCESSOR_FILENAME
        joblib.dump(
            {
                "imputer": self.imputer,
                "scaler": self.scaler,
                "feature_order": self.feature_order,
            },
            pkl_path,
        )

        manifest = {
            "feature_order": self.feature_order,
            "n_features": len(self.feature_order),
            "imputer_strategy": self.imputer.strategy,
            "scaler_class": type(self.scaler).__name__,
            "version": "1.0",
        }
        manifest_path = directory / _MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info("ClinicalPreprocessor saved to %s", directory)
        return pkl_path

    @classmethod
    def load(cls, directory: str | Path) -> "ClinicalPreprocessor":
        """
        Load a previously saved ClinicalPreprocessor from disk.

        Args:
            directory: Directory containing clinical_preprocessor.pkl.

        Returns:
            A fitted ClinicalPreprocessor instance.

        Raises:
            FileNotFoundError: If the .pkl file does not exist.
            ValueError:        If stored feature order does not match FEATURE_ORDER.
        """
        directory = Path(directory)
        pkl_path = directory / _PREPROCESSOR_FILENAME

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"ClinicalPreprocessor not found at {pkl_path}. "
                "Ensure the model was trained with save_preprocessing=True, "
                "or run scripts/train_longitudinal.py to produce the artifact."
            )

        state = joblib.load(pkl_path)
        obj = cls()
        obj.imputer = state["imputer"]
        obj.scaler = state["scaler"]
        obj.feature_order = state["feature_order"]
        obj.is_fitted = True

        # Validate feature order matches current code
        if obj.feature_order != FEATURE_ORDER:
            raise ValueError(
                f"Loaded preprocessor feature order does not match current FEATURE_ORDER.\n"
                f"Loaded:  {obj.feature_order}\n"
                f"Current: {FEATURE_ORDER}\n"
                "The checkpoint may be from a different version of the model. "
                "Retrain to get a compatible preprocessor."
            )

        logger.info("ClinicalPreprocessor loaded from %s (%d features)", directory, len(obj.feature_order))
        return obj

    # ── Internals ─────────────────────────────────────────────────────────────

    def _validate_columns(self, X_df: pd.DataFrame) -> None:
        """Check that X_df has exactly the expected feature columns."""
        missing = [c for c in self.feature_order if c not in X_df.columns]
        extra   = [c for c in X_df.columns if c not in self.feature_order]
        if missing or extra:
            raise ValueError(
                f"Feature mismatch.\n"
                f"  Missing columns: {missing}\n"
                f"  Unexpected columns: {extra}\n"
                f"  Expected: {self.feature_order}"
            )
        if len(X_df.columns) != len(self.feature_order):
            raise ValueError(
                f"Expected {len(self.feature_order)} features, got {len(X_df.columns)}."
            )
