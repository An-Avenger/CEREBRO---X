"""Feature engineering for the OASIS-2 clinical pipeline.

Defines feature groups and builds the model-ready feature matrix.
Each feature group is documented with its source, rationale, and missingness behavior.

CRITICAL: The TARGET columns (next_CDR, next_MMSE) must NEVER be added to
the feature matrix. This module enforces that via explicit exclusion.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cerebro_x.data.schemas import PairColumns

logger = logging.getLogger(__name__)

# Columns that must never appear in the feature matrix
FORBIDDEN_IN_FEATURES = {PairColumns.NEXT_CDR, PairColumns.NEXT_MMSE}


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, dict] = {
    "static_demographic": {
        "columns": ["curr_age", "sex", "hand", "educ", "ses"],
        "description": "Static/slowly-changing demographic variables from current visit.",
        "missingness_policy": {
            "ses": "median imputation on training data only",
        },
        "encoding": {
            "sex": "binary: M=1, F=0",
            "hand": "binary: R=1, L=0 (ambidextrous mapped to 0.5)",
        },
    },
    "clinical_state": {
        "columns": ["curr_mmse", "curr_cdr"],
        "description": "Current-visit cognitive assessment scores.",
        "missingness_policy": {
            "curr_mmse": "median imputation on training data only",
        },
        "note": "curr_cdr is the CURRENT CDR (input), not next_CDR (target).",
    },
    "volumetric": {
        "columns": ["curr_etiv", "curr_nwbv", "curr_asf"],
        "description": (
            "MRI-derived scalar volume measures from current visit. "
            "NOTE: These are scalars derived from MRI, not raw MRI tensors."
        ),
        "missingness_policy": "these columns should be complete per OASIS-2 documentation",
    },
    "timing": {
        "columns": ["current_mr_delay", "days_between_visits", "n_prior_visits"],
        "description": "Visit timing and longitudinal structure features.",
        "missingness_policy": "should be complete",
    },
    "temporal_engineered": {
        "columns": [
            "prev_mmse",
            "prev_cdr",
            "prev_nwbv",
            "mmse_delta",
            "cdr_delta",
            "nwbv_delta",
        ],
        "description": (
            "Engineered longitudinal features using information from the immediately "
            "preceding visit. NaN for first-visit pairs (no prior visit available)."
        ),
        "missingness_policy": {
            "prev_*": "NaN for first visit of each subject — impute with 0 or flag",
            "mmse_delta": "NaN for first visit — impute with 0",
            "cdr_delta": "NaN for first visit — impute with 0",
            "nwbv_delta": "NaN for first visit — impute with 0",
        },
        "leakage_note": (
            "All temporal features use ONLY information from the PREVIOUS visit "
            "(index i-1), not from the target visit (index i+1). "
            "This is verified in build_longitudinal_pairs.py."
        ),
    },
}


@dataclass
class FeatureSchema:
    """Documents exactly which features were included and how they were engineered."""

    feature_groups_active: list[str]
    feature_columns: list[str]
    categorical_encodings: dict = field(default_factory=dict)
    missingness_policies: dict = field(default_factory=dict)
    excluded_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def get_feature_columns(active_groups: list[str] | None = None) -> list[str]:
    """
    Return the ordered list of feature columns for the specified groups.

    Args:
        active_groups: List of group names to include.
                       If None, all groups are included.

    Returns:
        Ordered list of column names.

    Raises:
        ValueError: If an unknown group name is specified.
    """
    if active_groups is None:
        active_groups = list(FEATURE_GROUPS.keys())

    unknown = set(active_groups) - set(FEATURE_GROUPS.keys())
    if unknown:
        raise ValueError(
            f"Unknown feature groups: {unknown}. "
            f"Valid groups: {list(FEATURE_GROUPS.keys())}"
        )

    cols: list[str] = []
    for group in active_groups:
        cols.extend(FEATURE_GROUPS[group]["columns"])

    return cols


def build_feature_matrix(
    pairs_df: pd.DataFrame,
    active_groups: list[str] | None = None,
) -> tuple[pd.DataFrame, FeatureSchema]:
    """
    Build the input feature matrix X from the longitudinal pairs DataFrame.

    This function:
      1. Selects the specified feature columns.
      2. Encodes categorical variables (sex, hand).
      3. Verifies no target columns are included.
      4. Returns X and a FeatureSchema documenting what was built.

    Args:
        pairs_df:       Output of build_longitudinal_pairs().
        active_groups:  Feature group names to include. None = all groups.

    Returns:
        Tuple of (X DataFrame, FeatureSchema).

    Raises:
        ValueError: If any target column appears in the feature matrix.
        KeyError:   If a required column is missing from pairs_df.
    """
    feature_cols = get_feature_columns(active_groups)

    # Check required columns exist
    missing = [c for c in feature_cols if c not in pairs_df.columns]
    if missing:
        raise KeyError(
            f"Feature columns not found in pairs DataFrame: {missing}\n"
            f"Available columns: {list(pairs_df.columns)}"
        )

    X = pairs_df[feature_cols].copy()

    # ── Categorical encoding ─────────────────────────────────────────────────
    encodings: dict = {}

    if "sex" in X.columns:
        X["sex"] = X["sex"].map({"M": 1, "F": 0})
        encodings["sex"] = {"M": 1, "F": 0, "unknown": "NaN"}

    if "hand" in X.columns:
        X["hand"] = X["hand"].map({"R": 1, "L": 0}).fillna(0.5)
        encodings["hand"] = {"R": 1, "L": 0, "ambidextrous/unknown": 0.5}

    # curr_group is ordinal but NOT reliable for prediction without leakage risk.
    # Drop it from X if present — it contains diagnostic labels that may correlate
    # too directly with CDR.
    if "curr_group" in X.columns:
        X = X.drop(columns=["curr_group"])
        logger.warning(
            "curr_group dropped from feature matrix. "
            "It contains Group labels (Nondemented/Demented/Converted) that have "
            "high correlation with CDR and require careful leakage analysis before use."
        )

    # ── Safety check: no target columns in X ────────────────────────────────
    leaking = FORBIDDEN_IN_FEATURES & set(X.columns)
    if leaking:
        raise ValueError(
            f"TARGET LEAKAGE: The following target columns were found in the "
            f"feature matrix: {leaking}. This is a critical bug. "
            f"Fix build_feature_matrix() immediately."
        )

    # ── Feature schema ───────────────────────────────────────────────────────
    active_groups_actual = active_groups or list(FEATURE_GROUPS.keys())
    policies: dict = {}
    for g in active_groups_actual:
        mp = FEATURE_GROUPS[g].get("missingness_policy", "")
        if mp:
            policies[g] = mp

    schema = FeatureSchema(
        feature_groups_active=active_groups_actual,
        feature_columns=list(X.columns),
        categorical_encodings=encodings,
        missingness_policies=policies,
        excluded_columns=list(FORBIDDEN_IN_FEATURES),
        notes=[
            "curr_group excluded due to leakage risk — see feature docstring.",
            "Temporal features (prev_*, *_delta) are NaN for first-visit pairs.",
        ],
    )

    logger.info(
        "Feature matrix built: %d rows × %d features. Missing: %s",
        len(X),
        len(X.columns),
        X.isnull().sum()[X.isnull().sum() > 0].to_dict(),
    )

    return X, schema


def save_feature_schema(schema: FeatureSchema, output_path: str | Path) -> None:
    """Save feature schema to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(schema.to_dict(), fh, indent=2)
    logger.info("Feature schema saved: %s", output_path)
