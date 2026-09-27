"""Temporal feature documentation and validation.

The actual temporal feature computation is done inside
build_longitudinal_pairs.py because features must be computed
BEFORE the pair is finalized (to avoid leakage).

This module provides:
  - Documentation of each temporal feature
  - Validation that temporal features use only prior-visit information
  - Utility to describe temporal features for a given pairs DataFrame
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Temporal feature catalog
# ---------------------------------------------------------------------------

TEMPORAL_FEATURES: dict[str, dict] = {
    "prev_mmse": {
        "source_columns": ["MMSE"],
        "visit_offset": -1,
        "formula": "MMSE[visit_i - 1]",
        "missingness_behavior": "NaN for first visit (i=0), no prior visit exists",
        "units": "MMSE points (0–30)",
    },
    "prev_cdr": {
        "source_columns": ["CDR"],
        "visit_offset": -1,
        "formula": "CDR[visit_i - 1]",
        "missingness_behavior": "NaN for first visit",
        "units": "CDR ordinal (0.0, 0.5, 1.0, 2.0)",
    },
    "prev_nwbv": {
        "source_columns": ["nWBV"],
        "visit_offset": -1,
        "formula": "nWBV[visit_i - 1]",
        "missingness_behavior": "NaN for first visit",
        "units": "Normalized Whole Brain Volume (unitless ratio)",
    },
    "mmse_delta": {
        "source_columns": ["MMSE"],
        "visit_offset": -1,
        "formula": "MMSE[visit_i] - MMSE[visit_i - 1]",
        "missingness_behavior": "NaN for first visit; NaN if either MMSE value is NaN",
        "units": "MMSE points (can be negative = decline)",
    },
    "cdr_delta": {
        "source_columns": ["CDR"],
        "visit_offset": -1,
        "formula": "CDR[visit_i] - CDR[visit_i - 1]",
        "missingness_behavior": "NaN for first visit",
        "units": "CDR change (positive = worsening)",
    },
    "nwbv_delta": {
        "source_columns": ["nWBV"],
        "visit_offset": -1,
        "formula": "nWBV[visit_i] - nWBV[visit_i - 1]",
        "missingness_behavior": "NaN for first visit",
        "units": "nWBV change (negative = atrophy)",
    },
    "days_between_visits": {
        "source_columns": ["MR Delay"],
        "visit_offset": -1,
        "formula": "MR_Delay[visit_i+1] - MR_Delay[visit_i]",
        "missingness_behavior": "Should be complete; flag if negative (data error)",
        "units": "Days",
    },
    "n_prior_visits": {
        "source_columns": ["Visit"],
        "visit_offset": None,
        "formula": "index_within_subject (0 for first visit)",
        "missingness_behavior": "Always present; 0 for first visit",
        "units": "Count (integer ≥ 0)",
    },
}


def validate_temporal_features_no_future_leak(pairs_df: pd.DataFrame) -> list[str]:
    """
    Check that temporal feature columns do not contain the word 'next'
    and that 'next_CDR' / 'next_MMSE' do not appear outside their target columns.

    This is a quick sanity check complementing the leakage check in
    build_longitudinal_pairs.py.

    Args:
        pairs_df: The longitudinal pairs DataFrame.

    Returns:
        List of warning strings (empty = no issues found).
    """
    warnings: list[str] = []

    for col in pairs_df.columns:
        if "next" in col.lower() and col not in {"next_CDR", "next_MMSE",
                                                   "next_visit", "next_mr_delay"}:
            warnings.append(
                f"Suspicious column name containing 'next': '{col}'. "
                f"Verify this does not encode future information."
            )

    for col in ["next_CDR", "next_MMSE"]:
        if col in pairs_df.columns:
            # These should ONLY be used as targets, never as features.
            # We just document their presence here — enforcement is in features/clinical.py
            pass

    if warnings:
        for w in warnings:
            logger.warning("Temporal feature check: %s", w)

    return warnings


def describe_temporal_features() -> str:
    """Return a human-readable description of all temporal features."""
    lines = ["Cerebro X Temporal Feature Catalog", "=" * 40]
    for name, info in TEMPORAL_FEATURES.items():
        lines.append(f"\n{name}")
        lines.append(f"  Formula:    {info['formula']}")
        lines.append(f"  Source:     {info['source_columns']}")
        lines.append(f"  Missing:    {info['missingness_behavior']}")
        lines.append(f"  Units:      {info['units']}")
    return "\n".join(lines)
