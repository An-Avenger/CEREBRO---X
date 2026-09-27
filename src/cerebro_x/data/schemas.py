"""Schemas for OASIS-2 dataset records and validated DataFrames."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Column name constants — single source of truth for the verified OASIS-2 schema.
# If the CSV ever changes, update here and tests will catch breakage.
# ---------------------------------------------------------------------------

class Oasis2Columns:
    """Verified column names for oasis_longitudinal.csv (jboysen/mri-and-alzheimers)."""

    SUBJECT_ID = "Subject ID"
    MRI_ID = "MRI ID"
    GROUP = "Group"
    VISIT = "Visit"
    MR_DELAY = "MR Delay"
    SEX = "M/F"
    HAND = "Hand"
    AGE = "Age"
    EDUC = "EDUC"
    SES = "SES"
    MMSE = "MMSE"
    CDR = "CDR"
    ETIV = "eTIV"
    NWBV = "nWBV"
    ASF = "ASF"

    # All required columns in verified order
    REQUIRED: list[str] = [
        SUBJECT_ID, MRI_ID, GROUP, VISIT, MR_DELAY,
        SEX, HAND, AGE, EDUC, SES,
        MMSE, CDR, ETIV, NWBV, ASF,
    ]

    # CDR valid values
    CDR_VALID_VALUES: set[float] = {0.0, 0.5, 1.0, 2.0}

    # Group valid values
    GROUP_VALID_VALUES: set[str] = {"Nondemented", "Demented", "Converted"}


# ---------------------------------------------------------------------------
# Longitudinal pair schema — produced by build_longitudinal_pairs.py
# ---------------------------------------------------------------------------

class PairColumns:
    """Column names in the longitudinal next-visit pairs DataFrame."""

    SUBJECT_ID = "Subject ID"
    CURRENT_VISIT = "current_visit"
    NEXT_VISIT = "next_visit"
    CURRENT_MR_DELAY = "current_mr_delay"
    NEXT_MR_DELAY = "next_mr_delay"

    # Target columns (must NEVER appear in input feature matrix X)
    NEXT_CDR = "next_CDR"
    NEXT_MMSE = "next_MMSE"

    # Prefixes for current and previous visit columns
    CURRENT_PREFIX = "curr_"
    PREV_PREFIX = "prev_"


# ---------------------------------------------------------------------------
# Data validation helpers
# ---------------------------------------------------------------------------

def validate_oasis2_columns(df_columns: list[str]) -> None:
    """
    Raise ValueError if any required OASIS-2 column is missing.

    Args:
        df_columns: List of column names found in the loaded DataFrame.

    Raises:
        ValueError: With a clear message naming the missing columns.
    """
    required = set(Oasis2Columns.REQUIRED)
    found = set(df_columns)
    missing = required - found
    if missing:
        raise ValueError(
            f"OASIS-2 schema validation failed.\n"
            f"Missing columns: {sorted(missing)}\n"
            f"Expected all of: {Oasis2Columns.REQUIRED}\n"
            f"Found: {sorted(found)}\n"
            f"Check that you are loading the correct file: oasis_longitudinal.csv"
        )


def validate_oasis2_shape(df, expected_rows: int = 373, expected_cols: int = 15) -> None:
    """
    Warn (not raise) if the DataFrame shape differs from verified facts.

    We warn rather than raise because the user may have a different version
    of the dataset. Mismatch is reported, not silently ignored.

    Args:
        df: Loaded DataFrame.
        expected_rows: Verified row count (373 for Kaggle jboysen dataset).
        expected_cols: Verified column count (15).
    """
    import warnings
    actual_rows, actual_cols = df.shape
    if actual_rows != expected_rows or actual_cols != expected_cols:
        warnings.warn(
            f"OASIS-2 shape mismatch.\n"
            f"  Expected: ({expected_rows} rows, {expected_cols} cols)\n"
            f"  Found:    ({actual_rows} rows, {actual_cols} cols)\n"
            f"This may indicate a different version or filtered dataset. "
            f"Inspect the audit report before proceeding.",
            UserWarning,
            stacklevel=2,
        )


@dataclass
class VerifiedDatasetFacts:
    """
    Ground-truth facts about the verified OASIS-2 Kaggle CSV.
    Used by audit to compare actual vs. expected.
    """

    total_rows: int = 373
    total_columns: int = 15
    unique_subjects: int = 150
    visit_distribution: dict = field(default_factory=lambda: {2: 94, 3: 43, 4: 9, 5: 4})
    group_distribution: dict = field(
        default_factory=lambda: {"Nondemented": 190, "Demented": 146, "Converted": 37}
    )
    cdr_distribution: dict = field(
        default_factory=lambda: {0.0: 206, 0.5: 123, 1.0: 41, 2.0: 3}
    )
    mmse_count: int = 371
    mmse_mean: float = 27.34
    mmse_std: float = 3.68
    mmse_min: float = 4.0
    mmse_max: float = 30.0
    subjects_with_changing_cdr: int = 34
