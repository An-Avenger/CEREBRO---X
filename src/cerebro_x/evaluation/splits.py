"""Subject-level splitting for Cerebro X.

RULE (R-012): All longitudinal records for a given Subject ID must be
in exactly one split (train, validation, or test). Never split rows
from the same subject across splits.

This module provides:
  - Subject-level holdout split (train+val subjects vs. test subjects)
  - GroupKFold cross-validation on the training subjects
  - Split statistics and class distribution reporting
  - Validation that no subject appears in multiple splits
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from cerebro_x.data.schemas import PairColumns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split metadata
# ---------------------------------------------------------------------------

@dataclass
class SplitReport:
    """Describes a train/val/test split at the subject and pair level."""

    train_subjects: list
    val_subjects: list
    test_subjects: list
    train_pairs: int
    val_pairs: int
    test_pairs: int
    train_cdr_dist: dict
    val_cdr_dist: dict
    test_cdr_dist: dict
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Subject-level holdout split
# ---------------------------------------------------------------------------

def subject_level_split(
    pairs_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    target_col: str = PairColumns.NEXT_CDR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """
    Split the longitudinal pairs DataFrame by Subject ID.

    All pairs from a given subject end up in exactly ONE of: train, val, test.

    Strategy:
      1. Collect unique subjects.
      2. Assign each subject to test (test_size%), val (val_size%),
         or train (remaining).
      3. This is stratified by the subject's MOST COMMON CDR value to
         approximate class balance across splits.

    Args:
        pairs_df:   Longitudinal pairs DataFrame from build_next_visit_pairs().
        test_size:  Fraction of subjects for final test set.
        val_size:   Fraction of subjects for validation set.
        seed:       Random seed.
        target_col: Target column name (for distribution reporting).

    Returns:
        Tuple of (train_df, val_df, test_df, SplitReport).

    Raises:
        ValueError: If any subject appears in multiple splits.
    """
    rng = np.random.default_rng(seed)

    # Get the most common CDR per subject for stratification
    subject_cdr = (
        pairs_df.groupby(PairColumns.SUBJECT_ID)[target_col]
        .agg(lambda s: s.mode()[0] if len(s.mode()) > 0 else s.iloc[0])
        .reset_index()
    )
    subject_cdr.columns = [PairColumns.SUBJECT_ID, "modal_cdr"]

    subjects = subject_cdr[PairColumns.SUBJECT_ID].values
    modal_cdrs = subject_cdr["modal_cdr"].values
    n = len(subjects)

    # Shuffle subjects (with seed)
    idx = rng.permutation(n)
    subjects = subjects[idx]
    modal_cdrs = modal_cdrs[idx]

    n_test = max(1, round(n * test_size))
    n_val = max(1, round(n * val_size))
    n_train = n - n_test - n_val

    if n_train < 1:
        raise ValueError(
            f"Not enough subjects ({n}) to create train/val/test splits with "
            f"test_size={test_size}, val_size={val_size}. "
            f"Reduce split sizes or use a larger dataset."
        )

    test_subjects = subjects[:n_test].tolist()
    val_subjects = subjects[n_test : n_test + n_val].tolist()
    train_subjects = subjects[n_test + n_val :].tolist()

    # Verify no overlap (critical correctness check)
    _verify_no_subject_overlap(train_subjects, val_subjects, test_subjects)

    train_df = pairs_df[pairs_df[PairColumns.SUBJECT_ID].isin(train_subjects)].copy()
    val_df = pairs_df[pairs_df[PairColumns.SUBJECT_ID].isin(val_subjects)].copy()
    test_df = pairs_df[pairs_df[PairColumns.SUBJECT_ID].isin(test_subjects)].copy()

    report = _build_split_report(
        train_subjects, val_subjects, test_subjects,
        train_df, val_df, test_df,
        target_col,
    )

    _log_split_report(report)

    return train_df, val_df, test_df, report


def _verify_no_subject_overlap(
    train_subjects: list,
    val_subjects: list,
    test_subjects: list,
) -> None:
    """
    Raise ValueError if any Subject ID appears in more than one split.

    This is the most critical correctness check in the entire pipeline.
    """
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    issues = []
    if train_val_overlap:
        issues.append(f"Train ∩ Val overlap: {train_val_overlap}")
    if train_test_overlap:
        issues.append(f"Train ∩ Test overlap: {train_test_overlap}")
    if val_test_overlap:
        issues.append(f"Val ∩ Test overlap: {val_test_overlap}")

    if issues:
        raise ValueError(
            "CRITICAL: Subject-level split overlap detected!\n"
            + "\n".join(issues)
            + "\nThis is a data leakage violation. Do not train any model until this is fixed."
        )


def _build_split_report(
    train_subjects, val_subjects, test_subjects,
    train_df, val_df, test_df,
    target_col,
) -> SplitReport:
    def cdr_dist(df):
        d = df[target_col].value_counts().sort_index()
        return {str(float(k)): int(v) for k, v in d.items()}

    warnings = []
    # Warn if any CDR class is absent from test set
    all_classes = set(train_df[target_col].unique())
    test_classes = set(test_df[target_col].unique())
    missing_in_test = all_classes - test_classes
    if missing_in_test:
        warnings.append(
            f"CDR classes in train but ABSENT from test: {missing_in_test}. "
            f"Per-class metrics for these classes will be undefined."
        )

    return SplitReport(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        train_pairs=len(train_df),
        val_pairs=len(val_df),
        test_pairs=len(test_df),
        train_cdr_dist=cdr_dist(train_df),
        val_cdr_dist=cdr_dist(val_df),
        test_cdr_dist=cdr_dist(test_df),
        warnings=warnings,
    )


def _log_split_report(report: SplitReport) -> None:
    logger.info(
        "Split: train=%d subj/%d pairs | val=%d subj/%d pairs | test=%d subj/%d pairs",
        len(report.train_subjects), report.train_pairs,
        len(report.val_subjects), report.val_pairs,
        len(report.test_subjects), report.test_pairs,
    )
    logger.info("Train CDR dist: %s", report.train_cdr_dist)
    logger.info("Val CDR dist:   %s", report.val_cdr_dist)
    logger.info("Test CDR dist:  %s", report.test_cdr_dist)
    for w in report.warnings:
        logger.warning("Split warning: %s", w)


# ---------------------------------------------------------------------------
# Group cross-validation
# ---------------------------------------------------------------------------

def get_cv_folds(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    target_col: str = PairColumns.NEXT_CDR,
    stratified: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create subject-level cross-validation fold indices for the training set.

    Uses GroupKFold (or StratifiedGroupKFold if stratified=True) so that
    all pairs from one subject stay in the same fold.

    Args:
        train_df:   Training pairs DataFrame.
        n_splits:   Number of CV folds.
        seed:       Random seed (used only by StratifiedGroupKFold).
        target_col: Target column for stratification.
        stratified: Use StratifiedGroupKFold if True.

    Returns:
        List of (train_indices, val_indices) tuples into train_df.
    """
    groups = train_df[PairColumns.SUBJECT_ID].values
    y = train_df[target_col].values
    X_dummy = np.zeros((len(train_df), 1))

    if stratified:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X_dummy, y, groups=groups))
    else:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X_dummy, y, groups=groups))

    logger.info(
        "CV: %d folds, stratified=%s. Fold sizes: %s",
        n_splits,
        stratified,
        [len(s[1]) for s in splits],
    )
    return splits
