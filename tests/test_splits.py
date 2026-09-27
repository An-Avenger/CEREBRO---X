"""Tests for subject-level splitting.

CRITICAL TEST:
  - test_no_subject_in_multiple_splits
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs
from cerebro_x.data.schemas import PairColumns
from cerebro_x.evaluation.splits import subject_level_split, _verify_no_subject_overlap
from tests.conftest import make_synthetic_oasis2


def build_test_pairs(n_subjects=30, seed=42):
    """Helper to build synthetic pairs for split testing."""
    df = make_synthetic_oasis2(n_subjects=n_subjects, seed=seed)
    pairs_df, _ = build_next_visit_pairs(df)
    return pairs_df


class TestSubjectLevelSplit:
    """Tests for the subject-level train/val/test split."""

    def test_split_runs_without_error(self):
        """split function runs on synthetic pairs data."""
        pairs_df = build_test_pairs()
        train_df, val_df, test_df, report = subject_level_split(
            pairs_df, test_size=0.15, val_size=0.15, seed=42
        )
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

    def test_no_subject_in_multiple_splits(self):
        """
        CRITICAL: No Subject ID may appear in more than one split.

        This is Rule R-012 from Rules.md. Violation = data leakage.
        """
        pairs_df = build_test_pairs(n_subjects=40)
        train_df, val_df, test_df, report = subject_level_split(pairs_df, seed=42)

        train_subjects = set(train_df[PairColumns.SUBJECT_ID].unique())
        val_subjects = set(val_df[PairColumns.SUBJECT_ID].unique())
        test_subjects = set(test_df[PairColumns.SUBJECT_ID].unique())

        # Check each pair of splits
        train_val_overlap = train_subjects & val_subjects
        train_test_overlap = train_subjects & test_subjects
        val_test_overlap = val_subjects & test_subjects

        assert len(train_val_overlap) == 0, (
            f"CRITICAL LEAKAGE: {len(train_val_overlap)} subject(s) appear in "
            f"both train and val: {train_val_overlap}"
        )
        assert len(train_test_overlap) == 0, (
            f"CRITICAL LEAKAGE: {len(train_test_overlap)} subject(s) appear in "
            f"both train and test: {train_test_overlap}"
        )
        assert len(val_test_overlap) == 0, (
            f"CRITICAL LEAKAGE: {len(val_test_overlap)} subject(s) appear in "
            f"both val and test: {val_test_overlap}"
        )

    def test_all_pairs_accounted_for(self):
        """Total pairs in train + val + test must equal total pairs in input."""
        pairs_df = build_test_pairs()
        train_df, val_df, test_df, _ = subject_level_split(pairs_df, seed=42)
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(pairs_df), (
            f"Pair count mismatch: {len(train_df)} + {len(val_df)} + {len(test_df)} "
            f"= {total} ≠ {len(pairs_df)}"
        )

    def test_all_subjects_accounted_for(self):
        """All subjects must appear in exactly one of train/val/test."""
        pairs_df = build_test_pairs()
        all_subjects = set(pairs_df[PairColumns.SUBJECT_ID].unique())

        train_df, val_df, test_df, report = subject_level_split(pairs_df, seed=42)

        split_subjects = (
            set(train_df[PairColumns.SUBJECT_ID].unique())
            | set(val_df[PairColumns.SUBJECT_ID].unique())
            | set(test_df[PairColumns.SUBJECT_ID].unique())
        )
        assert split_subjects == all_subjects, (
            f"Some subjects are missing from splits: {all_subjects - split_subjects}"
        )

    def test_report_counts_match_dataframes(self):
        """SplitReport pair counts must match actual DataFrame lengths."""
        pairs_df = build_test_pairs()
        train_df, val_df, test_df, report = subject_level_split(pairs_df, seed=42)

        assert report.train_pairs == len(train_df)
        assert report.val_pairs == len(val_df)
        assert report.test_pairs == len(test_df)

    def test_report_subject_counts_are_correct(self):
        """SplitReport subject lists must match actual unique subjects per split."""
        pairs_df = build_test_pairs()
        train_df, val_df, test_df, report = subject_level_split(pairs_df, seed=42)

        assert len(report.train_subjects) == train_df[PairColumns.SUBJECT_ID].nunique()
        assert len(report.val_subjects) == val_df[PairColumns.SUBJECT_ID].nunique()
        assert len(report.test_subjects) == test_df[PairColumns.SUBJECT_ID].nunique()

    def test_different_seeds_produce_different_splits(self):
        """Different seeds should produce different split assignments."""
        pairs_df = build_test_pairs(n_subjects=40)
        _, _, test_a, _ = subject_level_split(pairs_df, seed=0)
        _, _, test_b, _ = subject_level_split(pairs_df, seed=99)
        # With enough subjects, different seeds produce different test sets
        test_a_subjects = set(test_a[PairColumns.SUBJECT_ID].unique())
        test_b_subjects = set(test_b[PairColumns.SUBJECT_ID].unique())
        assert test_a_subjects != test_b_subjects


class TestVerifyNoSubjectOverlap:
    """Tests for the internal overlap verification function."""

    def test_no_overlap_passes(self):
        """No exception when splits are disjoint."""
        _verify_no_subject_overlap(
            train_subjects=["A", "B", "C"],
            val_subjects=["D", "E"],
            test_subjects=["F", "G"],
        )  # should not raise

    def test_train_val_overlap_raises(self):
        """ValueError raised when subject appears in train AND val."""
        with pytest.raises(ValueError, match="Train.*Val overlap|Val.*Train overlap|overlap"):
            _verify_no_subject_overlap(
                train_subjects=["A", "B", "C"],
                val_subjects=["C", "D"],  # C is in both
                test_subjects=["E"],
            )

    def test_train_test_overlap_raises(self):
        """ValueError raised when subject appears in train AND test."""
        with pytest.raises(ValueError):
            _verify_no_subject_overlap(
                train_subjects=["A", "B", "X"],
                val_subjects=["C"],
                test_subjects=["X", "D"],  # X is in both
            )
