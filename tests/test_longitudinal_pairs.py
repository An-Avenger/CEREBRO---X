"""Tests for longitudinal pair construction.

CRITICAL TESTS:
  - test_next_cdr_not_in_input_features
  - test_pairs_are_consecutive_visits
  - test_last_visit_excluded
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs, TARGET_COLUMNS
from cerebro_x.data.schemas import Oasis2Columns, PairColumns
from tests.conftest import make_synthetic_oasis2


class TestPairConstruction:
    """Tests for the core pair-building logic."""

    def test_builds_pairs_from_synthetic_data(self, synthetic_oasis2_df):
        """Pair construction runs without error on synthetic data."""
        pairs_df, meta = build_next_visit_pairs(synthetic_oasis2_df)
        assert len(pairs_df) > 0
        assert meta.total_pairs == len(pairs_df)

    def test_correct_pair_count(self, synthetic_oasis2_df):
        """
        Pair count must equal sum of (n_visits - 1) across all subjects.
        Each subject with N visits contributes N-1 pairs.
        """
        # Expected: sum over subjects of (visits - 1)
        visit_counts = synthetic_oasis2_df.groupby("Subject ID").size()
        expected_pairs = int((visit_counts - 1).sum())

        pairs_df, meta = build_next_visit_pairs(synthetic_oasis2_df)
        assert meta.total_pairs == expected_pairs, (
            f"Expected {expected_pairs} pairs, got {meta.total_pairs}"
        )

    def test_pairs_are_consecutive_visits(self, synthetic_oasis2_df):
        """
        For each pair row, next_visit must equal current_visit + 1
        within the same subject's sequential visit numbering.
        """
        pairs_df, _ = build_next_visit_pairs(synthetic_oasis2_df)
        # current_visit and next_visit should be consecutive integer visit numbers
        for _, row in pairs_df.iterrows():
            assert row[PairColumns.NEXT_VISIT] > row[PairColumns.CURRENT_VISIT], (
                f"next_visit ({row[PairColumns.NEXT_VISIT]}) must be > "
                f"current_visit ({row[PairColumns.CURRENT_VISIT]}) "
                f"for subject {row[PairColumns.SUBJECT_ID]}"
            )

    def test_last_visit_excluded(self, synthetic_oasis2_df):
        """
        The last visit for each subject must NOT appear as a current_visit
        (because it has no next visit to predict).
        """
        pairs_df, _ = build_next_visit_pairs(synthetic_oasis2_df)

        # Get the last visit number per subject from the original data
        last_visits = (
            synthetic_oasis2_df.groupby("Subject ID")["Visit"].max()
        )

        for subj_id, last_v in last_visits.items():
            subj_pairs = pairs_df[pairs_df[PairColumns.SUBJECT_ID] == subj_id]
            current_visits_used = subj_pairs[PairColumns.CURRENT_VISIT].tolist()
            assert last_v not in current_visits_used, (
                f"Subject {subj_id}: last visit {last_v} incorrectly used "
                f"as current_visit. Last visit has no next visit to predict."
            )

    def test_all_subjects_in_pairs(self, synthetic_oasis2_df):
        """All subjects with ≥2 visits must appear in the pairs DataFrame."""
        visit_counts = synthetic_oasis2_df.groupby("Subject ID").size()
        multi_visit_subjects = set(visit_counts[visit_counts >= 2].index)

        pairs_df, _ = build_next_visit_pairs(synthetic_oasis2_df)
        subjects_in_pairs = set(pairs_df[PairColumns.SUBJECT_ID].unique())

        assert multi_visit_subjects == subjects_in_pairs

    def test_metadata_pair_count_matches_dataframe(self, synthetic_oasis2_df):
        """PairMetadata.total_pairs must match actual len(pairs_df)."""
        pairs_df, meta = build_next_visit_pairs(synthetic_oasis2_df)
        assert meta.total_pairs == len(pairs_df)

    def test_metadata_subjects_match(self, synthetic_oasis2_df):
        """PairMetadata.unique_subjects_with_pairs must match actual count."""
        pairs_df, meta = build_next_visit_pairs(synthetic_oasis2_df)
        actual_subjects = pairs_df[PairColumns.SUBJECT_ID].nunique()
        assert meta.unique_subjects_with_pairs == actual_subjects


class TestNoLeakage:
    """
    CRITICAL: These tests verify that target information never leaks into input features.
    If any of these tests fail, the model must not be trained.
    """

    def test_next_cdr_not_in_input_features(self, synthetic_oasis2_df):
        """
        next_CDR must not appear in the feature matrix that would be passed to a model.

        This is the most critical leakage test. The feature matrix X is defined as
        all columns EXCEPT Subject ID and TARGET_COLUMNS.
        """
        pairs_df, _ = build_next_visit_pairs(synthetic_oasis2_df)

        # Feature columns = all columns except subject ID and targets
        feature_cols = set(pairs_df.columns) - {PairColumns.SUBJECT_ID} - set(TARGET_COLUMNS)

        assert PairColumns.NEXT_CDR not in feature_cols, (
            "CRITICAL LEAKAGE: next_CDR appears in the input feature set! "
            "This would give the model direct access to the target it is predicting."
        )

    def test_next_mmse_not_in_input_features(self, synthetic_oasis2_df):
        """
        next_MMSE must not appear in the input feature set.
        """
        pairs_df, _ = build_next_visit_pairs(synthetic_oasis2_df)
        feature_cols = set(pairs_df.columns) - {PairColumns.SUBJECT_ID} - set(TARGET_COLUMNS)

        assert PairColumns.NEXT_MMSE not in feature_cols, (
            "CRITICAL LEAKAGE: next_MMSE appears in the input feature set!"
        )

    def test_no_future_visit_info_in_current_features(self, synthetic_oasis2_df):
        """
        Current-visit feature columns must only encode information from visit T,
        not from visit T+1 (the target visit).

        This test verifies that curr_* values in the pairs match the original
        DataFrame values for the corresponding subject and visit.
        """
        pairs_df, _ = build_next_visit_pairs(synthetic_oasis2_df)

        # For a sample of pairs, verify curr_mmse matches the original MMSE at current_visit
        sample = pairs_df.sample(min(20, len(pairs_df)), random_state=7)

        for _, pair_row in sample.iterrows():
            subj = pair_row[PairColumns.SUBJECT_ID]
            curr_v = pair_row[PairColumns.CURRENT_VISIT]
            expected_mmse = synthetic_oasis2_df[
                (synthetic_oasis2_df["Subject ID"] == subj) &
                (synthetic_oasis2_df["Visit"] == curr_v)
            ]["MMSE"].values

            if len(expected_mmse) > 0 and not pd.isna(expected_mmse[0]) and not pd.isna(pair_row["curr_mmse"]):
                assert abs(pair_row["curr_mmse"] - expected_mmse[0]) < 1e-6, (
                    f"curr_mmse for subject {subj} visit {curr_v} does not match "
                    f"original: expected {expected_mmse[0]}, got {pair_row['curr_mmse']}"
                )

    def test_temporal_features_are_nan_for_first_visit(self, synthetic_oasis2_df):
        """
        prev_mmse, prev_cdr, mmse_delta etc. must be NaN for first-visit pairs
        (where n_prior_visits == 0).
        """
        pairs_df, _ = build_next_visit_pairs(synthetic_oasis2_df)
        first_visit_pairs = pairs_df[pairs_df["n_prior_visits"] == 0]

        temporal_cols = ["prev_mmse", "prev_cdr", "prev_nwbv", "mmse_delta", "cdr_delta", "nwbv_delta"]
        for col in temporal_cols:
            n_non_nan = first_visit_pairs[col].notna().sum()
            assert n_non_nan == 0, (
                f"Column '{col}' has {n_non_nan} non-NaN values for first-visit pairs. "
                f"First visits have no prior visit, so these should be NaN."
            )
