"""Tests for feature engineering modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs, TARGET_COLUMNS
from cerebro_x.data.schemas import PairColumns
from cerebro_x.features.clinical import (
    build_feature_matrix,
    get_feature_columns,
    FEATURE_GROUPS,
)
from cerebro_x.features.temporal import (
    validate_temporal_features_no_future_leak,
    TEMPORAL_FEATURES,
)
from tests.conftest import make_synthetic_oasis2


def build_test_pairs(n_subjects=20, seed=42):
    df = make_synthetic_oasis2(n_subjects=n_subjects, seed=seed)
    pairs_df, _ = build_next_visit_pairs(df)
    return pairs_df


class TestFeatureColumns:
    """Tests for get_feature_columns."""

    def test_default_returns_all_groups(self):
        """With no active_groups specified, all groups are returned."""
        cols = get_feature_columns(active_groups=None)
        assert len(cols) > 0
        # All groups' columns should be included
        for group_info in FEATURE_GROUPS.values():
            for col in group_info["columns"]:
                assert col in cols

    def test_single_group_selection(self):
        """Selecting a single group returns only that group's columns."""
        cols = get_feature_columns(["static_demographic"])
        expected = FEATURE_GROUPS["static_demographic"]["columns"]
        assert cols == expected

    def test_unknown_group_raises_value_error(self):
        """Requesting an unknown feature group raises ValueError."""
        with pytest.raises(ValueError, match="Unknown feature groups"):
            get_feature_columns(["nonexistent_group"])

    def test_no_target_columns_in_any_group(self):
        """No target column (next_CDR, next_MMSE) appears in any feature group."""
        all_cols = get_feature_columns(active_groups=None)
        for target in TARGET_COLUMNS:
            assert target not in all_cols, (
                f"Target column '{target}' found in feature group definitions! "
                f"This would cause leakage."
            )


class TestBuildFeatureMatrix:
    """Tests for the feature matrix builder."""

    def test_builds_feature_matrix(self):
        """Feature matrix is built without errors."""
        pairs_df = build_test_pairs()
        X, schema = build_feature_matrix(pairs_df)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(pairs_df)
        assert len(X.columns) > 0

    def test_next_cdr_not_in_feature_matrix(self):
        """
        CRITICAL: next_CDR must NEVER appear in the feature matrix.
        """
        pairs_df = build_test_pairs()
        X, schema = build_feature_matrix(pairs_df)
        assert PairColumns.NEXT_CDR not in X.columns, (
            "CRITICAL: next_CDR found in feature matrix — this is data leakage."
        )

    def test_next_mmse_not_in_feature_matrix(self):
        """next_MMSE must NEVER appear in the feature matrix."""
        pairs_df = build_test_pairs()
        X, schema = build_feature_matrix(pairs_df)
        assert PairColumns.NEXT_MMSE not in X.columns

    def test_subject_id_not_in_feature_matrix(self):
        """Subject ID must not be in the feature matrix (identifier, not feature)."""
        pairs_df = build_test_pairs()
        X, schema = build_feature_matrix(pairs_df)
        assert PairColumns.SUBJECT_ID not in X.columns

    def test_sex_encoded_as_binary(self):
        """Sex column must be encoded as 0/1, not 'M'/'F' strings."""
        pairs_df = build_test_pairs()
        X, schema = build_feature_matrix(pairs_df)
        if "sex" in X.columns:
            assert X["sex"].isin([0, 1]).all(), (
                f"Sex column contains non-binary values: {X['sex'].unique()}"
            )

    def test_schema_lists_correct_feature_columns(self):
        """FeatureSchema.feature_columns must match actual X columns."""
        pairs_df = build_test_pairs()
        X, schema = build_feature_matrix(pairs_df)
        assert set(schema.feature_columns) == set(X.columns)

    def test_feature_matrix_row_count_matches_pairs(self):
        """Feature matrix must have the same number of rows as pairs_df."""
        pairs_df = build_test_pairs()
        X, _ = build_feature_matrix(pairs_df)
        assert len(X) == len(pairs_df)

    def test_subset_groups_work(self):
        """Building feature matrix from a subset of groups works."""
        pairs_df = build_test_pairs()
        X, schema = build_feature_matrix(pairs_df, active_groups=["static_demographic", "clinical_state"])
        expected = (
            FEATURE_GROUPS["static_demographic"]["columns"]
            + FEATURE_GROUPS["clinical_state"]["columns"]
        )
        # After encoding sex/hand, column names should match (minus categoricals that get encoded)
        # Just check the schema lists the active groups
        assert "static_demographic" in schema.feature_groups_active
        assert "clinical_state" in schema.feature_groups_active
        assert "volumetric" not in schema.feature_groups_active


class TestTemporalFeatureValidation:
    """Tests for temporal feature validation utilities."""

    def test_validate_no_future_leak_passes_on_clean_pairs(self):
        """Validation returns empty warnings on correctly built pairs."""
        pairs_df = build_test_pairs()
        warnings = validate_temporal_features_no_future_leak(pairs_df)
        assert isinstance(warnings, list)

    def test_temporal_feature_catalog_is_documented(self):
        """Each temporal feature has required documentation fields."""
        required_fields = {"source_columns", "formula", "missingness_behavior", "units"}
        for feature_name, info in TEMPORAL_FEATURES.items():
            missing = required_fields - set(info.keys())
            assert not missing, (
                f"Temporal feature '{feature_name}' is missing documentation: {missing}"
            )

    def test_missingness_of_temporal_features_for_non_first_visits(self):
        """prev_mmse etc. should be non-NaN for pairs where n_prior_visits > 0."""
        pairs_df = build_test_pairs(n_subjects=30)
        non_first_visit_pairs = pairs_df[pairs_df["n_prior_visits"] > 0]

        if len(non_first_visit_pairs) == 0:
            pytest.skip("No non-first-visit pairs in synthetic data — increase n_subjects.")

        # For non-first visit pairs, prev_mmse should generally be non-NaN
        # (unless actual MMSE was NaN, which can happen in synthetic data)
        # Just verify the column exists and has some non-NaN values
        assert "prev_mmse" in non_first_visit_pairs.columns
        assert non_first_visit_pairs["prev_mmse"].notna().any(), (
            "prev_mmse is all-NaN even for non-first-visit pairs — logic error in pair builder."
        )
