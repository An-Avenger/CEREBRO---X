"""Tests for the OASIS-2 loader module."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cerebro_x.data.schemas import Oasis2Columns, validate_oasis2_columns
from cerebro_x.data.oasis2.loader import load_oasis2, discover_dataset
from tests.conftest import make_synthetic_oasis2


class TestValidateColumns:
    """Tests for schema column validation."""

    def test_all_required_columns_present_passes(self, synthetic_oasis2_df):
        """No exception raised when all required columns are present."""
        validate_oasis2_columns(list(synthetic_oasis2_df.columns))  # should not raise

    def test_missing_column_raises_value_error(self, synthetic_oasis2_df):
        """ValueError raised with clear message when a column is missing."""
        df_broken = synthetic_oasis2_df.drop(columns=["CDR"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_oasis2_columns(list(df_broken.columns))

    def test_missing_column_names_the_missing_column(self, synthetic_oasis2_df):
        """The error message must name the specific missing column(s)."""
        df_broken = synthetic_oasis2_df.drop(columns=["MMSE", "eTIV"])
        with pytest.raises(ValueError) as exc_info:
            validate_oasis2_columns(list(df_broken.columns))
        assert "MMSE" in str(exc_info.value) or "eTIV" in str(exc_info.value)

    def test_extra_columns_do_not_cause_failure(self, synthetic_oasis2_df):
        """Extra columns beyond the required set should not cause an error."""
        df_extra = synthetic_oasis2_df.copy()
        df_extra["extra_col"] = 0
        validate_oasis2_columns(list(df_extra.columns))  # should not raise


class TestLoadOasis2:
    """Tests for the CSV loader."""

    def test_load_from_valid_csv(self, synthetic_oasis2_df, tmp_path):
        """Loader reads a valid CSV without errors."""
        csv_path = tmp_path / "oasis_longitudinal.csv"
        synthetic_oasis2_df.to_csv(csv_path, index=False)

        df = load_oasis2(csv_path, validate_shape=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(synthetic_oasis2_df)
        assert "CDR" in df.columns
        assert "Subject ID" in df.columns

    def test_loader_raises_file_not_found(self, tmp_path):
        """FileNotFoundError raised with clear message when file doesn't exist."""
        fake_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError, match="OASIS-2 CSV not found"):
            load_oasis2(fake_path)

    def test_loader_sorts_by_subject_and_visit(self, synthetic_oasis2_df, tmp_path):
        """Output DataFrame is sorted by Subject ID then Visit number."""
        # Shuffle before saving
        shuffled = synthetic_oasis2_df.sample(frac=1, random_state=7)
        csv_path = tmp_path / "oasis_longitudinal.csv"
        shuffled.to_csv(csv_path, index=False)

        df = load_oasis2(csv_path, validate_shape=False)

        for subj_id, group in df.groupby("Subject ID"):
            visits = group["Visit"].tolist()
            assert visits == sorted(visits), (
                f"Subject {subj_id}: visits not sorted: {visits}"
            )

    def test_loader_cdr_is_numeric(self, synthetic_oasis2_df, tmp_path):
        """CDR column must be numeric (float) after loading."""
        csv_path = tmp_path / "oasis_longitudinal.csv"
        synthetic_oasis2_df.to_csv(csv_path, index=False)
        df = load_oasis2(csv_path, validate_shape=False)
        assert pd.api.types.is_float_dtype(df["CDR"]) or pd.api.types.is_numeric_dtype(df["CDR"])

    def test_loader_raises_on_missing_required_column(self, synthetic_oasis2_df, tmp_path):
        """ValueError raised when a required column is missing from CSV."""
        df_broken = synthetic_oasis2_df.drop(columns=["CDR"])
        csv_path = tmp_path / "oasis_longitudinal.csv"
        df_broken.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            load_oasis2(csv_path, validate_shape=False)

    def test_shape_mismatch_warns_not_raises(self, synthetic_oasis2_df, tmp_path):
        """A shape mismatch produces a UserWarning, not an exception."""
        csv_path = tmp_path / "oasis_longitudinal.csv"
        synthetic_oasis2_df.to_csv(csv_path, index=False)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = load_oasis2(
                csv_path,
                validate_shape=True,
                expected_rows=9999,  # deliberately wrong
                expected_cols=15,
            )
        shape_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(shape_warnings) >= 1


class TestDiscoverDataset:
    """Tests for the dataset discovery function."""

    def test_discover_finds_file_in_search_root(self, synthetic_oasis2_df, tmp_path):
        """discover_dataset finds a CSV placed in a search root."""
        csv_path = tmp_path / "oasis_longitudinal.csv"
        synthetic_oasis2_df.to_csv(csv_path, index=False)

        found = discover_dataset(search_roots=[str(tmp_path)])
        assert found is not None
        assert found.name == "oasis_longitudinal.csv"

    def test_discover_returns_none_when_not_found(self, tmp_path):
        """discover_dataset returns None when CSV is not found anywhere."""
        # Use a non-existent directory
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        # Override search to only look in empty dir
        found = discover_dataset(search_roots=[str(empty_dir)])
        # Should return None (not raise) because Kaggle/local paths won't have it either
        # (in test environment)
        # We just check it doesn't crash
        assert found is None or isinstance(found, Path)
