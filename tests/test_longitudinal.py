"""Tests for Phase 5 — Longitudinal Sequence Dataset and Temporal Models.

Tests cover:
    - SequenceDataset output shapes and variable-length handling
    - Collate function padding correctness
    - LastVisitBaseline forward pass shapes
    - TemporalCerebroNet (GRU) forward pass shapes
    - Single-visit edge case handling
    - No subject leakage between train/test sequence datasets
    - Target column not present in features
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import torch

from cerebro_x.data.schemas import PairColumns
from cerebro_x.data.pytorch.sequence_dataset import SequenceDataset, sequence_collate_fn
from cerebro_x.models.deep.temporal import LastVisitBaseline, TemporalCerebroNet


# ---------------------------------------------------------------------------
# Fixtures: synthetic longitudinal pairs (same structure as conftest.py)
# ---------------------------------------------------------------------------

def _make_synthetic_pairs(n_subjects: int = 8, seed: int = 42) -> pd.DataFrame:
    """Create synthetic next_visit_pairs matching the real schema."""
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_subjects):
        sid = f"OAS2_TEST_{i:04d}"
        n_visits = rng.integers(2, 5)  # 2-4 visits → 1-3 pairs
        for v in range(1, n_visits):
            row = {
                PairColumns.SUBJECT_ID: sid,
                PairColumns.CURRENT_VISIT: v,
                PairColumns.NEXT_VISIT: v + 1,
                "current_mr_delay": v * 500,
                "next_mr_delay": (v + 1) * 500,
                "days_between_visits": 500,
                "curr_age": 70 + i + v,
                "sex": rng.choice(["M", "F"]),
                "hand": "R",
                "educ": rng.integers(8, 20),
                "ses": rng.choice([1.0, 2.0, 3.0, 4.0]),
                "curr_mmse": rng.integers(18, 31),
                "curr_cdr": rng.choice([0.0, 0.5, 1.0]),
                "curr_group": rng.choice(["Nondemented", "Demented"]),
                "curr_etiv": 1400 + rng.integers(-200, 200),
                "curr_nwbv": 0.7 + rng.random() * 0.1,
                "curr_asf": 1.1 + rng.random() * 0.3,
                "n_prior_visits": v - 1,
                "prev_mmse": float("nan") if v == 1 else rng.integers(18, 31),
                "prev_cdr": float("nan") if v == 1 else rng.choice([0.0, 0.5, 1.0]),
                "prev_nwbv": float("nan") if v == 1 else 0.7 + rng.random() * 0.1,
                "mmse_delta": float("nan") if v == 1 else rng.integers(-3, 4),
                "cdr_delta": float("nan") if v == 1 else rng.choice([-0.5, 0.0, 0.5]),
                "nwbv_delta": float("nan") if v == 1 else rng.random() * 0.02 - 0.01,
                PairColumns.NEXT_CDR: rng.choice([0.0, 0.5, 1.0, 2.0]),
                PairColumns.NEXT_MMSE: rng.integers(15, 31),
            }
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_pairs():
    return _make_synthetic_pairs(n_subjects=8)


@pytest.fixture
def train_dataset(synthetic_pairs):
    return SequenceDataset(synthetic_pairs, is_train=True)


# ---------------------------------------------------------------------------
# Tests: SequenceDataset
# ---------------------------------------------------------------------------

class TestSequenceDataset:
    """Tests for the SequenceDataset."""

    def test_dataset_not_empty(self, train_dataset):
        assert len(train_dataset) > 0

    def test_output_tuple_structure(self, train_dataset):
        """Each item should return (features, target, length)."""
        features, target, length = train_dataset[0]
        assert isinstance(features, torch.Tensor)
        assert isinstance(target, int)
        assert isinstance(length, int)

    def test_feature_shape(self, train_dataset):
        """Features should be (seq_len, num_features)."""
        features, _, length = train_dataset[0]
        assert features.ndim == 2
        assert features.shape[0] == length
        assert features.shape[1] == train_dataset.num_features

    def test_variable_lengths(self, train_dataset):
        """Dataset should contain samples of different sequence lengths."""
        lengths = [train_dataset[i][2] for i in range(len(train_dataset))]
        assert len(set(lengths)) > 1, "Expected variable-length sequences"

    def test_target_valid_class(self, train_dataset):
        """All targets should be in {0, 1, 2, 3}."""
        for i in range(len(train_dataset)):
            _, target, _ = train_dataset[i]
            assert target in {0, 1, 2, 3}

    def test_subject_ids_tracked(self, train_dataset):
        """Should be able to retrieve subject IDs for leakage checks."""
        sids = train_dataset.get_subject_ids()
        assert len(sids) > 0
        assert all(isinstance(s, str) for s in sids)

    def test_val_requires_fitted_transforms(self, synthetic_pairs, train_dataset):
        """Creating a val/test dataset without imputer/scaler should raise."""
        with pytest.raises(ValueError, match="imputer and scaler"):
            SequenceDataset(synthetic_pairs, is_train=False)

    def test_val_with_fitted_transforms(self, synthetic_pairs, train_dataset):
        """Creating a val/test dataset with pre-fit transforms should work."""
        val_ds = SequenceDataset(
            synthetic_pairs,
            imputer=train_dataset.imputer,
            scaler=train_dataset.scaler,
            is_train=False,
        )
        assert len(val_ds) > 0


# ---------------------------------------------------------------------------
# Tests: Collate function
# ---------------------------------------------------------------------------

class TestCollateFn:
    """Tests for sequence_collate_fn."""

    def test_padded_batch_shape(self, train_dataset):
        """Batch should be zero-padded to max length."""
        batch = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
        features, targets, lengths = sequence_collate_fn(batch)

        batch_size = len(batch)
        max_len = max(lengths).item()

        assert features.shape == (batch_size, max_len, train_dataset.num_features)
        assert targets.shape == (batch_size,)
        assert lengths.shape == (batch_size,)

    def test_padding_is_zeros(self, train_dataset):
        """Padded positions should be zero."""
        batch = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
        features, _, lengths = sequence_collate_fn(batch)

        for i, length in enumerate(lengths):
            length = length.item()
            max_len = features.shape[1]
            if length < max_len:
                padded_region = features[i, length:, :]
                assert torch.all(padded_region == 0), "Padded region should be zeros"


# ---------------------------------------------------------------------------
# Tests: LastVisitBaseline
# ---------------------------------------------------------------------------

class TestLastVisitBaseline:
    """Tests for the LastVisitBaseline model."""

    def test_forward_shape(self, train_dataset):
        """Output should be (batch, num_classes)."""
        model = LastVisitBaseline(input_dim=train_dataset.num_features)
        batch = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
        features, targets, lengths = sequence_collate_fn(batch)

        model.eval()
        with torch.no_grad():
            logits = model(features, lengths)

        assert logits.shape == (len(batch), 4)

    def test_single_visit_input(self, train_dataset):
        """Model should handle a single-visit sequence."""
        model = LastVisitBaseline(input_dim=train_dataset.num_features)
        # Create a single-visit input
        x = torch.randn(1, 1, train_dataset.num_features)
        lengths = torch.tensor([1])

        model.eval()
        with torch.no_grad():
            logits = model(x, lengths)

        assert logits.shape == (1, 4)


# ---------------------------------------------------------------------------
# Tests: TemporalCerebroNet
# ---------------------------------------------------------------------------

class TestTemporalCerebroNet:
    """Tests for the GRU-based temporal model."""

    def test_forward_shape(self, train_dataset):
        """Output should be (batch, num_classes)."""
        model = TemporalCerebroNet(input_dim=train_dataset.num_features)
        batch = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
        features, targets, lengths = sequence_collate_fn(batch)

        model.eval()
        with torch.no_grad():
            logits = model(features, lengths)

        assert logits.shape == (len(batch), 4)

    def test_single_visit_input(self, train_dataset):
        """GRU should handle a single-timestep sequence."""
        model = TemporalCerebroNet(input_dim=train_dataset.num_features)
        x = torch.randn(1, 1, train_dataset.num_features)
        lengths = torch.tensor([1])

        model.eval()
        with torch.no_grad():
            logits = model(x, lengths)

        assert logits.shape == (1, 4)

    def test_gradients_flow(self, train_dataset):
        """Loss.backward() should produce gradients on GRU parameters."""
        model = TemporalCerebroNet(input_dim=train_dataset.num_features)
        batch = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
        features, targets, lengths = sequence_collate_fn(batch)

        model.train()
        logits = model(features, lengths)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        # Check that at least one GRU parameter has a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.gru.parameters()
        )
        assert has_grad, "GRU parameters should have gradients after backward()"


# ---------------------------------------------------------------------------
# Tests: Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    """Verify no leakage or target contamination."""

    def test_no_subject_leakage(self, synthetic_pairs):
        """Train and test datasets should have no overlapping subjects."""
        from cerebro_x.evaluation.splits import subject_level_split

        train_df, _, test_df, _ = subject_level_split(
            synthetic_pairs, test_size=0.25, val_size=0.0, seed=42
        )

        train_ds = SequenceDataset(train_df, is_train=True)
        test_ds = SequenceDataset(
            test_df,
            imputer=train_ds.imputer,
            scaler=train_ds.scaler,
            is_train=False,
        )

        train_sids = set(train_ds.get_subject_ids())
        test_sids = set(test_ds.get_subject_ids())

        overlap = train_sids & test_sids
        assert len(overlap) == 0, f"Subject leakage detected: {overlap}"

    def test_target_not_in_features(self, train_dataset):
        """next_CDR and next_MMSE must not appear in feature names."""
        forbidden = {"next_CDR", "next_MMSE"}
        overlap = forbidden & set(train_dataset.feature_names)
        assert len(overlap) == 0, f"Target leakage in features: {overlap}"
