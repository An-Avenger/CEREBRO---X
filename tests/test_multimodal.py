"""Tests for Multi-Modal infrastructure."""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from cerebro_x.data.pytorch.datasets import MultiModalDataset
from cerebro_x.models.deep.cnn3d import Lightweight3DCNN
from cerebro_x.models.deep.bimodal_fusion import BimodalCerebroNet


@pytest.fixture
def dummy_pairs_df():
    """Create a minimal pairs dataframe."""
    return pd.DataFrame({
        "Subject ID": ["OAS2_0001", "OAS2_0001", "OAS2_0002"],
        "current_visit": [1, 2, 1],
        "next_visit": [2, 3, 2],
        "current_mr_delay": [0, 500, 0],
        "next_mr_delay": [500, 1000, 600],
        "days_between_visits": [500, 500, 600],
        "curr_age": [75, 76, 80],
        "sex": ["M", "M", "F"],
        "hand": ["R", "R", "L"],
        "educ": [12, 12, 16],
        "ses": [2.0, 2.0, 1.0],
        "curr_mmse": [28.0, 26.0, 30.0],
        "curr_cdr": [0.0, 0.5, 0.0],
        "curr_group": ["Nondemented", "Demented", "Nondemented"],
        "curr_etiv": [1500, 1500, 1400],
        "curr_nwbv": [0.8, 0.75, 0.82],
        "curr_asf": [1.1, 1.1, 1.2],
        "n_prior_visits": [0, 1, 0],
        "prev_mmse": [float('nan'), 28.0, float('nan')],
        "prev_cdr": [float('nan'), 0.0, float('nan')],
        "prev_nwbv": [float('nan'), 0.8, float('nan')],
        "mmse_delta": [float('nan'), -2.0, float('nan')],
        "cdr_delta": [float('nan'), 0.5, float('nan')],
        "nwbv_delta": [float('nan'), -0.05, float('nan')],
        "next_CDR": [0.5, 1.0, 0.0],
        "next_mmse": [26.0, 20.0, 30.0],
    })


def test_multimodal_dataset_returns_mri(dummy_pairs_df):
    """Test that dataset returns both clinical and MRI tensors."""
    dataset = MultiModalDataset(dummy_pairs_df, is_train=True)

    X_clin, X_mri, y = dataset[0]

    assert isinstance(X_clin, torch.Tensor)
    assert isinstance(X_mri, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # Check dummy MRI shape (1, 64, 64, 64)
    assert X_mri.shape == (1, 64, 64, 64)


def test_cnn3d_forward_pass():
    """Test the 3D CNN shape transformations."""
    batch_size = 2
    model = Lightweight3DCNN(in_channels=1, embedding_dim=64)

    # Create dummy MRI batch
    x_mri = torch.randn(batch_size, 1, 64, 64, 64)
    emb = model(x_mri)

    assert emb.shape == (batch_size, 64)


def test_bimodal_fusion_forward_pass():
    """Test the bimodal fusion network (Clinical GRU + MRI scalars)."""
    batch_size = 2
    clinical_dim = 19    # Standard feature set from OASIS-2
    mri_scalar_dim = 4   # [nWBV, eTIV, ASF, nwbv_delta]
    num_classes = 4

    model = BimodalCerebroNet(
        clinical_input_dim=clinical_dim,
        mri_input_dim=mri_scalar_dim,
        num_classes=num_classes,
    )

    # Padded clinical sequence (batch, seq_len, features)
    x_clin = torch.randn(batch_size, 3, clinical_dim)
    # MRI scalar features (batch, mri_dim)
    x_mri = torch.randn(batch_size, mri_scalar_dim)
    lengths = torch.tensor([3, 2], dtype=torch.long)

    logits = model(x_clin, x_mri, lengths)

    assert logits.shape == (batch_size, num_classes)


def test_bimodal_get_brain_state():
    """Test that get_brain_state returns correct Z_t shape."""
    batch_size = 2
    model = BimodalCerebroNet(clinical_input_dim=19, mri_input_dim=4)

    x_clin = torch.randn(batch_size, 2, 19)
    x_mri = torch.randn(batch_size, 4)
    lengths = torch.tensor([2, 1], dtype=torch.long)

    z_t = model.get_brain_state(x_clin, x_mri, lengths)

    # Z_t = [clinical_embedding (64) | mri_embedding (32)] = 96-dim
    assert z_t.shape == (batch_size, model.fused_dim)



