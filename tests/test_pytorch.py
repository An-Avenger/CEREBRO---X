"""Tests for PyTorch infrastructure."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
import torch

from cerebro_x.data.schemas import PairColumns
from cerebro_x.data.pytorch.datasets import LongitudinalClinicalDataset
from cerebro_x.models.deep.mlp import ClinicalMLP


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
        "prev_mmse": [np.nan, 28.0, np.nan],
        "prev_cdr": [np.nan, 0.0, np.nan],
        "prev_nwbv": [np.nan, 0.8, np.nan],
        "mmse_delta": [np.nan, -2.0, np.nan],
        "cdr_delta": [np.nan, 0.5, np.nan],
        "nwbv_delta": [np.nan, -0.05, np.nan],
        PairColumns.NEXT_CDR: [0.5, 1.0, 0.0],
        PairColumns.NEXT_MMSE: [26.0, 20.0, 30.0],
    })


def test_longitudinal_dataset_initialization(dummy_pairs_df):
    """Test that Dataset correctly processes dataframe to tensors."""
    dataset = LongitudinalClinicalDataset(dummy_pairs_df, is_train=True)
    
    assert len(dataset) == 3
    
    X, y = dataset[0]
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.dtype == torch.float32
    assert y.dtype == torch.long
    
    # Check CDR mapping: 0.5 -> 1
    assert y.item() == 1


def test_mlp_forward_pass():
    """Test that MLP accepts tabular tensors and outputs logits."""
    batch_size = 8
    input_dim = 19  # Typical number of features
    num_classes = 4
    
    model = ClinicalMLP(input_dim=input_dim, num_classes=num_classes)
    
    # Create dummy batch
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    logits = model(x)
    
    assert logits.shape == (batch_size, num_classes)
