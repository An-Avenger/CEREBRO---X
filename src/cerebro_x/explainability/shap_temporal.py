"""Temporal SHAP Explainability.

Calculates feature attributions over time for sequence models like the GRU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import shap


class TemporalModelWrapper(nn.Module):
    """Wraps the temporal model to supply hardcoded lengths for SHAP.
    
    SHAP explainers generally expect models that accept a single Tensor `x`
    so they can compute gradients with respect to it. Our GRU model requires
    `(x, lengths)`. This wrapper binds the lengths so SHAP can perturb `x`.
    """
    def __init__(self, model: nn.Module, lengths: torch.Tensor):
        super().__init__()
        self.model = model
        self.lengths = lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SHAP might pass batches of different sizes during gradient estimation.
        # We clamp or slice the lengths to match the current batch size.
        current_batch_size = x.size(0)
        
        if current_batch_size == self.lengths.size(0):
            batch_lengths = self.lengths
        elif current_batch_size < self.lengths.size(0):
            batch_lengths = self.lengths[:current_batch_size]
        else:
            # If SHAP duplicates samples, we duplicate the lengths
            repeats = (current_batch_size // self.lengths.size(0)) + 1
            batch_lengths = self.lengths.repeat(repeats)[:current_batch_size]
            
        return self.model(x, batch_lengths)


def compute_temporal_shap(
    model: nn.Module,
    background_data: torch.Tensor,
    background_lengths: torch.Tensor,
    test_data: torch.Tensor,
    test_lengths: torch.Tensor,
) -> np.ndarray:
    """Computes SHAP values for a sequence model using GradientExplainer.
    
    Args:
        model: Trained TemporalCerebroNet.
        background_data: Tensor of shape (num_background, seq_len, features)
        background_lengths: Tensor of shape (num_background,)
        test_data: Tensor of shape (num_test, seq_len, features)
        test_lengths: Tensor of shape (num_test,)
        
    Returns:
        SHAP values. Note that the shape returned by GradientExplainer
        might be a list of arrays (one per class) or a single array 
        depending on the SHAP version, but typically (num_classes, num_test, seq_len, features)
        or a list of length num_classes with arrays of (num_test, seq_len, features).
    """
    model.eval()
    
    # We wrap the model specifically for the background data initially
    wrapper = TemporalModelWrapper(model, background_lengths)
    
    # GradientExplainer is generally safer than DeepExplainer for RNNs with packed sequences
    explainer = shap.GradientExplainer(wrapper, background_data)
    
    # Update wrapper for the test data pass
    wrapper.lengths = test_lengths
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(test_data)
    
    return shap_values
