"""Multimodal SHAP Explainability.

Calculates feature attributions for Clinical features inside the TriModal architecture
by keeping MRI and EEG modalities fixed for a specific patient.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import shap


class ClinicalTriModalWrapper(nn.Module):
    """Wraps the TriModalCerebroNet for clinical feature SHAP explanation.
    
    GradientExplainer expects a model that accepts a single input `x` to perturb.
    This wrapper takes `clinical_seq` as `x`, and fixes `mri_vol` and `eeg_sig`
    internally. It also repeats the fixed inputs to match the batch size that 
    SHAP sends during its background gradient estimation.
    """
    def __init__(
        self, 
        model: nn.Module, 
        fixed_mri: torch.Tensor, 
        fixed_eeg: torch.Tensor
    ):
        super().__init__()
        self.model = model
        self.fixed_mri = fixed_mri
        self.fixed_eeg = fixed_eeg

    def forward(self, clinical_seq: torch.Tensor) -> torch.Tensor:
        current_batch_size = clinical_seq.size(0)
        
        # SHAP passes batches of perturbed clinical_seq. 
        # We must duplicate the fixed MRI and EEG to match this batch size.
        m_vol = self.fixed_mri.expand(current_batch_size, -1, -1, -1, -1)
        e_sig = self.fixed_eeg.expand(current_batch_size, -1, -1)
        
        return self.model(clinical_seq, m_vol, e_sig)


def compute_clinical_shap_multimodal(
    model: nn.Module,
    background_clinical: torch.Tensor,
    fixed_background_mri: torch.Tensor,
    fixed_background_eeg: torch.Tensor,
    test_clinical: torch.Tensor,
) -> list[np.ndarray] | np.ndarray:
    """Computes SHAP values for clinical features in the multimodal model.
    
    Args:
        model: Trained TriModalCerebroNet.
        background_clinical: Tensor of shape (num_background, seq_len, features)
        fixed_background_mri: Tensor of shape (1, channels, D, H, W)
        fixed_background_eeg: Tensor of shape (1, channels, time)
        test_clinical: Tensor of shape (num_test, seq_len, features)
        
    Returns:
        SHAP values for the clinical features.
    """
    model.eval()
    
    # We wrap the model, fixing the MRI and EEG for the patient we want to explain.
    # Note: Using the test patient's MRI/EEG as the "fixed" background reference
    # allows SHAP to explain how the *clinical* features specifically interact with
    # this specific patient's brain state.
    wrapper = ClinicalTriModalWrapper(model, fixed_background_mri, fixed_background_eeg)
    
    explainer = shap.GradientExplainer(wrapper, background_clinical)  # type: ignore
    
    shap_values = explainer.shap_values(test_clinical)
    
    return shap_values
