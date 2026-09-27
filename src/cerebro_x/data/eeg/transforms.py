"""EEG Pre-processing Transforms.

Provides lightweight digital filtering and normalization for EEG tensors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bandpass_filter(tensor: torch.Tensor, lowcut: float = 1.0, highcut: float = 40.0, fs: float = 250.0) -> torch.Tensor:
    """Apply a simple approximation of a bandpass filter using PyTorch.
    
    Note: In a production environment, you should use `scipy.signal` or `mne` 
    to apply standard Butterworth filters. This is a lightweight PyTorch implementation 
    for end-to-end differentiable fallback/testing.
    
    Args:
        tensor: Shape (Channels, Time_Steps)
        lowcut: Lower cutoff frequency.
        highcut: Upper cutoff frequency.
        fs: Sampling frequency in Hz.
        
    Returns:
        Filtered tensor.
    """
    # For this scaffold, we simply return the tensor. 
    # Proper FIR/IIR filtering requires complex state management in pure PyTorch
    # which is better deferred to MNE-Python during dataset loading.
    return tensor


def z_score_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Apply z-score normalization across the time dimension for each channel.
    
    Args:
        tensor: Shape (Channels, Time_Steps)
        
    Returns:
        Normalized tensor.
    """
    # tensor shape is (C, T)
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    
    # Avoid division by zero
    std = torch.where(std == 0, torch.tensor(1e-8, device=tensor.device), std)
    
    return (tensor - mean) / std


class ComposeEEG:
    """Composes several EEG transforms together."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            tensor = t(tensor)
        return tensor

def get_default_eeg_transforms() -> ComposeEEG:
    """Returns the standard pre-processing pipeline for EEG."""
    return ComposeEEG([
        z_score_normalize
    ])
