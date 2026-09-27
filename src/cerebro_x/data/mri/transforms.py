"""MRI Pre-processing Transforms.

Provides lightweight spatial and intensity normalization for 3D tensors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

def normalize_intensity(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply Zero-Mean Unit-Variance (Z-score) normalization to a 3D MRI tensor.
    Only considers non-zero voxels (assuming skull-stripped background is 0).
    """
    mask = tensor > 0
    if mask.sum() == 0:
        return tensor  # Empty tensor fallback
        
    mean = tensor[mask].mean()
    std = tensor[mask].std()
    
    if std == 0:
        return tensor
        
    normalized = tensor.clone()
    normalized[mask] = (normalized[mask] - mean) / std
    return normalized

def resize_volume(tensor: torch.Tensor, target_shape: tuple[int, int, int] = (112, 112, 112)) -> torch.Tensor:
    """
    Resize a 3D volume to a fixed target shape using trilinear interpolation.
    
    Args:
        tensor: (D, H, W)
        target_shape: (target_D, target_H, target_W)
        
    Returns:
        (target_D, target_H, target_W)
    """
    # F.interpolate expects (N, C, D, H, W)
    # We have (D, H, W) -> reshape to (1, 1, D, H, W)
    t = tensor.unsqueeze(0).unsqueeze(0)
    
    resized = F.interpolate(
        t, 
        size=target_shape, 
        mode='trilinear', 
        align_corners=False
    )
    
    return resized.squeeze(0).squeeze(0)

class MRITransformPipeline:
    """Standard pipeline for training/inference."""
    
    def __init__(self, target_shape: tuple[int, int, int] = (112, 112, 112)):
        self.target_shape = target_shape
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Expects a 3D tensor (D, H, W).
        Returns a normalized, resized tensor with a channel dim: (1, D, H, W).
        """
        # Ensure float32
        tensor = tensor.to(torch.float32)
        
        # Intensity Norm
        tensor = normalize_intensity(tensor)
        
        # Spatial Norm
        tensor = resize_volume(tensor, self.target_shape)
        
        # Add channel dimension (C, D, H, W) for PyTorch CNNs
        tensor = tensor.unsqueeze(0)
        
        return tensor
