"""
MRI Preprocessing Pipeline.

Applies spatial and intensity normalization to raw 3D MRI tensors.
Required to standardize brains of different sizes and scanner intensities
before feeding into the 3D CNN encoder.
"""
from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


class MRIPreprocessingPipeline:
    """
    Standardizes raw MRI tensors for the Deep Learning models.
    
    Operations performed in order:
    1. Spatial resize (interpolation) to fixed grid (e.g., 64x64x64 or 112x112x112)
    2. Intensity clipping (remove extreme outliers / artifacts)
    3. Z-score intensity normalization (mean=0, std=1 over brain tissue voxels)
    """
    
    def __init__(
        self,
        target_shape: tuple[int, int, int] = (64, 64, 64),
        clip_percentiles: tuple[float, float] = (0.5, 99.5),
        background_threshold: float = 1e-3,
    ):
        """
        Args:
            target_shape: (D, H, W) to resize all inputs to.
            clip_percentiles: Bottom/top percentiles for intensity clipping.
            background_threshold: Intensity below this is considered background
                                  and ignored during z-score calculation.
        """
        self.target_shape = target_shape
        self.clip_percentiles = clip_percentiles
        self.background_threshold = background_threshold
        
    def _resize_volume(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize 3D volume using trilinear interpolation.
        
        Args:
            tensor: Shape (D, H, W)
            
        Returns:
            Resized tensor of shape self.target_shape
        """
        # F.interpolate expects (batch, channels, D, H, W)
        # So we add batch and channel dims: (1, 1, D, H, W)
        t_5d = tensor.unsqueeze(0).unsqueeze(0)
        
        # Align_corners=False is standard for image interpolation
        resized = F.interpolate(
            t_5d, 
            size=self.target_shape, 
            mode='trilinear', 
            align_corners=False
        )
        
        # Remove batch and channel: (D, H, W)
        return resized.squeeze(0).squeeze(0)
        
    def _normalize_intensity(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalize the tensor, ignoring background.
        
        Background voxels (close to 0) are kept at 0.
        Brain tissue voxels are normalized to mean=0, std=1.
        """
        # Create a mask of brain tissue vs background
        mask = tensor > self.background_threshold
        
        if not mask.any():
            # If the image is completely empty, return as-is to avoid NaNs
            logger.warning("Tensor is entirely background; skipping intensity normalization.")
            return tensor
            
        # Get intensities of only the brain tissue
        brain_voxels = tensor[mask]
        
        # 1. Clip extreme outliers (e.g. bright blood vessels, scanner artifacts)
        # We approximate percentiles using quantile
        q_low = torch.quantile(brain_voxels, self.clip_percentiles[0] / 100.0)
        q_high = torch.quantile(brain_voxels, self.clip_percentiles[1] / 100.0)
        
        tensor = torch.clamp(tensor, min=float(q_low), max=float(q_high))
        
        # Re-extract after clipping for accurate mean/std
        brain_voxels = tensor[mask]
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        
        if std < 1e-6:
            # If std is 0 (constant image), just subtract mean
            logger.warning("Tensor has zero variance in brain tissue.")
            normalized = tensor - mean
        else:
            # Z-score normalization
            normalized = (tensor - mean) / std
            
        # Reset background to exactly a constant (e.g. min value or 0)
        # Since we z-scored, the mean is 0. Background should ideally be heavily negative
        # or just strictly 0. We'll set background to a small negative value based on min.
        
        # Typically in medical imaging after z-score, background is set to the minimum 
        # normalized value to maintain it as the darkest region.
        bg_val = normalized[mask].min() - 0.1
        normalized[~mask] = bg_val
        
        return normalized

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Run the full preprocessing pipeline.
        
        Args:
            tensor: Raw 3D MRI tensor (D, H, W).
            
        Returns:
            Preprocessed tensor (D, H, W) ready for the CNN.
        """
        if tensor is None:
            raise ValueError("Cannot preprocess a None tensor.")
            
        if len(tensor.shape) != 3:
            raise ValueError(f"Expected 3D tensor, got shape {tensor.shape}")
            
        resized = self._resize_volume(tensor)
        normalized = self._normalize_intensity(resized)
        
        return normalized
