"""
MRI Quality Control (QC) module.

Performs automated QC checks on loaded MRI tensors to identify
artifacts, corruption, or unusable scans before they enter the model.
"""
from __future__ import annotations

import logging
from typing import TypedDict
import torch

logger = logging.getLogger(__name__)


class QCResult(TypedDict):
    passed: bool
    reason: str
    metrics: dict[str, float]


class MRIQualityChecker:
    """
    Automated QC pipeline for MRI tensors.
    
    Evaluates basic physical properties of the loaded volume to catch
    gross errors (blank scans, massive noise, extreme clipping).
    """
    
    def __init__(
        self,
        min_non_zero_ratio: float = 0.05,  # At least 5% of voxels must have signal (brain tissue)
        max_non_zero_ratio: float = 0.95,  # Too much signal implies no background (cropping error/noise)
        min_dynamic_range: float = 10.0,   # Max - Min must be reasonable
    ):
        self.min_non_zero_ratio = min_non_zero_ratio
        self.max_non_zero_ratio = max_non_zero_ratio
        self.min_dynamic_range = min_dynamic_range
        
    def check(self, tensor: torch.Tensor) -> QCResult:
        """
        Run QC checks on an unnormalized MRI tensor.
        
        Args:
            tensor: 3D or 4D PyTorch tensor.
            
        Returns:
            QCResult dictionary indicating pass/fail.
        """
        if tensor is None or tensor.numel() == 0:
            return {"passed": False, "reason": "Empty tensor", "metrics": {}}
            
        # Basic metrics
        v_min = float(tensor.min())
        v_max = float(tensor.max())
        v_mean = float(tensor.mean())
        v_std = float(tensor.std())
        
        non_zero = float((tensor > 0).float().mean())
        dynamic_range = v_max - v_min
        
        metrics = {
            "min": v_min,
            "max": v_max,
            "mean": v_mean,
            "std": v_std,
            "non_zero_ratio": non_zero,
            "dynamic_range": dynamic_range,
        }
        
        # Check non-zero ratio (too little = mostly empty, too much = noise/no background)
        if non_zero < self.min_non_zero_ratio:
            return {
                "passed": False, 
                "reason": f"Signal ratio {non_zero:.2%} below minimum {self.min_non_zero_ratio:.2%}",
                "metrics": metrics
            }
            
        if non_zero > self.max_non_zero_ratio:
            return {
                "passed": False,
                "reason": f"Signal ratio {non_zero:.2%} above maximum {self.max_non_zero_ratio:.2%}",
                "metrics": metrics
            }
            
        # Check dynamic range
        if dynamic_range < self.min_dynamic_range:
            return {
                "passed": False,
                "reason": f"Dynamic range {dynamic_range:.1f} below minimum {self.min_dynamic_range}",
                "metrics": metrics
            }
            
        # Check for NaN/Inf
        if not torch.isfinite(tensor).all():
            return {
                "passed": False,
                "reason": "Tensor contains NaN or Inf values",
                "metrics": metrics
            }
            
        return {"passed": True, "reason": "OK", "metrics": metrics}
