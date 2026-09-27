"""Gradient-weighted Class Activation Mapping (Grad-CAM) scaffolding.

Extracts spatial heatmaps from the 3D CNN to highlight brain regions
most responsible for the prediction.
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM3D:
    """Computes 3D Grad-CAM for a given PyTorch CNN.
    
    Uses forward and backward hooks on a target convolutional layer
    to extract activations and gradients.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Registers hooks to capture intermediate maps and grads."""
        def forward_hook(module: nn.Module, input_tuple: Any, output: torch.Tensor) -> None:
            self.activations = output

        def backward_hook(
            module: nn.Module, 
            grad_input: Any, 
            grad_output: Any
        ) -> Any:
            self.gradients = grad_output[0]
            return None

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self, 
        clinical_seq: torch.Tensor, 
        mri_vol: torch.Tensor, 
        eeg_sig: torch.Tensor,
        target_class: int | None = None
    ) -> np.ndarray:
        """Generates the Grad-CAM heatmap for the 3D MRI input within a multimodal model.
        
        Args:
            clinical_seq: Shape (1, Seq_Len, Features)
            mri_vol: Shape (1, Channels, D, H, W)
            eeg_sig: Shape (1, Channels, Time)
            target_class: Which class to generate the heatmap for.
                          If None, uses the model's top prediction.
                          
        Returns:
            heatmap: 3D numpy array representing the activation map for the MRI.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass through the full TriModal network
        logits = self.model(clinical_seq, mri_vol, eeg_sig)
        
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
            
        # Backward pass on the target class score
        score = logits[0, target_class]
        score.backward()
        
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured. Are you sure the layer was used?")
            
        # Global average pooling on the gradients across D, H, W dimensions
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        
        # Weight the activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU to only keep positive influence
        cam = F.relu(cam)
        
        # Interpolate back to original input shape
        input_shape = mri_vol.shape[2:]
        cam = F.interpolate(cam, size=input_shape, mode='trilinear', align_corners=False)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
            
        return cam
