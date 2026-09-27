"""EEGNet Architecture for Brain-Computer Interfaces.

STATUS: PLANNED — NOT CURRENTLY TRAINABLE.
No EEG dataset with patient-level alignment to OASIS-2 is available.
This architecture is preserved for future integration.
See DOCS/EEG_LIMITATION.md for details.

DO NOT use this model in active experiments.
DO NOT report metrics from any checkpoint trained with synthetic EEG tensors.

A PyTorch implementation of EEGNet, a compact convolutional neural network
for EEG-based brain-computer interfaces.
Reference: Lawhern et al., J. Neural Eng., 2018.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    """Convolution 2d with MaxNorm constraint."""
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    """EEGNet architecture for extracting temporal-spatial features from EEG."""
    
    def __init__(
        self,
        channels: int = 19,
        samples: int = 2000,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        embed_dim: int = 128
    ):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.channels = channels

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(self.F1)
        )

        # Spatial Convolution
        self.spatial_conv = Conv2dWithConstraint(
            self.F1, self.F1 * self.D, (self.channels, 1), 
            groups=self.F1, bias=False, max_norm=1.0
        )
        
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        # Block 3: Separable Convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(
                self.F1 * self.D, self.F1 * self.D, (1, 16), 
                padding=(0, 16 // 2), groups=self.F1 * self.D, bias=False
            ),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Compute output shape of convolutions to initialize linear layer
        out = self._forward_features(torch.zeros(1, 1, self.channels, self.samples))
        conv_out_dim = out.view(1, -1).size(1)

        # Final projection to fixed embedding size
        self.fc = nn.Linear(conv_out_dim, embed_dim)

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.spatial_conv(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Channels, Time)
        Returns:
            Embedding vector of shape (Batch, embed_dim)
        """
        # EEGNet expects input shape (Batch, 1, Channels, Time)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
