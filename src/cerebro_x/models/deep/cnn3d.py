"""Lightweight 3D Convolutional Neural Network for MRI processing."""
from __future__ import annotations

import torch
import torch.nn as nn


class Lightweight3DCNN(nn.Module):
    """
    A custom, lightweight 3D CNN designed to process (64x64x64) MRI volumes
    and output a flattened feature embedding.
    """
    def __init__(self, in_channels: int = 1, embedding_dim: int = 64):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: (1, 64, 64, 64) -> (8, 32, 32, 32)
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 2: (8, 32, 32, 32) -> (16, 16, 16, 16)
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 3: (16, 16, 16, 16) -> (32, 8, 8, 8)
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 4: (32, 8, 8, 8) -> (64, 4, 4, 4)
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Adaptive Pool to ensure fixed size before flatten: (64, 2, 2, 2)
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        
        # 64 channels * 2 * 2 * 2 = 512
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: MRI volume of shape (B, C, D, H, W)
            
        Returns:
            Embedding vector of shape (B, embedding_dim)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class MRICerebroNet(nn.Module):
    """
    End-to-End MRI Model predicting next CDR directly from raw scans.
    """
    def __init__(self, in_channels: int = 1, embedding_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.cnn = Lightweight3DCNN(in_channels=in_channels, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.cnn(x)
        logits = self.classifier(embeds)
        return logits
