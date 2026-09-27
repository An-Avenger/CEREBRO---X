"""Basic Multi-Layer Perceptron for tabular clinical data."""
from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalMLP(nn.Module):
    """
    A feed-forward neural network for predicting next-visit CDR
    from tabular clinical features.
    
    Includes Batch Normalization and Dropout to combat overfitting
    on small tabular datasets.
    """
    
    def __init__(self, input_dim: int, num_classes: int = 4, hidden_dims: list[int] = [64, 32]):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            current_dim = h_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)
