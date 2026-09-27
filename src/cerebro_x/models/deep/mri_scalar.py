"""MRI Scalar Encoder for Cerebro X — Phase 3.

STATUS: IMPLEMENTED — trained on real OASIS-2 MRI-derived scalar features.

This module provides:
    MRIScalarEncoder  — MLP encoder: [nWBV, eTIV, ASF, nwbv_delta] → embedding
    MRIScalarClassifier — End-to-end classifier using the encoder

IMPORTANT: This branch operates on MRI-DERIVED SCALAR MEASUREMENTS, not raw
3D MRI volumes. The inputs (nWBV, eTIV, ASF) are computed by FreeSurfer from
T1-weighted OASIS-2 MRI scans. This is a legitimate MRI representation given
our available data. It is NOT equivalent to a 3D CNN on raw volumes.

Features:
    nWBV       — Normalized Whole Brain Volume (atrophy marker)
    eTIV       — Estimated Total Intracranial Volume (head size normalization)
    ASF        — Atlas Scaling Factor (MNI normalization factor)
    nwbv_delta — Change in nWBV from previous visit (longitudinal atrophy rate)

These four features capture the core structural MRI signal available in
OASIS-2 for training a dedicated MRI branch.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# Feature names for this branch (order matters — must match dataset)
MRI_SCALAR_FEATURES = ["curr_nwbv", "curr_etiv", "curr_asf", "nwbv_delta"]
MRI_SCALAR_DIM = len(MRI_SCALAR_FEATURES)  # 4


class MRIScalarEncoder(nn.Module):
    """MLP encoder for MRI-derived scalar features.

    Architecture:
        Input (B, 4) — [nWBV, eTIV, ASF, nwbv_delta]
            ↓
        Linear(4, 32) + BatchNorm + ReLU + Dropout
            ↓
        Linear(32, embed_dim) + BatchNorm + ReLU
            ↓
        Output (B, embed_dim) — MRI embedding

    Args:
        input_dim:  Number of MRI scalar features (default: 4).
        embed_dim:  Output embedding dimension (default: 32).
        dropout:    Dropout rate (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int = MRI_SCALAR_DIM,
        embed_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode MRI scalar features into an embedding.

        Args:
            x: Tensor of shape (batch, input_dim) — MRI scalar features.

        Returns:
            Embedding of shape (batch, embed_dim).
        """
        return self.encoder(x)


class MRIScalarClassifier(nn.Module):
    """End-to-end MRI-scalar-only classifier for ablation experiments.

    Architecture:
        MRIScalarEncoder → Classification Head

    Used in Phase 5 ablation to measure MRI-Only performance.
    """

    def __init__(
        self,
        input_dim: int = MRI_SCALAR_DIM,
        embed_dim: int = 32,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = MRIScalarEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: MRI scalar features (batch, input_dim).

        Returns:
            Logits (batch, num_classes).
        """
        embedding = self.encoder(x)
        return self.head(embedding)
