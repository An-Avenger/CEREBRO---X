"""Bimodal Fusion Model — Clinical GRU + MRI Scalar Encoder — Phase 5.

STATUS: IMPLEMENTED — trained on real OASIS-2 clinical + MRI-derived scalar data.

Architecture:
    Clinical Visit History (seq)
        ↓
    ClinicalGRUEncoder (GRU, hidden_dim=64) → clinical_embedding (64-dim)

    MRI Scalars [nWBV, eTIV, ASF, nwbv_delta]
        ↓
    MRIScalarEncoder (MLP) → mri_embedding (32-dim)

    [clinical_embedding ‖ mri_embedding] → Z_t (96-dim brain state vector)
        ↓
    Fusion Head (MLP) → num_classes logits

The 96-dim vector Z_t IS the Digital Brain Twin representation for
the bimodal (Clinical + MRI) configuration.

Used in ablation study:
    - Clinical Only  : ClinicalGRUEncoder → head
    - MRI Only       : MRIScalarEncoder   → head
    - Clinical + MRI : BimodalCerebroNet  → head
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ClinicalGRUEncoder(nn.Module):
    """GRU encoder that returns a fixed-dim embedding (NOT logits).

    This is the clinical branch of BimodalCerebroNet.
    Identical to TemporalCerebroNet but without the final classification head —
    it returns the GRU hidden state for fusion.

    Args:
        input_dim:     Number of clinical features per visit.
        hidden_dim:    GRU hidden state dimension (= clinical embedding dim).
        num_layers:    Number of GRU layers.
        dropout:       Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode visit sequence into a fixed-dim embedding.

        Args:
            x:       Padded sequence (batch, max_seq_len, num_features).
            lengths: True sequence lengths (batch,).

        Returns:
            Embedding (batch, hidden_dim).
        """
        lengths_clamped = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths_clamped, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return h_n[-1]  # (batch, hidden_dim)


class ClinicalOnlyClassifier(nn.Module):
    """Clinical-only ablation model (GRU → head).

    Used in Phase 5 ablation as the clinical-only condition.
    Mirrors TemporalCerebroNet architecture exactly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = ClinicalGRUEncoder(input_dim, hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, clinical_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(clinical_seq, lengths)
        return self.head(emb)

    def get_embedding(self, clinical_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return raw embedding (for Digital Brain Twin extraction)."""
        return self.encoder(clinical_seq, lengths)


class BimodalCerebroNet(nn.Module):
    """Clinical GRU + MRI Scalar fusion model.

    The concatenated embedding [clinical ‖ mri] = Z_t is the
    Digital Brain Twin representation for this bimodal configuration.

    Args:
        clinical_input_dim:  Number of clinical features per visit.
        mri_input_dim:       Number of MRI scalar features (default: 4).
        clinical_hidden_dim: GRU hidden state size (clinical embedding dim).
        mri_embed_dim:       MRI scalar MLP output dim.
        num_classes:         Number of CDR classes.
        dropout:             Dropout rate.
    """

    def __init__(
        self,
        clinical_input_dim: int,
        mri_input_dim: int = 4,
        clinical_hidden_dim: int = 64,
        mri_embed_dim: int = 32,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.clinical_encoder = ClinicalGRUEncoder(
            input_dim=clinical_input_dim,
            hidden_dim=clinical_hidden_dim,
            dropout=dropout,
        )
        self.mri_encoder = nn.Sequential(
            nn.Linear(mri_input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, mri_embed_dim),
            nn.BatchNorm1d(mri_embed_dim),
            nn.ReLU(),
        )

        fused_dim = clinical_hidden_dim + mri_embed_dim  # 64 + 32 = 96

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

        self.clinical_hidden_dim = clinical_hidden_dim
        self.mri_embed_dim = mri_embed_dim
        self.fused_dim = fused_dim

    def get_brain_state(
        self,
        clinical_seq: torch.Tensor,
        mri_scalars: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the Z_t brain state vector without predicting.

        This is the Digital Brain Twin representation:
            Z_t = [clinical_embedding ‖ mri_embedding]

        Returns:
            Z_t of shape (batch, fused_dim).
        """
        v_c = self.clinical_encoder(clinical_seq, lengths)
        v_m = self.mri_encoder(mri_scalars)
        return torch.cat([v_c, v_m], dim=1)

    def forward(
        self,
        clinical_seq: torch.Tensor,
        mri_scalars: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: inputs → logits.

        Args:
            clinical_seq: Padded visit history (batch, max_seq_len, clinical_features).
            mri_scalars:  MRI scalar features (batch, mri_input_dim).
            lengths:      True sequence lengths (batch,).

        Returns:
            Logits (batch, num_classes).
        """
        z_t = self.get_brain_state(clinical_seq, mri_scalars, lengths)
        return self.fusion_head(z_t)
