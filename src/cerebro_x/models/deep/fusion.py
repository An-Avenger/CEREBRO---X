"""Multimodal Fusion Architecture for Cerebro-X.

STATUS: SCAFFOLDED — NOT FOR ACTIVE USE IN EXPERIMENTS.

This file contains the planned tri-modal fusion architecture (Clinical + MRI + EEG).
It has NEVER been trained on real data:
  - MRI encoder: requires raw 3D NIfTI volumes (not available in OASIS-2 Kaggle CSV)
  - EEG encoder: requires EEG recordings (no aligned dataset exists)
  - The checkpoint artifacts/EXP-MULTIMODAL-001/multimodal_cpu.pt was trained
    with is_synthetic_mri=True and is_synthetic_eeg=True — it is scientifically invalid.

For active bimodal fusion (Clinical + MRI scalars), see:
    src/cerebro_x/models/deep/bimodal_fusion.py
    artifacts/EXP-FUSION-BIMODAL-001/

For EEG status, see: DOCS/EEG_LIMITATION.md
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cerebro_x.models.deep.temporal import LSTMCerebroNet
from cerebro_x.models.deep.cnn3d import MRICerebroNet
from cerebro_x.models.deep.eeg_net import EEGNet


class TriModalCerebroNet(nn.Module):
    """Fuses Clinical, MRI, and EEG representations."""
    
    def __init__(
        self,
        clinical_features: int,
        clinical_embed_dim: int = 64,
        mri_embed_dim: int = 128,
        eeg_embed_dim: int = 128,
        num_classes: int = 4,
        dropout_rate: float = 0.5
    ):
        super(TriModalCerebroNet, self).__init__()
        
        # Modality Encoders
        self.clinical_encoder = LSTMCerebroNet(
            input_dim=clinical_features, 
            lstm_hidden_dim=clinical_embed_dim,
            num_classes=clinical_embed_dim  # Output embedding size
        )
        
        self.mri_encoder = MRICerebroNet(
            num_classes=mri_embed_dim  # Output embedding size
        )
        
        self.eeg_encoder = EEGNet(
            embed_dim=eeg_embed_dim
        )
        
        # We replace the final classification heads of the encoders with identity
        # so they return raw embeddings instead of logits.
        self.clinical_encoder.fc = nn.Identity()
        self.mri_encoder.fc = nn.Identity()
        # EEGNet already returns the embedding
        
        # Fusion Head
        total_embed_dim = clinical_embed_dim + mri_embed_dim + eeg_embed_dim
        
        self.fusion_head = nn.Sequential(
            nn.Linear(total_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(
        self, 
        clinical_seq: torch.Tensor, 
        mri_vol: torch.Tensor, 
        eeg_sig: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for multimodal fusion.
        
        Args:
            clinical_seq: Shape (Batch, Seq_Len, Features)
            mri_vol: Shape (Batch, 1, D, H, W)
            eeg_sig: Shape (Batch, Channels, Time)
            
        Returns:
            Logits of shape (Batch, num_classes)
        """
        # Extract Unimodal Embeddings
        # LSTMCerebroNet returns (logits, attn_weights) usually, but we set fc=Identity
        # Wait, LSTMCerebroNet forward is:
        # out, attn = self.attention(out)
        # return self.fc(out)
        # So setting fc to Identity makes it return the attended hidden state!
        lengths = torch.ones(clinical_seq.size(0), dtype=torch.long, device=clinical_seq.device)
        v_c = self.clinical_encoder(clinical_seq, lengths)
        
        v_m = self.mri_encoder(mri_vol)
        
        v_e = self.eeg_encoder(eeg_sig)
        
        # Concatenate into the Digital Brain Twin representation Z_t
        z_t = torch.cat([v_c, v_m, v_e], dim=1)
        
        # Predict
        logits = self.fusion_head(z_t)
        
        return logits
