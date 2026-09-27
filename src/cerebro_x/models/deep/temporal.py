"""Temporal models for longitudinal Digital Brain Twin prediction.

Contains:
    LastVisitBaseline   — Ignores history; uses only the last visit's features.
    TemporalCerebroNet  — GRU-based model consuming visit sequences.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LastVisitBaseline(nn.Module):
    """Baseline: predict next CDR using only the last visit's features.

    This answers the question: "Does visit history add value over the latest snapshot?"
    If the GRU cannot beat this, sequence modelling is not justified (Rule R-022).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_classes: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Padded sequences of shape (batch, max_seq_len, num_features).
            lengths: True sequence lengths of shape (batch,).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Extract the last valid timestep for each sample
        batch_size = x.size(0)
        last_indices = (lengths - 1).long()  # (batch,)
        last_features = x[torch.arange(batch_size), last_indices]  # (batch, num_features)
        return self.head(last_features)


class TemporalCerebroNet(nn.Module):
    """GRU-based longitudinal model for the Digital Brain Twin.

    Consumes a patient's entire visit history as a sequence and predicts
    the next-visit CDR class from the final hidden state.

    Architecture:
        Input (batch, seq_len, num_features)
            ↓
        GRU (hidden_dim, num_layers, unidirectional)
            ↓
        Final hidden state → (batch, hidden_dim)
            ↓
        MLP Head → (batch, num_classes)

    Design notes:
        - GRU over LSTM: fewer parameters, suitable for short sequences (<5 visits).
        - Unidirectional: causal constraint — cannot see future visits.
        - Packed sequences: efficient handling of variable-length inputs.
    """

    def __init__(
        self,
        input_dim: int,
        gru_hidden_dim: int = 64,
        gru_num_layers: int = 1,
        head_hidden_dim: int = 32,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden_dim, head_hidden_dim),
            nn.BatchNorm1d(head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with packed sequence handling.

        Args:
            x: Padded sequences of shape (batch, max_seq_len, num_features).
            lengths: True sequence lengths of shape (batch,).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Pack padded sequences for efficient GRU processing
        # Clamp lengths to minimum 1 to avoid empty sequences
        lengths_clamped = lengths.clamp(min=1).cpu()

        packed = pack_padded_sequence(
            x, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        # h_n shape: (num_layers, batch, hidden_dim)
        # Take the last layer's hidden state
        final_hidden = h_n[-1]  # (batch, hidden_dim)

        logits = self.head(final_hidden)
        return logits


class LSTMCerebroNet(nn.Module):
    """LSTM-based longitudinal baseline model for the Digital Brain Twin."""

    def __init__(
        self,
        input_dim: int,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 1,
        head_hidden_dim: int = 32,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden_dim),
            nn.BatchNorm1d(head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths_clamped = lengths.clamp(min=1).cpu()

        packed = pack_padded_sequence(
            x, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self.lstm(packed)
        final_hidden = h_n[-1]  # (batch, hidden_dim)

        logits = self.head(final_hidden)
        return logits

