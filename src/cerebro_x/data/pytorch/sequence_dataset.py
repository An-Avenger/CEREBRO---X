"""Longitudinal Sequence Dataset for temporal modelling.

Groups visit pairs by Subject ID and builds variable-length sequences:
    Patient with visits [V1, V2, V3, V4] produces samples:
        [V1]          → predict CDR at V2
        [V1, V2]      → predict CDR at V3
        [V1, V2, V3]  → predict CDR at V4

This maximises training samples and teaches the model to handle any history length.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from cerebro_x.data.schemas import PairColumns
from cerebro_x.features.clinical import build_feature_matrix

logger = logging.getLogger(__name__)

# CDR ordinal → class index mapping (same as LongitudinalClinicalDataset)
CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}


class SequenceDataset(Dataset):
    """PyTorch Dataset yielding per-patient visit sequences.

    Each item is a tuple of:
        features : Tensor (seq_len, num_features)  — ordered visit history
        target   : int                              — next_CDR class index for the last visit
        length   : int                              — true sequence length (before padding)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        imputer: SimpleImputer | None = None,
        scaler: StandardScaler | None = None,
        is_train: bool = True,
    ):
        """Build sequence samples from the longitudinal pairs DataFrame.

        Args:
            df: Longitudinal pairs from build_longitudinal_pairs(). Must contain
                Subject ID, current_visit, and next_CDR columns.
            imputer: Pre-fit imputer (required when is_train=False).
            scaler: Pre-fit scaler (required when is_train=False).
            is_train: If True, fits imputer and scaler. If False, transforms only.
        """
        self.df = df.copy()

        # ── Extract features & targets ────────────────────────────────────────
        X_df, _ = build_feature_matrix(df)
        self.feature_names = X_df.columns.tolist()
        self.num_features = len(self.feature_names)

        # ── Imputation and scaling ────────────────────────────────────────────
        X_raw = X_df.values

        if is_train:
            self.imputer = SimpleImputer(strategy="median") if imputer is None else imputer
            X_imputed = self.imputer.fit_transform(X_raw)
            self.scaler = StandardScaler() if scaler is None else scaler
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            if imputer is None or scaler is None:
                raise ValueError("Must provide pre-fit imputer and scaler for val/test sets.")
            self.imputer = imputer
            self.scaler = scaler
            X_imputed = self.imputer.transform(X_raw)
            X_scaled = self.scaler.transform(X_imputed)

        # Ensure X_scaled is a dense numpy array for indexing (fixes IDE spmatrix error)
        X_scaled = np.asarray(X_scaled)
        
        # ── Attach scaled features back to rows ──────────────────────────────
        targets = df[PairColumns.NEXT_CDR].values

        # Build per-row records: (subject_id, visit_number, feature_vector, target_class)
        records: list[dict] = []
        for i in range(len(df)):
            records.append({
                "subject_id": df.iloc[i][PairColumns.SUBJECT_ID],
                "visit": int(df.iloc[i][PairColumns.CURRENT_VISIT]),
                "features": X_scaled[i],
                "target": CDR_MAP[float(targets[i])],
            })

        # ── Group by patient, sort by visit, build sequence samples ──────────
        # For patient with N pairs, we create N samples of increasing length.
        from collections import defaultdict
        patient_visits: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            patient_visits[rec["subject_id"]].append(rec)

        # Sort each patient's records by visit number
        for sid in patient_visits:
            patient_visits[sid].sort(key=lambda r: r["visit"])

        self.samples: list[dict] = []
        for sid, visit_list in patient_visits.items():
            for end_idx in range(len(visit_list)):
                # Sequence = visits 0..end_idx, target = CDR for the last visit
                seq_features = np.stack(
                    [visit_list[j]["features"] for j in range(end_idx + 1)],
                    axis=0,
                )  # (seq_len, num_features)
                target = visit_list[end_idx]["target"]

                self.samples.append({
                    "subject_id": sid,
                    "features": torch.tensor(seq_features, dtype=torch.float32),
                    "target": target,
                    "length": end_idx + 1,
                })

        logger.info(
            "SequenceDataset: %d samples from %d patients, %d features per visit.",
            len(self.samples), len(patient_visits), self.num_features,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Returns (features, target, length)."""
        s = self.samples[idx]
        return s["features"], s["target"], s["length"]

    def get_subject_ids(self) -> list[str]:
        """Return unique subject IDs in this dataset (for leakage checks)."""
        return list({s["subject_id"] for s in self.samples})


def sequence_collate_fn(
    batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences into a padded batch.

    Args:
        batch: List of (features, target, length) tuples.

    Returns:
        features : Tensor (batch_size, max_seq_len, num_features) — zero-padded
        targets  : Tensor (batch_size,) — class indices
        lengths  : Tensor (batch_size,) — true sequence lengths
    """
    features_list, targets_list, lengths_list = zip(*batch)

    # Pad sequences to max length in this batch
    max_len = max(lengths_list)
    num_features = features_list[0].shape[-1]

    padded = torch.zeros(len(batch), max_len, num_features, dtype=torch.float32)
    for i, (feat, _, length) in enumerate(batch):
        padded[i, :length, :] = feat

    targets = torch.tensor(targets_list, dtype=torch.long)
    lengths = torch.tensor(lengths_list, dtype=torch.long)

    return padded, targets, lengths
