"""Bimodal Dataset for Clinical + MRI Scalar fusion — Phase 5.

Each sample yields:
    clinical_seq : Tensor (seq_len, num_clinical_features) — visit history
    mri_scalars  : Tensor (num_mri_features,)              — current-visit MRI scalars
    target       : int                                      — next_CDR class index
    length       : int                                      — true sequence length

The MRI scalars come from the SAME visit as the last entry in clinical_seq.
This prevents temporal leakage: we only use information available at the
time of the clinical visit being processed.

MRI features included:
    curr_nwbv   — current Normalized Whole Brain Volume
    curr_etiv   — current Estimated Total Intracranial Volume
    curr_asf    — current Atlas Scaling Factor
    nwbv_delta  — change in nWBV from prior visit (NaN → 0 for first visit)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from cerebro_x.data.schemas import PairColumns
from cerebro_x.features.clinical import build_feature_matrix
from cerebro_x.models.deep.mri_scalar import MRI_SCALAR_FEATURES

logger = logging.getLogger(__name__)

CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}


class BimodalSequenceDataset(Dataset):
    """PyTorch Dataset yielding (clinical_seq, mri_scalars, target, length).

    Clinical sequence: variable-length visit history up to current visit.
    MRI scalars: fixed vector from the current visit only.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        clinical_imputer: SimpleImputer | None = None,
        clinical_scaler: StandardScaler | None = None,
        mri_imputer: SimpleImputer | None = None,
        mri_scaler: StandardScaler | None = None,
        is_train: bool = True,
    ):
        self.df = df.copy()

        # ── Clinical features (full feature set for sequence) ─────────────────
        X_clin_df, _ = build_feature_matrix(df)
        self.clinical_feature_names = X_clin_df.columns.tolist()
        X_clin_raw = X_clin_df.values

        if is_train:
            self.clinical_imputer = SimpleImputer(strategy="median") if clinical_imputer is None else clinical_imputer
            X_clin_imp = self.clinical_imputer.fit_transform(X_clin_raw)
            self.clinical_scaler = StandardScaler() if clinical_scaler is None else clinical_scaler
            X_clin_scaled = self.clinical_scaler.fit_transform(X_clin_imp)
        else:
            if clinical_imputer is None or clinical_scaler is None:
                raise ValueError("Must provide pre-fit clinical imputer and scaler for val/test.")
            self.clinical_imputer = clinical_imputer
            self.clinical_scaler = clinical_scaler
            X_clin_imp = self.clinical_imputer.transform(X_clin_raw)
            X_clin_scaled = self.clinical_scaler.transform(X_clin_imp)

        X_clin_scaled = np.asarray(X_clin_scaled)

        # ── MRI scalar features ───────────────────────────────────────────────
        # Verify all MRI features are present
        missing_mri = [f for f in MRI_SCALAR_FEATURES if f not in df.columns]
        if missing_mri:
            raise KeyError(
                f"MRI scalar features missing from DataFrame: {missing_mri}\n"
                f"These columns must be present: {MRI_SCALAR_FEATURES}"
            )

        X_mri_raw = df[MRI_SCALAR_FEATURES].values.astype(float)

        if is_train:
            self.mri_imputer = SimpleImputer(strategy="median") if mri_imputer is None else mri_imputer
            X_mri_imp = self.mri_imputer.fit_transform(X_mri_raw)
            self.mri_scaler = StandardScaler() if mri_scaler is None else mri_scaler
            X_mri_scaled = self.mri_scaler.fit_transform(X_mri_imp)
        else:
            if mri_imputer is None or mri_scaler is None:
                raise ValueError("Must provide pre-fit MRI imputer and scaler for val/test.")
            self.mri_imputer = mri_imputer
            self.mri_scaler = mri_scaler
            X_mri_imp = self.mri_imputer.transform(X_mri_raw)
            X_mri_scaled = self.mri_scaler.transform(X_mri_imp)

        X_mri_scaled = np.asarray(X_mri_scaled)

        # ── Build per-row records ─────────────────────────────────────────────
        targets = df[PairColumns.NEXT_CDR].values
        records = []
        for i in range(len(df)):
            records.append({
                "subject_id": df.iloc[i][PairColumns.SUBJECT_ID],
                "visit": int(df.iloc[i][PairColumns.CURRENT_VISIT]),
                "clinical_features": X_clin_scaled[i],
                "mri_scalars": X_mri_scaled[i],
                "target": CDR_MAP[float(targets[i])],
            })

        # ── Group by patient, sort by visit, build sequences ──────────────────
        from collections import defaultdict
        patient_visits: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            patient_visits[rec["subject_id"]].append(rec)
        for sid in patient_visits:
            patient_visits[sid].sort(key=lambda r: r["visit"])

        self.samples: list[dict] = []
        for sid, visit_list in patient_visits.items():
            for end_idx in range(len(visit_list)):
                seq_features = np.stack(
                    [visit_list[j]["clinical_features"] for j in range(end_idx + 1)],
                    axis=0,
                )
                # MRI scalars from the LAST visit in the sequence (current visit)
                mri_scalars = visit_list[end_idx]["mri_scalars"]
                target = visit_list[end_idx]["target"]

                self.samples.append({
                    "subject_id": sid,
                    "clinical_seq": torch.tensor(seq_features, dtype=torch.float32),
                    "mri_scalars": torch.tensor(mri_scalars, dtype=torch.float32),
                    "target": target,
                    "length": end_idx + 1,
                })

        self.num_clinical_features = X_clin_scaled.shape[1]
        self.num_mri_features = X_mri_scaled.shape[1]

        logger.info(
            "BimodalSequenceDataset: %d samples from %d patients | "
            "clinical_features=%d | mri_features=%d",
            len(self.samples), len(patient_visits),
            self.num_clinical_features, self.num_mri_features,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s["clinical_seq"], s["mri_scalars"], s["target"], s["length"]

    def get_subject_ids(self) -> list[str]:
        return list({s["subject_id"] for s in self.samples})


def bimodal_collate_fn(batch):
    """Collate variable-length clinical sequences + fixed MRI scalars.

    Returns:
        clinical_seqs : Tensor (B, max_seq_len, num_clinical_features) — padded
        mri_scalars   : Tensor (B, num_mri_features)
        targets       : Tensor (B,)
        lengths       : Tensor (B,)
    """
    clin_list, mri_list, tgt_list, len_list = zip(*batch)

    max_len = max(len_list)
    num_clin = clin_list[0].shape[-1]

    padded_clin = torch.zeros(len(batch), max_len, num_clin, dtype=torch.float32)
    for i, (feat, _, length) in enumerate(zip(clin_list, mri_list, len_list)):
        padded_clin[i, :length, :] = feat

    mri_scalars = torch.stack(mri_list, dim=0)
    targets = torch.tensor(tgt_list, dtype=torch.long)
    lengths = torch.tensor(len_list, dtype=torch.long)

    return padded_clin, mri_scalars, targets, lengths
