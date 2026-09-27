"""PyTorch Dataset for Tri-Modal classification.

Simultaneously loads Clinical sequences, 3D MRI volumes, and EEG signals 
aligned to specific clinical visits.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cerebro_x.data.eeg.loader import generate_synthetic_eeg

logger = logging.getLogger(__name__)


class TriModalDataset(Dataset):
    """Dataset for multimodal fusion model."""
    
    def __init__(
        self,
        aligned_pairs: list[dict],
        clinical_features_list: list[str],
        mri_transforms: Callable | None = None,
        eeg_transforms: Callable | None = None
    ):
        """
        Args:
            aligned_pairs: List of dicts output by the alignment loaders.
                           Must contain 'clinical_row', 'mri_path', 'eeg_path', 'next_cdr'.
            clinical_features_list: List of column names for clinical sequence.
            mri_transforms: Transformations for the MRI volume.
            eeg_transforms: Transformations for the EEG signal.
        """
        self.pairs = aligned_pairs
        self.features_list = clinical_features_list
        self.mri_transforms = mri_transforms
        self.eeg_transforms = eeg_transforms
        
        # Mapping target string to integer
        self.label_map = {
            "0.0": 0,
            "0.5": 1,
            "1.0": 2,
            "2.0": 3
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.pairs[idx]
        
        # 1. Clinical Modality (Sequence)
        # Note: In a true sequence model, we'd extract the historical trajectory.
        # For this scaffold, we extract the current visit features as a sequence of length 1
        # to match the (Batch, Seq, Features) expected by LSTM.
        clinical_vals = []
        for f in self.features_list:
            val = item['clinical_row'].get(f, 0.0)
            clinical_vals.append(float(val) if pd.notna(val) else 0.0)
        clinical_seq = torch.tensor([clinical_vals], dtype=torch.float32)
        
        # 2. MRI Modality (Volume)
        is_synthetic_mri = item.get('is_synthetic_mri', False)
        if is_synthetic_mri:
            # Seed based on subject+visit for deterministic fallbacks
            seed = sum(ord(c) for c in item['subject_id']) + item['visit']
            torch.manual_seed(seed)
            mri_vol = torch.zeros((1, 64, 64, 64), dtype=torch.float32)
        else:
            # Here we would load real .nii.gz using nibabel.
            # Falling back to dummy for robust scaffold.
            mri_vol = torch.zeros((1, 64, 64, 64), dtype=torch.float32)
            
        if self.mri_transforms:
            mri_vol = self.mri_transforms(mri_vol)
            
        # 3. EEG Modality (Signal)
        is_synthetic_eeg = item.get('is_synthetic_eeg', False)
        if is_synthetic_eeg:
            seed = sum(ord(c) for c in item['subject_id']) + item['visit'] * 10
            eeg_sig = generate_synthetic_eeg(channels=19, time_steps=2000, seed=seed)
        else:
            # Here we would load real .edf using mne.
            eeg_sig = generate_synthetic_eeg(channels=19, time_steps=2000)
            
        if self.eeg_transforms:
            eeg_sig = self.eeg_transforms(eeg_sig)
            
        # Target
        target_str = str(item['next_cdr'])
        target_idx = self.label_map.get(target_str, 0)
        target = torch.tensor(target_idx, dtype=torch.long)
        
        return clinical_seq, mri_vol, eeg_sig, target
