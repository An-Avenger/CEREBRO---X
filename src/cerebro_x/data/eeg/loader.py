"""EEG Dataloader logic with synthetic fallback for missing files.

This module provides the loading mechanisms for raw EEG files (.edf or .fif)
while implementing a robust synthetic fallback when raw files are unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

logger = logging.getLogger(__name__)


def generate_synthetic_eeg(channels: int = 19, time_steps: int = 2000, seed: int | None = None) -> torch.Tensor:
    """Generate a mathematically valid synthetic EEG tensor for structural testing.
    
    Args:
        channels: Number of EEG channels (e.g., standard 10-20 montage).
        time_steps: Number of time samples.
        seed: Random seed for reproducibility based on patient ID.
        
    Returns:
        Tensor of shape (Channels, Time_Steps).
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    # Generate pink noise or random normal to simulate EEG
    tensor = torch.randn(channels, time_steps)
    return tensor


def align_clinical_to_eeg(pairs_df: pd.DataFrame, eeg_dir: str | Path, fallback: bool = True) -> list[dict]:
    """Align the longitudinal clinical visits with their corresponding EEG files.
    
    Args:
        pairs_df: DataFrame containing the longitudinal clinical pairs.
        eeg_dir: Base directory containing raw EEG files.
        fallback: If True, generate dummy tensors when files are missing.
        
    Returns:
        A list of dictionaries containing the clinical inputs, target, and EEG tensor path/data.
    """
    eeg_dir = Path(eeg_dir)
    aligned_data = []
    missing_count = 0
    
    for _, row in pairs_df.iterrows():
        subject_id = row['Subject']
        visit_id = int(row['Visit'])
        next_cdr = row['next_CDR']
        
        # Clinical feature extraction logic happens in the Dataset, just pass the row
        
        # Determine EEG filename (e.g. OAS2_0001_EEG1.edf)
        eeg_id = f"{subject_id}_EEG{visit_id}"
        eeg_path = eeg_dir / f"{eeg_id}.edf"
        
        item = {
            'subject_id': subject_id,
            'visit': visit_id,
            'clinical_row': row.to_dict(),
            'next_cdr': next_cdr,
            'eeg_path': eeg_path,
            'eeg_id': eeg_id,
            'is_synthetic': False
        }
        
        if not eeg_path.exists():
            missing_count += 1
            if fallback:
                item['is_synthetic'] = True
            else:
                logger.warning(f"EEG file missing for {eeg_id} and fallback is disabled.")
                continue
                
        aligned_data.append(item)
        
    logger.info(f"Aligned {len(aligned_data)} EEG pairs. Missing mapping for {missing_count} pairs.")
    return aligned_data
