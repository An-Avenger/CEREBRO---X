"""PyTorch Dataset for 3D MRI classification.

Loads .nii.gz files aligned to clinical visits and applies transforms.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError:
    nib = None

logger = logging.getLogger(__name__)

# Same mapping as clinical pipeline
CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}


class MRIDataset(Dataset):
    """
    Dataset yielding 3D MRI tensors and their target next-visit CDR class.
    """

    def __init__(
        self,
        aligned_df: pd.DataFrame,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """
        Args:
            aligned_df: DataFrame from mri_loader.align_pairs_with_mri().
                        Must contain 'mri_path' and 'next_cdr' columns.
            transform: Callable transform (e.g., MRITransformPipeline).
        """
        if nib is None:
            raise ImportError("nibabel is required to load .nii.gz MRI files. Run: pip install nibabel")

        self.df = aligned_df.copy().reset_index(drop=True)
        self.transform = transform
        
        # Verify columns exist
        if "mri_path" not in self.df.columns or "next_cdr" not in self.df.columns:
            raise ValueError("DataFrame must contain 'mri_path' and 'next_cdr'")

        logger.info(f"Initialized MRIDataset with {len(self.df)} volumes.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        mri_path = row["mri_path"]
        target_cdr = float(row["next_cdr"])
        
        # 1. Load NIfTI
        try:
            nii = nib.load(mri_path)
            # Standardize orientation to RAS or just get array
            # For simplicity, we just extract the raw data array
            data = nii.get_fdata()
        except Exception as e:
            logger.error(f"Failed to load {mri_path}: {e}")
            # Fallback for empty or corrupted paths (useful for Kaggle placeholders)
            data = np.zeros((112, 112, 112), dtype=np.float32)

        tensor = torch.tensor(data, dtype=torch.float32)
        
        # 2. Transform
        if self.transform:
            tensor = self.transform(tensor)
            
        # 3. Label Mapping
        target = CDR_MAP[target_cdr]
        
        return tensor, target
