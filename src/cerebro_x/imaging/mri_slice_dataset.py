"""
PyTorch Dataset for 2D MRI slices.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MRISliceDataset(Dataset):
    """
    Dataset for loading cached 2D MRI slices (.npy) with subject-level splitting.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        cache_path = row["cache_path"]
        label = int(row["label_idx"])
        subject_id = str(row["subject_id"])

        # Load cached slice
        arr = np.load(cache_path).astype(np.float32)  # (224, 224) in [0, 1]

        # Convert to tensor (1, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label, subject_id
