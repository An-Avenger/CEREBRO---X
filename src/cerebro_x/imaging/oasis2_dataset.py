from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

from cerebro_x.imaging.nifti_loader import NiftiLoader
from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline

logger = logging.getLogger(__name__)

class RealOASIS2MRIDataset(Dataset):
    """
    PyTorch Dataset for Real OASIS-2 T1-weighted NIfTI volumes.
    Loads real MRI data aligned with next-visit CDR score, processes it, and yields (tensor, label, subject_id).
    """

    def __init__(
        self,
        manifest_path: Path | str,
        augment: bool = False,
        target_shape: tuple[int, int, int] = (64, 64, 64)
    ):
        self.manifest_path = Path(manifest_path)
        self.augment = augment
        self.target_shape = target_shape

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"OASIS-2 aligned manifest not found at {self.manifest_path}")

        self.manifest = pd.read_csv(self.manifest_path)
        
        # Verify schema
        required_cols = {"subject_id", "visit_id", "nifti_path", "next_CDR"}
        if not required_cols.issubset(self.manifest.columns):
            raise ValueError(f"Manifest missing required columns. Expected: {required_cols}")

        # Map CDR score to integer class
        def map_cdr(val):
            if pd.isna(val): return 0
            val = float(val)
            if val == 0.0: return 0
            elif val == 0.5: return 1
            elif val == 1.0: return 2
            elif val >= 2.0: return 3
            else: return 0
            
        self.manifest["label"] = self.manifest["next_CDR"].apply(map_cdr)

        # Preprocessing pipeline
        self.pipeline = MRIPreprocessingPipeline(target_shape=self.target_shape)
        
        logger.info(f"Loaded Real OASIS-2 MRI Dataset with {len(self.manifest)} samples")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.manifest.iloc[idx]
        file_path = Path(row["nifti_path"])

        if not file_path.exists():
            raise FileNotFoundError(f"Missing NIfTI file: {file_path}")

        # Load NIfTI (real nibabel load)
        loader = NiftiLoader()
        load_result = loader.load(file_path)
        
        if not load_result.success or load_result.volume is None:
            raise ValueError(f"Failed to load NIfTI {file_path}: {load_result.error_message}")

        # Run preprocessing pipeline (resize, clip, z-score)
        vol_data = self.pipeline.process(load_result.volume)
        
        # Convert to tensor and add channel dim (1, D, H, W)
        vol_tensor = torch.tensor(vol_data, dtype=torch.float32).unsqueeze(0)

        # Augmentation (random flips)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                vol_tensor = torch.flip(vol_tensor, dims=[1])
            if torch.rand(1).item() > 0.5:
                vol_tensor = torch.flip(vol_tensor, dims=[2])

        label = torch.tensor(row["label"], dtype=torch.long)
        subject_id = str(row["subject_id"])
        
        return vol_tensor, label, subject_id
