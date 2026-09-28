from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

# Need nifti processing tools
from cerebro_x.imaging.nifti_loader import NiftiLoader
from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline

logger = logging.getLogger(__name__)

class RealADNIMRIDataset(Dataset):
    """
    PyTorch Dataset for Real ADNI T1-weighted NIfTI volumes.
    Loads real MRI data, processes it through the pipeline, and yields (tensor, label).
    """

    def __init__(
        self,
        manifest_path: Path | str,
        base_dir: Path | str,
        augment: bool = False,
        target_shape: tuple[int, int, int] = (64, 64, 64)
    ):
        self.manifest_path = Path(manifest_path)
        self.base_dir = Path(base_dir)
        self.augment = augment
        self.target_shape = target_shape

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"ADNI manifest not found at {self.manifest_path}")

        self.manifest = pd.read_csv(self.manifest_path)
        
        # Verify schema
        required_cols = {"subject_id", "visit_id", "nifti_path", "cdr_score"}
        if not required_cols.issubset(self.manifest.columns):
            raise ValueError(f"Manifest missing required columns. Expected: {required_cols}")

        # Map CDR score to integer class
        # 0.0 -> 0, 0.5 -> 1, 1.0 -> 2, 2.0 -> 3
        def map_cdr(val):
            if val == 0.0: return 0
            elif val == 0.5: return 1
            elif val == 1.0: return 2
            elif val >= 2.0: return 3
            else: return 0
            
        self.manifest["label"] = self.manifest["cdr_score"].apply(map_cdr)

        # Preprocessing pipeline
        self.pipeline = MRIPreprocessingPipeline(target_shape=self.target_shape)
        
        logger.info(f"Loaded Real ADNI Dataset with {len(self.manifest)} samples from {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        file_path = self.base_dir / row["nifti_path"]

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
        
        return vol_tensor, label
