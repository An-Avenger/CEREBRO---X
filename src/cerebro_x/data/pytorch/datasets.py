"""PyTorch Dataset for clinical longitudinal pairs."""
from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from cerebro_x.data.schemas import PairColumns
from cerebro_x.features.clinical import build_feature_matrix


class LongitudinalClinicalDataset(Dataset):
    """
    PyTorch Dataset for tabular clinical features predicting next-visit CDR.
    
    Handles missing value imputation and scaling internally based on the
    provided fit_scaler flag (useful for train vs test splits).
    """
    
    # Class mapping for CDR. We map ordinal values to 0-3 indices for CrossEntropyLoss.
    CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        imputer: SimpleImputer | None = None,
        scaler: StandardScaler | None = None,
        is_train: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            df: The dataframe of pairs.
            imputer: Optional pre-fit imputer. If None and is_train=True, creates one.
            scaler: Optional pre-fit scaler. If None and is_train=True, creates one.
            is_train: If True, fits the imputer and scaler. If False, transforms only.
        """
        self.df = df
        
        # Extract features
        X_df, _ = build_feature_matrix(df)
        self.feature_names = X_df.columns.tolist()
        
        # Extract targets
        raw_targets = df[PairColumns.NEXT_CDR].values
        self.y = np.array([self.CDR_MAP[float(val)] for val in raw_targets], dtype=np.int64)
        
        # Handle scaling and imputation
        X_raw = X_df.values
        
        if is_train:
            self.imputer = SimpleImputer(strategy="median") if imputer is None else imputer
            X_imputed = self.imputer.fit_transform(X_raw)
            
            self.scaler = StandardScaler() if scaler is None else scaler
            self.X = self.scaler.fit_transform(X_imputed)
        else:
            if imputer is None or scaler is None:
                raise ValueError("Must provide imputer and scaler for validation/test sets.")
            self.imputer = imputer
            self.scaler = scaler
            
            X_imputed = self.imputer.transform(X_raw)
            self.X = self.scaler.transform(X_imputed)
            
        # Convert to tensors
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X_tensor[idx], self.y_tensor[idx]

class MultiModalDataset(LongitudinalClinicalDataset):
    """
    Extends LongitudinalClinicalDataset to also load 3D MRI volumes.
    
    If the MRI directory is not provided or the file is missing, it returns
    a dummy tensor (useful for local debugging/scaffolding).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        mri_dir: str | None = None,
        imputer: SimpleImputer | None = None,
        scaler: StandardScaler | None = None,
        is_train: bool = True
    ):
        super().__init__(df, imputer, scaler, is_train)
        self.mri_dir = mri_dir
        
        # Determine the target volume size for resizing
        self.target_shape = (1, 64, 64, 64) # (C, D, H, W)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            X_clinical: (num_features,)
            X_mri: (1, 64, 64, 64)
            y: (1,)
        """
        X_clinical, y = super().__getitem__(idx)
        
        # Scaffolding: In a real scenario, we'd load the .nii.gz file using nibabel
        # based on the Subject ID or visit ID. For now, since we don't have the 
        # actual gigabytes of MRI data locally, we return a dummy tensor of noise.
        # This allows us to build and test the CNN architecture shapes.
        
        # Simulate loading and preprocessing an MRI volume
        X_mri = torch.randn(self.target_shape, dtype=torch.float32)
        
        return X_clinical, X_mri, y
