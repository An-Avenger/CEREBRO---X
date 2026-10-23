"""
src/cerebro_x/imaging/adni_dataset.py
---------------------------------------
PyTorch Dataset for REAL ADNI T1-weighted NIfTI volumes.

Requires a manifest CSV with columns:
    subject_id, visit_id, nifti_path, cdr_score, [image_id, scan_date, label]

Produced by:
    scripts/build_adni_manifest.py  →  manifest_imaging.csv
    scripts/prepare_adni_clinical.py  →  manifest_with_labels.csv

Design principles:
    - Fails closed: raises ValueError/FileNotFoundError on bad data.
    - Never silently maps invalid CDR values to a default class.
    - Uses MRILoadResult.tensor (not .volume — which does not exist).
    - Does not fall back to synthetic data.
    - Preprocessing statistics are computed per-sample (standard for MRI).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from cerebro_x.imaging.nifti_loader import NiftiLoader, MRILoadStatus
from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline

logger = logging.getLogger(__name__)

# Valid CDR values → integer class mapping.
# Any CDR value NOT in this mapping is INVALID and must not be silently converted.
CDR_CLASS_MAP: dict[float, int] = {
    0.0: 0,   # Normal
    0.5: 1,   # Very Mild / MCI
    1.0: 2,   # Mild AD
    2.0: 3,   # Moderate AD
    3.0: 3,   # Severe AD — merged with Moderate for cohort size reasons
}

CDR_CLASS_LABELS: dict[int, str] = {
    0: "Normal",
    1: "Very_Mild_MCI",
    2: "Mild_AD",
    3: "Moderate_Severe_AD",
}

# Required manifest columns
REQUIRED_COLS = {"subject_id", "visit_id", "nifti_path", "cdr_score"}


def map_cdr_to_class(val: float, row_index: int) -> int:
    """
    Map a CDR float value to an integer class label.

    Raises ValueError for any value not in CDR_CLASS_MAP.
    Never silently converts unrecognised values to class 0.
    """
    try:
        val_f = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Row {row_index}: cdr_score '{val}' is not a valid float. "
            f"Cannot create a label. Remove or fix this entry."
        ) from e

    if val_f not in CDR_CLASS_MAP:
        raise ValueError(
            f"Row {row_index}: cdr_score={val_f} is not in the valid CDR set "
            f"{sorted(CDR_CLASS_MAP.keys())}. "
            f"This value is invalid or unsupported. Remove or fix this entry instead of "
            f"silently mapping it to class 0."
        )

    return CDR_CLASS_MAP[val_f]


class RealADNIMRIDataset(Dataset):
    """
    PyTorch Dataset for Real ADNI T1-weighted NIfTI volumes.

    Args:
        manifest_path: Path to manifest CSV with columns:
            subject_id, visit_id, nifti_path, cdr_score
        nifti_base_dir: Root directory that nifti_path values are relative to.
            If nifti_path values are absolute, set this to Path('/').
        augment: If True, apply random left-right and anterior-posterior flips.
        target_shape: (D, H, W) to resize all volumes to.
        validate_subject_in_filename: If True, verify subject_id appears in the
            NIfTI filename before loading (extra safety against wrong-file loading).

    Raises:
        FileNotFoundError: If manifest does not exist.
        ValueError: If manifest is missing required columns or contains invalid CDR values.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        nifti_base_dir: Path | str,
        augment: bool = False,
        target_shape: tuple[int, int, int] = (64, 64, 64),
        validate_subject_in_filename: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.nifti_base_dir = Path(nifti_base_dir)
        self.augment = augment
        self.target_shape = target_shape
        self.validate_subject_in_filename = validate_subject_in_filename

        # ── Load and validate manifest ────────────────────────────────────────
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"ADNI manifest not found: {self.manifest_path}. "
                f"Run scripts/build_adni_manifest.py and scripts/prepare_adni_clinical.py first."
            )

        self.manifest = pd.read_csv(self.manifest_path)
        logger.info(
            "Loaded manifest: %d rows, %d subjects, from %s",
            len(self.manifest),
            self.manifest["subject_id"].nunique() if "subject_id" in self.manifest.columns else "?",
            self.manifest_path,
        )

        # Validate schema
        missing_cols = REQUIRED_COLS - set(self.manifest.columns)
        if missing_cols:
            raise ValueError(
                f"Manifest missing required columns: {missing_cols}. "
                f"Found: {list(self.manifest.columns)}. "
                f"Re-run scripts/prepare_adni_clinical.py to generate a complete manifest."
            )

        # Validate CDR values and build label column
        # This raises ValueError on any invalid CDR value — fail closed.
        labels = []
        for idx, row in self.manifest.iterrows():
            cdr_val = row["cdr_score"]
            if pd.isna(cdr_val):
                raise ValueError(
                    f"Row {idx}: cdr_score is NaN for subject={row['subject_id']}, "
                    f"visit={row['visit_id']}. Remove entries without labels or run "
                    f"prepare_adni_clinical.py to fill them."
                )
            label = map_cdr_to_class(cdr_val, row_index=idx)
            labels.append(label)

        self.manifest = self.manifest.copy()
        self.manifest["label"] = labels

        # ── Log class distribution ────────────────────────────────────────────
        label_counts = self.manifest["label"].value_counts().sort_index()
        logger.info("CDR class distribution:")
        for lbl, cnt in label_counts.items():
            logger.info(
                "  Class %d (%s): %d scans (%.1f%%)",
                lbl, CDR_CLASS_LABELS.get(int(lbl), "?"),
                cnt, 100 * cnt / len(self.manifest),
            )

        # ── Preprocessing pipeline ────────────────────────────────────────────
        self.pipeline = MRIPreprocessingPipeline(target_shape=self.target_shape)
        self._loader = NiftiLoader(min_dim_voxels=32, max_dim_voxels=512)

        logger.info(
            "RealADNIMRIDataset ready: %d samples | augment=%s | target_shape=%s",
            len(self.manifest), augment, target_shape,
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        subject_id = str(row["subject_id"])
        visit_id   = str(row["visit_id"])

        # Resolve NIfTI path
        nifti_rel = str(row["nifti_path"])
        file_path = self.nifti_base_dir / nifti_rel

        # ── Load via NiftiLoader ──────────────────────────────────────────────
        load_result = self._loader.load(
            path=file_path,
            subject_id=subject_id,
            visit=1,  # visit as int not needed for ADNI; kept for API compatibility
            validate_subject_in_filename=self.validate_subject_in_filename,
        )

        if load_result.status != MRILoadStatus.OK:
            raise ValueError(
                f"Failed to load NIfTI for subject={subject_id}, visit={visit_id}, "
                f"path={file_path}: status={load_result.status.name}, "
                f"message={load_result.message}"
            )

        if load_result.tensor is None:
            raise ValueError(
                f"NiftiLoader returned OK status but tensor is None for {file_path}."
            )

        # ── Preprocess ────────────────────────────────────────────────────────
        # load_result.tensor is shape (D, H, W), float32
        vol_3d = load_result.tensor
        vol_preprocessed = self.pipeline(vol_3d)  # → (D, H, W)

        # Add channel dim → (1, D, H, W)
        vol_tensor = vol_preprocessed.unsqueeze(0)

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment:
            # Left-right flip (sagittal axis = dim 1)
            if torch.rand(1).item() > 0.5:
                vol_tensor = torch.flip(vol_tensor, dims=[1])
            # Anterior-posterior flip (coronal axis = dim 2)
            if torch.rand(1).item() > 0.5:
                vol_tensor = torch.flip(vol_tensor, dims=[2])

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return vol_tensor, label

    @property
    def num_classes(self) -> int:
        return len(CDR_CLASS_MAP)

    @property
    def class_counts(self) -> dict[int, int]:
        """Return {class_label: count} for computing class weights."""
        return {
            int(lbl): int(cnt)
            for lbl, cnt in self.manifest["label"].value_counts().items()
        }

    @property
    def subjects(self) -> list[str]:
        """Unique subject IDs in this dataset."""
        return list(self.manifest["subject_id"].unique())
