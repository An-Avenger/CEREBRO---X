"""
Cerebro-X MRI Preprocessing Pipeline.

Provides a unified pipeline for loading, validating, preprocessing, and
extracting features from T1-weighted NIfTI MRI files.

All processing is REAL — no hash-based simulation, no hardcoded values.
Scalar features (nWBV proxy, volume stats) are derived from the actual
voxel data of the uploaded NIfTI volume.

Status: IMPLEMENTED (real NIfTI processing)
CNN3D checkpoint: NOT YET TRAINED — 3D embedding unavailable.
Scalar extraction: REAL (computed from actual voxel intensities).

Pipeline:
    NIfTI file (.nii / .nii.gz)
        ↓ NiftiLoader (nibabel)
        ↓ validate_mri()
        ↓ normalize_mri() — z-score over brain tissue
        ↓ resize_to_model_input() — trilinear → (64, 64, 64)
        ↓ extract_scalar_features() — REAL anatomical proxies
        ↓ MRIProcessingResult
"""
from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from cerebro_x.imaging.nifti_loader import (
    MRILoadResult,
    MRILoadStatus,
    NiftiLoader,
)
from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline

logger = logging.getLogger(__name__)

# Model input shape: must match Lightweight3DCNN input assumption
MODEL_INPUT_SHAPE = (64, 64, 64)

# Background intensity threshold: voxels below this are considered non-brain
BACKGROUND_THRESHOLD = 0.05


@dataclass
class MRIScalarFeatures:
    """
    Real scalar MRI features derived from the actual NIfTI volume.

    These are NOT hash-derived. They are computed from real voxel data:

    nwbv_proxy   — Normalised Whole Brain Volume proxy.
                   = (brain voxel count) / (total voxels in volume)
                   Approximates the standard OASIS nWBV metric.
                   Higher = more brain tissue relative to total head volume.

    brain_fraction — Fraction of voxels above background threshold (0–1).
                     Alias of nwbv_proxy, different semantic framing.

    mean_intensity — Mean intensity of brain tissue voxels (z-scored).
                     Zero-centered after normalization.

    std_intensity  — Std-dev of brain tissue voxel intensities (z-scored).
                     Should be ≈1.0 after z-score normalization.

    volume_shape   — (D, H, W) of the original NIfTI volume before resizing.

    voxel_size_mm  — Physical voxel dimensions in mm (dx, dy, dz).
                     None if not available in header.

    total_voxels   — Total voxels in the volume (D × H × W).

    brain_voxels   — Voxels above background threshold.

    preprocessing_target — Shape the volume was resized to for the CNN.
    """

    nwbv_proxy: float
    brain_fraction: float
    mean_intensity: float
    std_intensity: float
    volume_shape: tuple
    voxel_size_mm: Optional[tuple]
    total_voxels: int
    brain_voxels: int
    preprocessing_target: tuple = MODEL_INPUT_SHAPE
    warnings: list[str] = field(default_factory=list)


@dataclass
class MRIProcessingResult:
    """
    Result of the full MRI processing pipeline.

    success         — True iff the MRI was loaded, validated, and processed.
    scalars         — Real scalar features from the MRI (if success=True).
    preprocessed    — Torch tensor (1, D, H, W) ready for 3D CNN (if success=True).
    error_code      — Short error identifier if success=False.
    error_message   — Human-readable error detail.
    warnings        — Non-fatal warnings.
    cnn_embedding   — 64-dim 3D CNN embedding. None until a trained checkpoint exists.
    cnn_status      — Human-readable status of 3D CNN embedding.
    """

    success: bool
    scalars: Optional[MRIScalarFeatures] = None
    preprocessed: Optional[torch.Tensor] = None   # (1, D, H, W)
    error_code: str = ""
    error_message: str = ""
    warnings: list[str] = field(default_factory=list)
    cnn_embedding: Optional[list[float]] = None
    cnn_status: str = "CNN3D_CHECKPOINT_NOT_AVAILABLE"


def extract_scalar_features(
    preprocessed: torch.Tensor,
    load_result: MRILoadResult,
) -> MRIScalarFeatures:
    """
    Compute real scalar features from the preprocessed MRI tensor.

    Args:
        preprocessed: Normalised 3D tensor (D, H, W) from MRIPreprocessingPipeline.
        load_result:  The MRILoadResult from NiftiLoader (for metadata).

    Returns:
        MRIScalarFeatures with all values derived from actual voxel data.
    """
    vol = preprocessed  # shape (D, H, W)
    total = int(vol.numel())

    # Brain mask: voxels that are NOT background
    # After z-score normalization, background was set to a very negative value.
    # We use a threshold slightly above the absolute minimum to separate brain from BG.
    vol_min = float(vol.min())
    vol_max = float(vol.max())

    # Use mid-range threshold between minimum and 0 to distinguish background
    brain_mask = vol > (vol_min + BACKGROUND_THRESHOLD * (vol_max - vol_min + 1e-6))
    brain_count = int(brain_mask.sum().item())
    brain_fraction = brain_count / total if total > 0 else 0.0

    # nWBV proxy: same as brain_fraction but named to match OASIS convention
    nwbv_proxy = brain_fraction

    # Mean and std of brain-tissue voxels
    if brain_count > 0:
        brain_vals = vol[brain_mask]
        mean_intensity = float(brain_vals.mean().item())
        std_intensity = float(brain_vals.std().item())
    else:
        mean_intensity = 0.0
        std_intensity = 0.0

    return MRIScalarFeatures(
        nwbv_proxy=round(nwbv_proxy, 4),
        brain_fraction=round(brain_fraction, 4),
        mean_intensity=round(mean_intensity, 4),
        std_intensity=round(std_intensity, 4),
        volume_shape=load_result.shape or (0, 0, 0),
        voxel_size_mm=load_result.voxel_size,
        total_voxels=total,
        brain_voxels=brain_count,
        preprocessing_target=MODEL_INPUT_SHAPE,
        warnings=list(load_result.warnings),
    )


def process_mri_bytes(
    file_bytes: bytes,
    filename: str,
    subject_id: str = "UPLOAD",
) -> MRIProcessingResult:
    """
    Full pipeline: bytes → validation → preprocessing → scalar features.

    Accepts raw bytes from an uploaded NIfTI file.
    Writes them to a temp file (required by nibabel), then processes.

    Args:
        file_bytes:  Raw bytes of the .nii or .nii.gz file.
        filename:    Original filename (used for format detection).
        subject_id:  Patient identifier (for logging).

    Returns:
        MRIProcessingResult — always returned, never raises.
    """
    # ── 1. Validate extension ─────────────────────────────────────────────────
    fname_lower = filename.lower()
    if not (fname_lower.endswith(".nii") or fname_lower.endswith(".nii.gz")):
        return MRIProcessingResult(
            success=False,
            error_code="INVALID_EXTENSION",
            error_message=(
                f"Unsupported file extension: '{filename}'. "
                "Please upload a NIfTI file (.nii or .nii.gz)."
            ),
        )

    # ── 2. Check nibabel availability ─────────────────────────────────────────
    try:
        import nibabel  # noqa: F401
    except ImportError:
        return MRIProcessingResult(
            success=False,
            error_code="NIBABEL_UNAVAILABLE",
            error_message=(
                "nibabel is not installed. "
                "Run: pip install nibabel"
            ),
        )

    # ── 3. Write bytes to a temp file (nibabel needs a file path) ─────────────
    suffix = ".nii.gz" if fname_lower.endswith(".nii.gz") else ".nii"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
    except Exception as e:
        return MRIProcessingResult(
            success=False,
            error_code="TEMP_FILE_ERROR",
            error_message=f"Could not write temporary file: {e}",
        )

    # ── 4. Load with NiftiLoader ──────────────────────────────────────────────
    loader = NiftiLoader(min_dim_voxels=32, max_dim_voxels=512)
    load_result = loader.load(
        path=tmp_path,
        subject_id=subject_id,
        visit=1,
        validate_subject_in_filename=False,
    )

    # Clean up temp file
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    # ── 5. Handle load failures ───────────────────────────────────────────────
    if load_result.status == MRILoadStatus.MISSING:
        return MRIProcessingResult(
            success=False,
            error_code="FILE_MISSING",
            error_message="MRI file could not be read from disk.",
        )
    elif load_result.status == MRILoadStatus.UNREADABLE:
        return MRIProcessingResult(
            success=False,
            error_code="INVALID_NIFTI",
            error_message=(
                f"Could not read NIfTI file: {load_result.message}. "
                "Ensure the file is a valid, non-corrupt NIfTI volume."
            ),
        )
    elif load_result.status == MRILoadStatus.WRONG_DIMENSIONS:
        return MRIProcessingResult(
            success=False,
            error_code="WRONG_DIMENSIONS",
            error_message=(
                f"Unexpected volume dimensions {load_result.shape}. "
                "Expected a 3D T1-weighted MRI with each axis between 32 and 512 voxels. "
                f"Detail: {load_result.message}"
            ),
        )
    elif load_result.status == MRILoadStatus.EMPTY:
        return MRIProcessingResult(
            success=False,
            error_code="EMPTY_VOLUME",
            error_message=(
                "The MRI volume is empty (all zeros). "
                "This may indicate a corrupt acquisition or wrong file."
            ),
        )
    elif load_result.status == MRILoadStatus.STUB:
        return MRIProcessingResult(
            success=False,
            error_code="NIBABEL_STUB",
            error_message="nibabel not installed. Cannot process real MRI files.",
        )
    elif not load_result.ok:
        return MRIProcessingResult(
            success=False,
            error_code=load_result.status.name,
            error_message=load_result.message,
        )

    # ── 6. Preprocess ──────────────────────────────────────────────────────────
    try:
        pipeline = MRIPreprocessingPipeline(
            target_shape=MODEL_INPUT_SHAPE,
            clip_percentiles=(0.5, 99.5),
            background_threshold=1e-3,
        )
        # load_result.tensor is (D, H, W) float32
        preprocessed = pipeline(load_result.tensor)   # → (D, H, W)
    except Exception as e:
        return MRIProcessingResult(
            success=False,
            error_code="PREPROCESSING_FAILED",
            error_message=f"MRI preprocessing failed: {e}",
            warnings=list(load_result.warnings),
        )

    # ── 7. Extract real scalar features ───────────────────────────────────────
    try:
        scalars = extract_scalar_features(preprocessed, load_result)
    except Exception as e:
        return MRIProcessingResult(
            success=False,
            error_code="FEATURE_EXTRACTION_FAILED",
            error_message=f"Scalar feature extraction failed: {e}",
            warnings=list(load_result.warnings),
        )

    # ── 8. Add channel dim for CNN: (1, D, H, W) ──────────────────────────────
    preprocessed_4d = preprocessed.unsqueeze(0)   # (1, 64, 64, 64)

    logger.info(
        "MRI processed: subject=%s original_shape=%s → preprocessed=%s "
        "nwbv_proxy=%.4f brain_voxels=%d/%d",
        subject_id,
        load_result.shape,
        preprocessed_4d.shape,
        scalars.nwbv_proxy,
        scalars.brain_voxels,
        scalars.total_voxels,
    )

    return MRIProcessingResult(
        success=True,
        scalars=scalars,
        preprocessed=preprocessed_4d,
        warnings=scalars.warnings,
        cnn_embedding=None,
        cnn_status=(
            "CNN3D_CHECKPOINT_NOT_AVAILABLE: "
            "The Lightweight3DCNN architecture exists in models/deep/cnn3d.py "
            "but no trained checkpoint is present in artifacts/. "
            "Run training/train_cnn3d.py to train the model. "
            "Using nwbv_proxy from real voxel data for bimodal prediction fallback."
        ),
    )
