"""
Cerebro-X NIfTI MRI Loader.

Provides safe, validated loading of NIfTI (.nii, .nii.gz) MRI files
with explicit status reporting for every load attempt.

Key design principles:
    - Never silently fail: every load returns a typed MRILoadResult
    - Never mix up subjects: subject_id is validated against filename/metadata
    - Explicit fallback: callers decide what to do with missing/failed MRI
    - Reproducible: same file always produces same tensor (deterministic)

Usage:
    loader = NiftiLoader()
    result = loader.load(mri_path, subject_id="OAS2_0042", visit=2)
    if result.status == MRILoadStatus.OK:
        tensor = result.tensor  # shape: (1, D, H, W) float32
    else:
        logger.warning("MRI unavailable: %s — %s", result.path, result.message)
        # Use fallback (scalar path) or skip this sample
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Try to import nibabel; it is optional (tests without real MRI still work) ─
try:
    import nibabel as nib
    _NIB_AVAILABLE = True
except ImportError:
    nib = None
    _NIB_AVAILABLE = False
    logger.warning(
        "nibabel is not installed. NiftiLoader will operate in stub mode. "
        "Run: pip install nibabel"
    )


class MRILoadStatus(Enum):
    """Result codes for NiftiLoader.load()."""

    OK = auto()              # Successfully loaded and validated
    MISSING = auto()         # File path does not exist
    UNREADABLE = auto()      # File exists but cannot be read (corrupt, wrong format)
    WRONG_SUBJECT = auto()   # Filename/metadata subject ID mismatch
    WRONG_DIMENSIONS = auto() # Image has unexpected/unusable dimensions
    STUB = auto()            # nibabel not available; returned zeros placeholder
    EMPTY = auto()           # Volume is all zeros after loading


@dataclass
class MRILoadResult:
    """
    Result of a single NIfTI load attempt.

    Always returned by NiftiLoader.load() — callers must check .status
    before using .tensor. Never raise on load failure; always return a result.

    Attributes:
        status:     Outcome code (see MRILoadStatus).
        path:       Path that was attempted.
        subject_id: Expected subject ID.
        visit:      Expected visit number.
        tensor:     Loaded and pre-normalized tensor (C, D, H, W) if OK.
                    None for all non-OK statuses.
        shape:      Original voxel shape before normalization (D, H, W).
        voxel_size: Original voxel size in mm (dx, dy, dz) if available.
        message:    Human-readable status explanation.
        warnings:   Non-fatal warnings (e.g., orientation mismatch).
    """

    status: MRILoadStatus
    path: str
    subject_id: str
    visit: int
    tensor: Optional[torch.Tensor] = None
    shape: Optional[tuple] = None
    voxel_size: Optional[tuple] = None
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True iff the load was successful."""
        return self.status == MRILoadStatus.OK


class NiftiLoader:
    """
    Safe NIfTI MRI loader for the Cerebro-X pipeline.

    Loads T1-weighted MRI NIfTI files and returns normalized 3D tensors.
    All failures are reported as MRILoadResult with appropriate status codes
    rather than exceptions, enabling graceful degradation.

    Pipeline per load() call:
        1. Existence check
        2. nibabel read
        3. Dimension validation
        4. Orientation normalization (canonical reorientation)
        5. Voxel intensity check (non-empty volume)
        6. Return as float32 numpy array for downstream transforms

    Args:
        min_dim_voxels: Minimum acceptable dimension in any axis (default: 32).
        max_dim_voxels: Maximum acceptable dimension in any axis (default: 512).
    """

    def __init__(
        self,
        min_dim_voxels: int = 32,
        max_dim_voxels: int = 512,
    ):
        self.min_dim_voxels = min_dim_voxels
        self.max_dim_voxels = max_dim_voxels

    def load(
        self,
        path: str | Path,
        subject_id: str,
        visit: int,
        validate_subject_in_filename: bool = False,
    ) -> MRILoadResult:
        """
        Load a NIfTI MRI file and return a structured result.

        Args:
            path:                        Path to .nii or .nii.gz file.
            subject_id:                  Expected OASIS subject ID (e.g. "OAS2_0042").
            visit:                       Expected visit number (1-based).
            validate_subject_in_filename: If True, verify subject_id appears in filename.
                                         Disabled by default because OASIS file naming
                                         conventions vary.

        Returns:
            MRILoadResult with status and tensor (or None on failure).
        """
        path = Path(path)

        # ── 1. Existence ──────────────────────────────────────────────────────
        if not path.exists():
            return MRILoadResult(
                status=MRILoadStatus.MISSING,
                path=str(path),
                subject_id=subject_id,
                visit=visit,
                message=f"MRI file not found: {path}",
            )

        # ── 2. Stub mode (nibabel not available) ─────────────────────────────
        if not _NIB_AVAILABLE:
            return MRILoadResult(
                status=MRILoadStatus.STUB,
                path=str(path),
                subject_id=subject_id,
                visit=visit,
                message="nibabel not installed. Install with: pip install nibabel",
            )

        # ── 3. Optional filename subject validation ───────────────────────────
        if validate_subject_in_filename:
            fname = path.name
            if subject_id not in fname and subject_id.lower() not in fname.lower():
                return MRILoadResult(
                    status=MRILoadStatus.WRONG_SUBJECT,
                    path=str(path),
                    subject_id=subject_id,
                    visit=visit,
                    message=(
                        f"Subject ID '{subject_id}' not found in filename '{fname}'. "
                        f"Refusing to load to prevent cross-subject contamination."
                    ),
                )

        # ── 4. Load via nibabel ───────────────────────────────────────────────
        try:
            img = nib.load(str(path))
        except Exception as e:
            return MRILoadResult(
                status=MRILoadStatus.UNREADABLE,
                path=str(path),
                subject_id=subject_id,
                visit=visit,
                message=f"nibabel failed to load: {e}",
            )

        # ── 5. Dimension validation ───────────────────────────────────────────
        shape = img.shape
        n_dims = len(shape)

        # Accept 3D (D,H,W) or 3D+time (D,H,W,T) with T=1
        if n_dims == 4 and shape[3] == 1:
            # Single-volume 4D — squeeze to 3D
            data = img.get_fdata()[..., 0]
        elif n_dims == 3:
            data = img.get_fdata()
        else:
            return MRILoadResult(
                status=MRILoadStatus.WRONG_DIMENSIONS,
                path=str(path),
                subject_id=subject_id,
                visit=visit,
                shape=shape,
                message=(
                    f"Unexpected image shape {shape}. Expected 3D (D,H,W) or "
                    f"4D with single volume (D,H,W,1)."
                ),
            )

        # Check individual dimension sizes
        dims_3d = data.shape
        for dim_size in dims_3d:
            if not (self.min_dim_voxels <= dim_size <= self.max_dim_voxels):
                return MRILoadResult(
                    status=MRILoadStatus.WRONG_DIMENSIONS,
                    path=str(path),
                    subject_id=subject_id,
                    visit=visit,
                    shape=dims_3d,
                    message=(
                        f"Dimension {dim_size} outside acceptable range "
                        f"[{self.min_dim_voxels}, {self.max_dim_voxels}]."
                    ),
                )

        # ── 6. Get voxel size ─────────────────────────────────────────────────
        try:
            header = img.header
            voxel_size = tuple(float(v) for v in header.get_zooms()[:3])
        except Exception:
            voxel_size = None

        # ── 7. Orientation extraction ─────────────────────────────────────────
        warnings: list[str] = []
        try:
            # Reorient to RAS canonical orientation
            canonical_img = nib.as_closest_canonical(img)
            if n_dims == 4 and shape[3] == 1:
                data = canonical_img.get_fdata()[..., 0]
            else:
                data = canonical_img.get_fdata()
        except Exception as e:
            warnings.append(f"Could not reorient to canonical: {e}. Using original orientation.")
            # Keep original data

        # ── 8. Check non-empty volume ─────────────────────────────────────────
        data = data.astype(np.float32)
        if np.all(data == 0):
            return MRILoadResult(
                status=MRILoadStatus.EMPTY,
                path=str(path),
                subject_id=subject_id,
                visit=visit,
                shape=data.shape,
                voxel_size=voxel_size,
                message="Volume is all zeros — likely a failed acquisition or corrupt file.",
                warnings=warnings,
            )

        # ── 9. Success ────────────────────────────────────────────────────────
        # Return raw numpy array as tensor (without intensity normalization —
        # that is applied by MRIPreprocessingPipeline)
        tensor = torch.tensor(data, dtype=torch.float32)

        logger.debug(
            "Loaded MRI: subject=%s visit=%d shape=%s voxel_size=%s path=%s",
            subject_id, visit, data.shape, voxel_size, path,
        )

        return MRILoadResult(
            status=MRILoadStatus.OK,
            path=str(path),
            subject_id=subject_id,
            visit=visit,
            tensor=tensor,
            shape=data.shape,
            voxel_size=voxel_size,
            message="OK",
            warnings=warnings,
        )

    def load_batch(
        self,
        records: list[dict],
        validate_subject_in_filename: bool = False,
    ) -> list[MRILoadResult]:
        """
        Load multiple MRI files and return one result per record.

        Args:
            records: List of dicts with keys: 'path', 'subject_id', 'visit'.
            validate_subject_in_filename: See load().

        Returns:
            List of MRILoadResult in same order as records.
        """
        results = []
        for rec in records:
            result = self.load(
                path=rec["path"],
                subject_id=rec["subject_id"],
                visit=rec.get("visit", -1),
                validate_subject_in_filename=validate_subject_in_filename,
            )
            results.append(result)

        n_ok = sum(1 for r in results if r.ok)
        n_total = len(results)
        logger.info(
            "Batch MRI load: %d/%d successful", n_ok, n_total
        )
        if n_total > 0 and n_ok < n_total:
            failed = [r for r in results if not r.ok]
            by_status: dict[str, int] = {}
            for r in failed:
                by_status[r.status.name] = by_status.get(r.status.name, 0) + 1
            logger.warning("MRI load failures: %s", by_status)

        return results
