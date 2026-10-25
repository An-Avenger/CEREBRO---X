"""
DICOM -> 2D Slice Extraction Pipeline for ADNI MRI data.

Optimized for fast CPU execution:
    1. Reads DICOM headers first (`stop_before_pixels=True`) to determine spatial ordering.
    2. Directly accesses and loads only the required central slice planes.
    3. Normalises intensity and resizes to target dimension (e.g. 224x224).
    4. Caches preprocessed arrays as `.npy` files for instant reloading.
    5. Saves complete slice manifest linking subject, slice index, offset, and verified clinical label.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SliceExtractionConfig:
    """Configuration for the 2D slice extraction pipeline."""

    dicom_root: str = "Clinical_T1w_Imaging_Cohort_MRI_2/ADNI"
    cohort_csv: str = "data/cohort_scans_labeled.csv"
    output_dir: str = "data/processed/mri_2d"
    cache_dir: str = "data/processed/mri_2d/slices"

    target_size: tuple[int, int] = (224, 224)
    slice_offsets: list[int] = field(default_factory=lambda: [-10, -5, 0, 5, 10])
    min_brain_fraction: float = 0.05

    clip_percentile_low: float = 1.0
    clip_percentile_high: float = 99.0

    label_to_class: dict[str, int] = field(
        default_factory=lambda: {"CN": 0, "MCI": 1, "AD": 2}
    )


@dataclass
class SliceRecord:
    """Metadata for an extracted and cached 2D MRI slice."""

    subject_id: str
    image_id: str
    slice_index: int
    offset: int
    cache_path: str
    label_name: str
    label_idx: int
    status: str = "ok"
    exclusion_reason: str = ""


def normalise_and_resize_slice(
    slice_2d: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> np.ndarray:
    """Normalise intensity to [0, 1] and resize to target_size."""
    arr = slice_2d.astype(np.float32)
    # Background baseline shift
    arr = arr - arr.min()

    p1 = np.percentile(arr, clip_low)
    p99 = np.percentile(arr, clip_high)

    if p99 > p1:
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1)
    else:
        arr = np.zeros_like(arr)

    arr_uint8 = (arr * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(arr_uint8)
    img_pil = img_pil.resize(target_size, Image.Resampling.BILINEAR)

    return (np.array(img_pil, dtype=np.float32) / 255.0).astype(np.float32)


def process_scan_fast(scan_row: dict, config: SliceExtractionConfig) -> list[SliceRecord]:
    """Fast extraction: reads headers only, then reads target slices."""
    try:
        import pydicom
    except ImportError:
        logger.error("pydicom is required.")
        return []

    records = []
    subject_id = str(scan_row["subject_id"])
    image_id = str(scan_row.get("image_id", "0"))
    scan_path = Path(scan_row["scan_path"])
    label_name = str(scan_row["final_label"])
    label_idx = config.label_to_class.get(label_name, -1)

    subj_cache_dir = Path(config.cache_dir) / subject_id
    subj_cache_dir.mkdir(parents=True, exist_ok=True)

    dcm_files = sorted(scan_path.glob("*.dcm"))
    if not dcm_files:
        dcm_files = [f for f in sorted(scan_path.glob("*")) if f.is_file() and not f.name.startswith(".")]

    if len(dcm_files) < 3:
        records.append(
            SliceRecord(
                subject_id=subject_id,
                image_id=image_id,
                slice_index=-1,
                offset=0,
                cache_path="",
                label_name=label_name,
                label_idx=label_idx,
                status="error",
                exclusion_reason=f"Too few DICOM files: {len(dcm_files)}",
            )
        )
        return records

    # Read headers to sort slices along slice normal or position
    file_sort_keys = []
    norm = None
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            if norm is None:
                iop = getattr(ds, "ImageOrientationPatient", None)
                if iop is not None and len(iop) >= 6:
                    r = np.array(iop[0:3], dtype=float)
                    c = np.array(iop[3:6], dtype=float)
                    norm = np.cross(r, c)
                    norm_len = np.linalg.norm(norm)
                    if norm_len > 1e-6:
                        norm = norm / norm_len
                    else:
                        norm = np.array([0.0, 0.0, 1.0])
                else:
                    norm = np.array([0.0, 0.0, 1.0])

            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) >= 3:
                pos = float(np.dot(np.array(ipp, dtype=float), norm))
            else:
                pos = float(getattr(ds, "InstanceNumber", 0))
            file_sort_keys.append((pos, f))
        except Exception:
            continue

    if len(file_sort_keys) < 3:
        records.append(
            SliceRecord(
                subject_id=subject_id,
                image_id=image_id,
                slice_index=-1,
                offset=0,
                cache_path="",
                label_name=label_name,
                label_idx=label_idx,
                status="error",
                exclusion_reason="Failed to parse slice positions",
            )
        )
        return records

    file_sort_keys.sort(key=lambda x: x[0])
    total_slices = len(file_sort_keys)
    mid = total_slices // 2

    for off in config.slice_offsets:
        idx = mid + off
        if idx < 0 or idx >= total_slices:
            continue

        cache_file = subj_cache_dir / f"{image_id}_slice_{idx}_off_{off}.npy"
        if not cache_file.exists():
            target_f = file_sort_keys[idx][1]
            try:
                ds = pydicom.dcmread(str(target_f), force=True)
                raw = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                raw = raw * slope + intercept
                sl_proc = normalise_and_resize_slice(
                    raw,
                    target_size=config.target_size,
                    clip_low=config.clip_percentile_low,
                    clip_high=config.clip_percentile_high,
                )
                np.save(str(cache_file), sl_proc)
            except Exception as e:
                logger.warning("Error processing slice %s: %s", target_f, e)
                continue

        records.append(
            SliceRecord(
                subject_id=subject_id,
                image_id=image_id,
                slice_index=idx,
                offset=off,
                cache_path=str(cache_file),
                label_name=label_name,
                label_idx=label_idx,
                status="ok",
                exclusion_reason="",
            )
        )

    return records


def run_preprocessing_pipeline(
    config: Optional[SliceExtractionConfig] = None,
    max_workers: int = 4,
) -> list[SliceRecord]:
    """Run fast parallel extraction across cohort scans."""
    import pandas as pd

    if config is None:
        config = SliceExtractionConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    df_cohort = pd.read_csv(config.cohort_csv)
    logger.info("Extracting slices for %d subjects using %d workers...", len(df_cohort), max_workers)

    rows = [r.to_dict() for _, r in df_cohort.iterrows()]
    all_records = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_scan_fast, row, config) for row in rows]
        for i, fut in enumerate(futures):
            recs = fut.result()
            all_records.extend(recs)
            if (i + 1) % 50 == 0 or (i + 1) == len(rows):
                logger.info("Progress: %d/%d subjects processed.", i + 1, len(rows))

    df_manifest = pd.DataFrame([r.__dict__ for r in all_records])
    manifest_path = output_dir / "slices_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    logger.info("Preprocessing complete! Saved %d slice records to %s", len(df_manifest), manifest_path)

    return all_records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_preprocessing_pipeline()
