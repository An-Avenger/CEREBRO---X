#!/usr/bin/env python
"""
scripts/build_adni_manifest.py
--------------------------------
Build a training-ready manifest CSV from the ADNI NIfTI output directory
(produced by dcm2niix) and the LONI IDA imaging manifest.

The output manifest has the schema expected by RealADNIMRIDataset:
    subject_id, visit_id, image_id, series_id, scan_date,
    image_description, nifti_path, [cdr_score — filled in Phase 3]

This script does NOT add CDR scores. Use prepare_adni_clinical.py for that.

Usage:
    python scripts/build_adni_manifest.py \\
        --nifti-dir   data/raw/mri/ADNI/nifti \\
        --loni-manifest "C:/Users/HP/.../Clinical_T1w_Imaging_Cohort_Manifest_15Oct2026.csv" \\
        --out-dir     data/raw/mri/ADNI \\
        --out-file    manifest_imaging.csv

After running, review manifest_imaging.csv and then run:
    python scripts/prepare_adni_clinical.py ...
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("build_adni_manifest")

# dcm2niix with -f %i_%s_%d produces names like:
#   {image_id}_{series_desc}_{date}.nii.gz
# We try to extract the image_id (I\d+ or numeric) from the filename.
IMAGE_ID_PATTERNS = [
    re.compile(r"I(\d+)"),          # Matches I10216986 in ADNI filenames
    re.compile(r"^(\d+)_"),         # Starts with numeric image_id
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a training-ready imaging manifest from NIfTI dir + LONI manifest."
    )
    parser.add_argument(
        "--nifti-dir",
        required=True,
        help="Directory containing .nii.gz files produced by dcm2niix.",
    )
    parser.add_argument(
        "--loni-manifest",
        required=True,
        help="Path to the LONI IDA imaging manifest CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/mri/ADNI",
        help="Output directory for the built manifest.",
    )
    parser.add_argument(
        "--out-file",
        default="manifest_imaging.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--prefer-mprage",
        action="store_true",
        default=True,
        help=(
            "When a subject has multiple scans per visit, prefer MPRAGE/IR-FSPGR series "
            "and pick the one with the highest image_id (latest acquisition). Default: True."
        ),
    )
    return parser.parse_args()


def extract_image_id_from_stem(stem: str) -> str | None:
    """Try to extract image_id from a dcm2niix output filename stem."""
    for pat in IMAGE_ID_PATTERNS:
        m = pat.search(stem)
        if m:
            return m.group(1)
    return None


def scan_nifti_dir(nifti_dir: Path) -> pd.DataFrame:
    """
    Scan nifti_dir recursively for all .nii and .nii.gz files.

    Returns a DataFrame with columns:
        nifti_path (relative to nifti_dir), stem, extracted_image_id
    """
    nifti_files = list(nifti_dir.rglob("*.nii.gz")) + list(nifti_dir.rglob("*.nii"))
    logger.info("Found %d NIfTI files in %s", len(nifti_files), nifti_dir)

    rows = []
    for f in nifti_files:
        stem = f.name
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        extracted_id = extract_image_id_from_stem(stem)
        try:
            rel = str(f.relative_to(nifti_dir))
        except ValueError:
            rel = str(f)
        rows.append({
            "nifti_path": rel,
            "nifti_stem": stem,
            "extracted_image_id": extracted_id,
            "abs_path": str(f),
        })

    return pd.DataFrame(rows)


def load_loni_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error("LONI manifest not found: %s", path)
        sys.exit(1)
    df = pd.read_csv(path)
    df["image_id"] = df["image_id"].astype(str)
    logger.info("LONI manifest: %d rows, %d subjects", len(df), df["subject_id"].nunique())
    return df


def select_preferred_scan(group: pd.DataFrame, prefer_mprage: bool) -> pd.Series:
    """
    Given multiple scans for the same subject+visit, pick one.

    Priority:
      1. Contains 'MPRAGE' or 'IR-FSPGR' in image_description.
      2. Does NOT contain 'Repeat' or 'RPT' (prefer original, not repeat).
      3. Highest image_id (latest).
    """
    if len(group) == 1:
        return group.iloc[0]

    if prefer_mprage:
        preferred_types = group[
            group["image_description"].str.contains(
                "MPRAGE|IR-FSPGR", case=False, na=False
            )
        ]
        if not preferred_types.empty:
            group = preferred_types

        # Avoid 'Repeat' unless that's all we have
        non_repeat = group[
            ~group["image_description"].str.contains("Repeat|RPT", case=False, na=False)
        ]
        if not non_repeat.empty:
            group = non_repeat

    # Pick highest image_id (most recent acquisition within session)
    group = group.copy()
    group["_image_id_int"] = pd.to_numeric(group["image_id"], errors="coerce")
    best = group.sort_values("_image_id_int", ascending=False).iloc[0]
    return best.drop(labels=["_image_id_int"])


def build_manifest(
    nifti_df: pd.DataFrame,
    loni_df: pd.DataFrame,
    prefer_mprage: bool,
) -> tuple[pd.DataFrame, dict]:
    """
    Join NIfTI files to LONI manifest rows by image_id.

    Returns (merged_df, stats_dict).
    """
    stats = {
        "nifti_files_found": len(nifti_df),
        "loni_entries": len(loni_df),
        "matched_by_image_id": 0,
        "unmatched_nifti": 0,
        "loni_entries_without_nifti": 0,
        "subjects_with_at_least_one_scan": 0,
        "duplicate_visit_scans_deduplicated": 0,
    }

    if nifti_df.empty:
        logger.warning(
            "No NIfTI files found. Manifest will be empty. "
            "Run dcm2niix on the DICOM archive first."
        )
        stats["loni_entries_without_nifti"] = len(loni_df)
        return pd.DataFrame(), stats

    # Merge on image_id (extracted from NIfTI filename vs LONI column)
    loni_df = loni_df.copy()
    loni_df["image_id_str"] = loni_df["image_id"].astype(str)

    nifti_df = nifti_df.copy()
    nifti_df["image_id_str"] = nifti_df["extracted_image_id"].astype(str)

    merged = loni_df.merge(
        nifti_df[["nifti_path", "image_id_str"]],
        on="image_id_str",
        how="left",
    )

    n_matched = merged["nifti_path"].notna().sum()
    stats["matched_by_image_id"] = int(n_matched)
    stats["loni_entries_without_nifti"] = int(merged["nifti_path"].isna().sum())

    unmatched_nifti_ids = set(nifti_df["image_id_str"]) - set(
        merged.loc[merged["nifti_path"].notna(), "image_id_str"]
    )
    stats["unmatched_nifti"] = len(unmatched_nifti_ids)

    # Filter to matched only
    matched = merged[merged["nifti_path"].notna()].copy()

    if matched.empty:
        logger.warning(
            "No matches found between NIfTI files and LONI manifest. "
            "Check dcm2niix output filename format."
        )
        return pd.DataFrame(), stats

    # ── Deduplicate: one scan per subject + visit ─────────────────────────────
    before_dedup = len(matched)
    deduped_rows = []
    for (subject_id, image_visit), group in matched.groupby(
        ["subject_id", "image_visit"], sort=False
    ):
        best = select_preferred_scan(group, prefer_mprage=prefer_mprage)
        deduped_rows.append(best)

    deduped = pd.DataFrame(deduped_rows)
    after_dedup = len(deduped)
    stats["duplicate_visit_scans_deduplicated"] = before_dedup - after_dedup
    stats["subjects_with_at_least_one_scan"] = int(deduped["subject_id"].nunique())

    # ── Rename columns to training-ready schema ───────────────────────────────
    deduped = deduped.rename(columns={"image_visit": "visit_id", "image_date": "scan_date"})

    # Add placeholder cdr_score column — filled by prepare_adni_clinical.py
    deduped["cdr_score"] = float("nan")
    deduped["label"] = float("nan")  # will be filled after clinical alignment

    # Select and order columns
    keep_cols = [
        "image_id", "subject_id", "study_id", "series_id",
        "visit_id", "scan_date", "image_description",
        "nifti_path", "cdr_score", "label",
    ]
    for col in keep_cols:
        if col not in deduped.columns:
            deduped[col] = None

    deduped = deduped[keep_cols].reset_index(drop=True)

    return deduped, stats


def main() -> None:
    args = parse_args()
    nifti_dir     = Path(args.nifti_dir)
    loni_manifest = Path(args.loni_manifest)
    out_dir       = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Scan NIfTI dir ────────────────────────────────────────────────────────
    nifti_df = scan_nifti_dir(nifti_dir)

    # ── Load LONI manifest ────────────────────────────────────────────────────
    loni_df = load_loni_manifest(loni_manifest)

    # ── Build manifest ────────────────────────────────────────────────────────
    manifest_df, stats = build_manifest(nifti_df, loni_df, prefer_mprage=args.prefer_mprage)

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_path = out_dir / args.out_file
    if not manifest_df.empty:
        manifest_df.to_csv(out_path, index=False)
        logger.info("Imaging manifest written: %s (%d rows)", out_path, len(manifest_df))
    else:
        logger.warning("Empty manifest — not written to disk.")

    build_report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "nifti_dir": str(nifti_dir),
        "loni_manifest": str(loni_manifest),
        "output_manifest": str(out_path),
        "stats": stats,
        "note": (
            "cdr_score column is empty. Run prepare_adni_clinical.py to fill it. "
            "Do not start training until cdr_score is populated."
        ),
    }
    report_path = out_dir / "build_manifest_report.json"
    report_path.write_text(json.dumps(build_report, indent=2))

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ADNI Manifest Build Summary")
    logger.info("=" * 60)
    logger.info("  LONI manifest entries:       %d", stats["loni_entries"])
    logger.info("  NIfTI files found:           %d", stats["nifti_files_found"])
    logger.info("  Matched by image_id:         %d", stats["matched_by_image_id"])
    logger.info("  Unmatched NIfTI files:       %d", stats["unmatched_nifti"])
    logger.info("  LONI entries w/o NIfTI:      %d", stats["loni_entries_without_nifti"])
    logger.info("  Visits deduplicated:         %d", stats["duplicate_visit_scans_deduplicated"])
    logger.info("  Subjects in output:          %d", stats["subjects_with_at_least_one_scan"])
    logger.info("  Output manifest:             %s", out_path)
    logger.info("-" * 60)
    logger.info(
        "NEXT STEP: Run prepare_adni_clinical.py to add CDR labels.\n"
        "  python scripts/prepare_adni_clinical.py \\\n"
        "    --imaging-manifest %s \\\n"
        "    --clinical-csv <path/to/ADNI_clinical_export.csv> \\\n"
        "    --out-dir %s", out_path, out_dir
    )
    logger.info("=" * 60)

    if stats["nifti_files_found"] == 0:
        logger.error(
            "BLOCKED: No NIfTI files. Convert DICOM archive first with dcm2niix."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
