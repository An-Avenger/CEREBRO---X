#!/usr/bin/env python
"""
scripts/validate_adni_manifest.py
-----------------------------------
Validate the ADNI Clinical T1w Imaging Cohort manifest (CSV from LONI IDA)
against the actual extracted NIfTI files on disk.

The LONI manifest has these columns:
    image_id, subject_id, study_id, series_id, image_visit, image_date, image_description

This script:
  1. Reads the LONI manifest CSV.
  2. Scans the NIfTI output directory (produced by dcm2niix) for .nii/.nii.gz files.
  3. Attempts to match each manifest entry to an extracted NIfTI file.
  4. Reports: matched, missing, duplicates, unmatched NIfTI files.
  5. Outputs a cleaned manifest CSV with nifti_path column filled in.
  6. NEVER invents or infers paths — unmatched entries are marked as missing.

Usage:
    python scripts/validate_adni_manifest.py \\
        --manifest "C:/Users/HP/AppData/Local/Temp/.../Clinical_T1w_Imaging_Cohort_Manifest_15Oct2026.csv" \\
        --nifti-dir data/raw/mri/ADNI/nifti \\
        --out-dir   data/raw/mri/ADNI

NOTE: NIfTI files do not exist yet — run dcm2niix first (see instructions in
      the validation report). This script will report all entries as
      'missing_nifti' until NIfTI conversion is complete.

Outputs:
    data/raw/mri/ADNI/validation_report.json
    data/raw/mri/ADNI/manifest_validated.csv   — only matched entries
    data/raw/mri/ADNI/manifest_unmatched.csv   — entries without a NIfTI match
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("validate_adni_manifest")

# Expected columns in the LONI IDA manifest
LONI_REQUIRED_COLS = {"image_id", "subject_id", "image_visit", "image_date", "image_description"}

# Columns we add to produce a training-ready manifest
OUTPUT_COLS = [
    "image_id",
    "subject_id",
    "study_id",
    "series_id",
    "image_visit",
    "image_date",
    "image_description",
    "nifti_path",           # relative path from nifti_dir to matched .nii.gz
    "match_confidence",     # 'exact_image_id' | 'subject_visit_date' | 'subject_visit'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ADNI LONI manifest against extracted NIfTI files."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the LONI IDA manifest CSV (e.g. Clinical_T1w_Imaging_Cohort_Manifest_15Oct2026.csv).",
    )
    parser.add_argument(
        "--nifti-dir",
        required=True,
        help="Directory containing NIfTI files produced by dcm2niix.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/mri/ADNI",
        help="Directory to write cleaned manifest and report.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Scan nifti-dir recursively (default: True).",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load and validate the LONI manifest CSV."""
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        logger.error("Failed to read manifest: %e", e)
        sys.exit(1)

    logger.info("Manifest loaded: %d rows, columns: %s", len(df), list(df.columns))

    missing_cols = LONI_REQUIRED_COLS - set(df.columns)
    if missing_cols:
        logger.error(
            "Manifest missing required columns: %s. "
            "This does not look like a LONI IDA imaging manifest.",
            missing_cols,
        )
        sys.exit(1)

    # Validate no duplicate image_ids
    dup_mask = df["image_id"].duplicated(keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        logger.warning("Found %d rows with duplicate image_id values.", n_dup)

    return df


def scan_nifti_dir(nifti_dir: Path, recursive: bool) -> dict[str, Path]:
    """
    Scan nifti_dir for .nii and .nii.gz files.

    Returns a dict mapping filename stem → absolute path.
    dcm2niix typically names files like:
        {subject_id}_{image_id}_{series_desc}.nii.gz
    """
    if not nifti_dir.exists():
        logger.warning(
            "NIfTI directory does not exist: %s. "
            "This is expected if dcm2niix has not been run yet.",
            nifti_dir,
        )
        return {}

    pattern = "**/*.nii*" if recursive else "*.nii*"
    nifti_files = list(nifti_dir.glob(pattern))
    nifti_files = [f for f in nifti_files if f.suffix in (".nii", ".gz")]

    logger.info("Found %d NIfTI files in %s", len(nifti_files), nifti_dir)

    # Build lookup dict: filename stem (without .nii or .nii.gz) → path
    stem_to_path: dict[str, Path] = {}
    for f in nifti_files:
        stem = f.name.replace(".nii.gz", "").replace(".nii", "")
        if stem in stem_to_path:
            logger.warning("Duplicate NIfTI stem: %s (%s vs %s)", stem, stem_to_path[stem], f)
        else:
            stem_to_path[stem] = f

    return stem_to_path


def match_manifest_to_nifti(
    df: pd.DataFrame,
    stem_to_path: dict[str, Path],
    nifti_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Try to match each manifest row to a NIfTI file.

    Matching strategy (in order of confidence):
      1. Exact: any NIfTI filename stem contains the image_id (as string)
      2. Subject+visit+date: stem contains subject_id AND image_visit AND image_date
      3. Subject+visit: stem contains subject_id AND image_visit

    Returns:
        matched_df, unmatched_df, match_log (list of dicts)
    """
    matched_rows = []
    unmatched_rows = []
    match_log = []

    if not stem_to_path:
        logger.warning(
            "No NIfTI files found — all %d manifest entries will be reported as unmatched. "
            "Run dcm2niix to convert the DICOM archive first.",
            len(df),
        )
        for _, row in df.iterrows():
            unmatched_rows.append(row.to_dict())
            match_log.append({
                "image_id": row["image_id"],
                "subject_id": row["subject_id"],
                "status": "missing_nifti_dir",
                "reason": "NIfTI directory does not exist or is empty.",
            })
        return pd.DataFrame(), pd.DataFrame(unmatched_rows), match_log

    for _, row in df.iterrows():
        image_id_str  = str(row["image_id"])
        subject_id    = str(row["subject_id"])
        image_visit   = str(row.get("image_visit", "")).replace("/", "_")
        image_date    = str(row.get("image_date", "")).replace("-", "")

        matched_path = None
        confidence = None

        # Strategy 1: Exact image_id match
        for stem, path in stem_to_path.items():
            if image_id_str in stem:
                matched_path = path
                confidence = "exact_image_id"
                break

        # Strategy 2: Subject + visit + date
        if matched_path is None:
            for stem, path in stem_to_path.items():
                if subject_id in stem and image_visit in stem and image_date in stem:
                    matched_path = path
                    confidence = "subject_visit_date"
                    break

        # Strategy 3: Subject + visit only
        if matched_path is None:
            for stem, path in stem_to_path.items():
                if subject_id in stem and image_visit in stem:
                    matched_path = path
                    confidence = "subject_visit"
                    break

        if matched_path is not None:
            row_dict = row.to_dict()
            # Store relative path from nifti_dir
            try:
                rel_path = matched_path.relative_to(nifti_dir)
            except ValueError:
                rel_path = matched_path  # fallback: absolute
            row_dict["nifti_path"] = str(rel_path)
            row_dict["match_confidence"] = confidence
            matched_rows.append(row_dict)
            match_log.append({
                "image_id": row["image_id"],
                "subject_id": row["subject_id"],
                "status": "matched",
                "confidence": confidence,
                "nifti_path": str(rel_path),
            })
        else:
            unmatched_rows.append(row.to_dict())
            match_log.append({
                "image_id": row["image_id"],
                "subject_id": row["subject_id"],
                "status": "missing_nifti",
                "reason": "No NIfTI file found matching this manifest entry.",
            })

    matched_df = pd.DataFrame(matched_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)

    return matched_df, unmatched_df, match_log


def validate_nifti_integrity(matched_df: pd.DataFrame, nifti_dir: Path) -> list[dict]:
    """
    For matched NIfTI files, verify they can be opened by nibabel and have
    valid 3D dimensions. Returns a list of validation issues.
    """
    issues = []
    try:
        import nibabel as nib
    except ImportError:
        logger.warning("nibabel not installed — skipping NIfTI integrity checks.")
        return issues

    if matched_df.empty:
        return issues

    logger.info("Validating NIfTI integrity for %d matched files...", len(matched_df))
    for i, row in matched_df.iterrows():
        path = nifti_dir / row["nifti_path"]
        if not path.exists():
            issues.append({"image_id": row["image_id"], "issue": "FILE_MISSING", "path": str(path)})
            continue
        try:
            img = nib.load(str(path))
            shape = img.shape
            if len(shape) < 3:
                issues.append({
                    "image_id": row["image_id"],
                    "issue": "WRONG_DIMS",
                    "shape": list(shape),
                    "path": str(path),
                })
            elif any(d < 32 for d in shape[:3]):
                issues.append({
                    "image_id": row["image_id"],
                    "issue": "DIM_TOO_SMALL",
                    "shape": list(shape),
                    "path": str(path),
                })
        except Exception as e:
            issues.append({
                "image_id": row["image_id"],
                "issue": "UNREADABLE",
                "error": str(e),
                "path": str(path),
            })

    return issues


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    nifti_dir     = Path(args.nifti_dir)
    out_dir       = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load manifest ─────────────────────────────────────────────────────────
    df = load_manifest(manifest_path)

    # ── Scan NIfTI directory ──────────────────────────────────────────────────
    stem_to_path = scan_nifti_dir(nifti_dir, recursive=args.recursive)

    # ── Match manifest to NIfTI files ─────────────────────────────────────────
    matched_df, unmatched_df, match_log = match_manifest_to_nifti(df, stem_to_path, nifti_dir)

    # ── Validate NIfTI integrity ──────────────────────────────────────────────
    integrity_issues = validate_nifti_integrity(matched_df, nifti_dir)

    # ── Compute statistics ────────────────────────────────────────────────────
    n_total    = len(df)
    n_matched  = len(matched_df)
    n_missing  = len(unmatched_df)
    n_issues   = len(integrity_issues)

    unique_subjects_total   = df["subject_id"].nunique()
    unique_subjects_matched = matched_df["subject_id"].nunique() if not matched_df.empty else 0

    # ── Write outputs ─────────────────────────────────────────────────────────
    if not matched_df.empty:
        validated_path = out_dir / "manifest_validated.csv"
        matched_df.to_csv(validated_path, index=False)
        logger.info("Validated manifest written: %s (%d rows)", validated_path, len(matched_df))
    else:
        logger.warning("No matched entries — manifest_validated.csv not written.")

    if not unmatched_df.empty:
        unmatched_path = out_dir / "manifest_unmatched.csv"
        unmatched_df.to_csv(unmatched_path, index=False)
        logger.info("Unmatched entries written: %s (%d rows)", unmatched_path, len(unmatched_df))

    # ── Write validation report ───────────────────────────────────────────────
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "manifest_path": str(manifest_path),
        "nifti_dir": str(nifti_dir),
        "totals": {
            "manifest_entries": n_total,
            "matched": n_matched,
            "missing_nifti": n_missing,
            "integrity_issues": n_issues,
        },
        "subjects": {
            "total_in_manifest": unique_subjects_total,
            "with_matched_nifti": unique_subjects_matched,
        },
        "integrity_issues": integrity_issues,
        "match_log_sample": match_log[:20],  # first 20 for brevity
        "warnings": [],
        "next_steps": [],
    }

    if n_matched == 0 and n_missing > 0:
        msg = (
            f"All {n_missing} manifest entries are missing NIfTI files. "
            "You must convert the DICOM archive to NIfTI before training. "
            "See instructions below."
        )
        report["warnings"].append(msg)
        report["next_steps"].append({
            "step": 1,
            "action": "Install dcm2niix",
            "command": "pip install dcm2niix   # or download binary from https://github.com/rordenlab/dcm2niix/releases",
        })
        report["next_steps"].append({
            "step": 2,
            "action": "Extract DICOM archive to a working directory",
            "command": (
                'python -c "import zipfile; zipfile.ZipFile(\'Clinical_T1w_Imaging_Cohort_MRI_2.zip\').extractall(\'data/raw/mri/ADNI/dicom\')"'
            ),
            "note": "WARNING: 29.4 GB. Ensure ≥60 GB free disk space (DICOM + NIfTI).",
        })
        report["next_steps"].append({
            "step": 3,
            "action": "Convert DICOM to NIfTI (per subject)",
            "command": (
                "for subj_dir in data/raw/mri/ADNI/dicom/ADNI/*/; do\n"
                "  dcm2niix -o data/raw/mri/ADNI/nifti -f %i_%s_%d -z y -ba n \"$subj_dir\"\n"
                "done"
            ),
            "note": (
                "-f %i_%s_%d → filename = image_id_series_date. "
                "-z y → gzip NIfTI. "
                "-ba n → no BIDS anonymization (preserves ADNI IDs). "
                "Expected output: ~1803 .nii.gz files at ~15 GB total."
            ),
        })
        report["next_steps"].append({
            "step": 4,
            "action": "Re-run this validation script",
            "command": (
                f"python scripts/validate_adni_manifest.py "
                f"--manifest \"{manifest_path}\" "
                f"--nifti-dir data/raw/mri/ADNI/nifti "
                f"--out-dir data/raw/mri/ADNI"
            ),
        })

    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Validation report written: %s", report_path)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ADNI Manifest Validation Summary")
    logger.info("=" * 60)
    logger.info("  Manifest entries:         %d", n_total)
    logger.info("  Matched (has NIfTI):      %d (%.1f%%)", n_matched, 100 * n_matched / max(n_total, 1))
    logger.info("  Missing NIfTI:            %d (%.1f%%)", n_missing, 100 * n_missing / max(n_total, 1))
    logger.info("  NIfTI integrity issues:   %d", n_issues)
    logger.info("  Subjects in manifest:     %d", unique_subjects_total)
    logger.info("  Subjects with NIfTI:      %d", unique_subjects_matched)
    logger.info("-" * 60)

    if report["warnings"]:
        for w in report["warnings"]:
            logger.warning("  !! %s", w)

    if report["next_steps"]:
        logger.info("Next steps (see validation_report.json for full details):")
        for step in report["next_steps"]:
            logger.info("  Step %d: %s", step["step"], step["action"])
            logger.info("    > %s", step["command"])

    logger.info("=" * 60)

    # Exit non-zero if blocking issues present
    if n_matched == 0 and n_total > 0:
        logger.error("BLOCKED: No NIfTI files matched. Convert DICOM first.")
        sys.exit(2)


if __name__ == "__main__":
    main()
