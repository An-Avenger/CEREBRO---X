#!/usr/bin/env python
"""
scripts/prepare_adni_clinical.py
----------------------------------
Align ADNI clinical CDR data to the imaging manifest produced by
build_adni_manifest.py.

The ADNI CDR clinical export (typically CDR.csv from LONI) has columns like:
    RID, VISCODE, VISCODE2, EXAMDATE, CDGLOBAL, CDMEMORY, CDORIENT, ...

This script:
  1. Loads the imaging manifest (with nifti_path filled).
  2. Loads the ADNI clinical CDR export.
  3. Matches imaging visits to CDR assessments by subject_id + visit code.
  4. Reports time gaps, unmatched visits, missing CDR values.
  5. Outputs a merged manifest with cdr_score and label columns.
  6. Does NOT invent CDR scores or infer missing labels.
  7. Treats unsupported CDR values as invalid (not silently mapped to 0).

CDR Scoring:
    0.0  → Normal            → class 0
    0.5  → Very Mild / MCI   → class 1
    1.0  → Mild AD           → class 2
    2.0  → Moderate AD       → class 3
    3.0  → Severe AD         → class 3 (merged with Moderate for small cohorts)
    Any other value → INVALID → excluded from training

ADNI visit code alignment:
    The imaging manifest uses LONI image_visit codes (e.g. 'sc', 'm06', 'm12').
    The CDR export uses VISCODE (bl, m06, m12, ...) or VISCODE2 (sc=bl).
    This script normalises both to a canonical form before matching.

NOTE: If you do not have the CDR export, this script will tell you exactly
      which file to download from ADNI.

Usage:
    python scripts/prepare_adni_clinical.py \\
        --imaging-manifest data/raw/mri/ADNI/manifest_imaging.csv \\
        --clinical-csv     data/raw/clinical/CDR.csv \\
        --out-dir          data/raw/mri/ADNI

Outputs:
    data/raw/mri/ADNI/manifest_with_labels.csv     — matched entries with cdr_score
    data/raw/mri/ADNI/manifest_unmatched_labels.csv — entries without CDR match
    data/raw/mri/ADNI/label_alignment_report.json  — full alignment report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("prepare_adni_clinical")

# Valid CDR values and their class mapping
CDR_CLASS_MAP = {
    0.0: 0,   # Normal
    0.5: 1,   # Very Mild / MCI
    1.0: 2,   # Mild AD
    2.0: 3,   # Moderate AD
    3.0: 3,   # Severe AD (merged into class 3 — document this choice)
}
CDR_CLASS_LABELS = {
    0: "Normal",
    1: "Very_Mild_MCI",
    2: "Mild_AD",
    3: "Moderate_Severe_AD",
}

# Maximum acceptable gap (days) between imaging date and CDR assessment date
MAX_DAYS_GAP = 180  # 6 months — document this choice

# VISCODE normalisation: map LONI visit codes to canonical form
def normalise_visit(code: str) -> str:
    """Normalise ADNI visit codes to lowercase form for matching."""
    if pd.isna(code):
        return ""
    code = str(code).strip().lower()
    # sc = screening = bl in many ADNI tables
    if code in ("sc", "scmri", "f"):
        return "bl"
    return code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align ADNI CDR clinical data to imaging manifest."
    )
    parser.add_argument(
        "--imaging-manifest",
        required=True,
        help="Path to manifest_imaging.csv (output of build_adni_manifest.py).",
    )
    parser.add_argument(
        "--clinical-csv",
        required=False,
        default=None,
        help=(
            "Path to ADNI CDR clinical export CSV (CDR.csv from LONI). "
            "If not provided, the script exits with instructions on where to download it."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/mri/ADNI",
        help="Output directory for merged manifest and report.",
    )
    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=MAX_DAYS_GAP,
        help=(
            f"Maximum days between imaging date and CDR assessment date to "
            f"consider a match valid (default: {MAX_DAYS_GAP})."
        ),
    )
    return parser.parse_args()


def check_clinical_file_instructions(imaging_manifest_path: Path) -> None:
    """Print instructions for obtaining the CDR clinical file and exit."""
    logger.error(
        "Clinical CDR export file not provided (--clinical-csv is required)."
    )
    logger.info("")
    logger.info("=" * 70)
    logger.info("HOW TO OBTAIN THE ADNI CDR CLINICAL DATA")
    logger.info("=" * 70)
    logger.info("1. Log in to the LONI IDA portal: https://ida.loni.usc.edu/")
    logger.info("2. Go to: Search > Data Collections > ADNI")
    logger.info("3. Navigate to: Download > Study Data")
    logger.info("4. Select: 'Clinical and Cognitive Assessments'")
    logger.info("5. Download: 'Clinical Dementia Rating (CDR)' → CDR.csv")
    logger.info("   The file is typically named: CDR_[date].csv or CDR.csv")
    logger.info("")
    logger.info("Key columns you need:")
    logger.info("  RID      — Subject ID (maps to subject_id in imaging manifest)")
    logger.info("  VISCODE  — Visit code (bl, m06, m12, ...)")
    logger.info("  CDGLOBAL — Global CDR score (0, 0.5, 1, 2, 3)")
    logger.info("  EXAMDATE — Assessment date (YYYY-MM-DD)")
    logger.info("")
    logger.info(
        "Once downloaded, run:\n"
        f"  python scripts/prepare_adni_clinical.py \\\n"
        f"    --imaging-manifest {imaging_manifest_path} \\\n"
        f"    --clinical-csv data/raw/clinical/CDR.csv \\\n"
        f"    --out-dir data/raw/mri/ADNI"
    )
    logger.info("=" * 70)
    sys.exit(3)


def load_clinical_data(clinical_csv: Path) -> pd.DataFrame:
    """Load and validate the ADNI CDR CSV."""
    if not clinical_csv.exists():
        logger.error("Clinical CSV not found: %s", clinical_csv)
        sys.exit(1)

    df = pd.read_csv(clinical_csv)
    logger.info("Clinical CDR file: %d rows, columns: %s", len(df), list(df.columns))

    # Normalize column names to uppercase for robust matching
    df.columns = [c.strip().upper() for c in df.columns]

    # Map common alternative column names
    col_aliases = {
        "PTID": "RID",       # PTID is sometimes used instead of RID
        "SUBJECT": "RID",
        "VISCODEM": "VISCODE",
        "CDR": "CDGLOBAL",
        "GLOBAL_CDR": "CDGLOBAL",
    }
    for alias, canonical in col_aliases.items():
        if alias in df.columns and canonical not in df.columns:
            df = df.rename(columns={alias: canonical})
            logger.info("Renamed column %s → %s", alias, canonical)

    required = {"RID", "VISCODE", "CDGLOBAL"}
    missing = required - set(df.columns)
    if missing:
        logger.error(
            "Clinical CSV missing required columns: %s\n"
            "Found columns: %s\n"
            "Expected at minimum: RID (subject ID), VISCODE (visit code), CDGLOBAL (CDR score).",
            missing, list(df.columns),
        )
        logger.info(
            "If the file uses different column names, check the ADNI data dictionary "
            "at https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/."
        )
        sys.exit(1)

    # Normalise RID to match subject_id format used in imaging manifest
    # ADNI subject_id format: '037_S_1421', RID is '1421' (last 4 digits)
    # We keep RID as-is and will attempt several matching strategies
    df["RID_STR"] = df["RID"].astype(str).str.strip()

    # Parse EXAMDATE if available
    if "EXAMDATE" in df.columns:
        df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

    # Normalise visit codes
    df["VISCODE_NORM"] = df["VISCODE"].apply(normalise_visit)

    # Remove rows with missing CDGLOBAL
    before = len(df)
    df = df[df["CDGLOBAL"].notna()].copy()
    after = len(df)
    if before > after:
        logger.warning("Dropped %d rows with missing CDGLOBAL.", before - after)

    # Convert CDGLOBAL to float
    df["CDGLOBAL"] = pd.to_numeric(df["CDGLOBAL"], errors="coerce")
    invalid_mask = df["CDGLOBAL"].isna()
    if invalid_mask.any():
        logger.warning(
            "Dropped %d rows with non-numeric CDGLOBAL values.", invalid_mask.sum()
        )
        df = df[~invalid_mask].copy()

    logger.info("Clinical data after cleaning: %d rows", len(df))
    return df


def match_to_imaging(
    imaging_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    max_gap_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Match each imaging row to a CDR assessment.

    Matching rules (documented):
    1. Primary: subject_id + normalised visit code (exact visit match).
       - ADNI subject_id format: 037_S_1421 → RID = 1421 (last 4 chars after last '_')
    2. Fallback: subject_id + nearest assessment within max_gap_days of scan date.

    We never silently match across unrelated visits. If no match is found within
    max_gap_days and visit code doesn't match, the entry is flagged as unmatched.
    """
    # Extract RID from subject_id: '037_S_1421' → '1421'
    imaging_df = imaging_df.copy()
    imaging_df["RID_STR"] = imaging_df["subject_id"].str.split("_").str[-1]
    imaging_df["scan_date_parsed"] = pd.to_datetime(imaging_df["scan_date"], errors="coerce")
    imaging_df["visit_norm"] = imaging_df["visit_id"].apply(normalise_visit)

    # Build clinical lookup: RID → list of (viscode_norm, examdate, cdglobal)
    clinical_by_rid: dict[str, list[dict]] = {}
    for _, row in clinical_df.iterrows():
        rid = row["RID_STR"]
        if rid not in clinical_by_rid:
            clinical_by_rid[rid] = []
        clinical_by_rid[rid].append({
            "viscode_norm": row["VISCODE_NORM"],
            "examdate": row.get("EXAMDATE"),
            "cdglobal": row["CDGLOBAL"],
        })

    matched_rows = []
    unmatched_rows = []
    match_log = []

    for _, row in imaging_df.iterrows():
        rid = row["RID_STR"]
        visit = row["visit_norm"]
        scan_date = row["scan_date_parsed"]
        image_id = row.get("image_id", "")

        cdr_value: Optional[float] = None
        match_type: str = "unmatched"
        gap_days: Optional[int] = None
        matched_viscode: Optional[str] = None
        matched_examdate: Optional[str] = None

        assessments = clinical_by_rid.get(rid, [])

        if not assessments:
            match_log.append({
                "image_id": image_id,
                "subject_id": row["subject_id"],
                "visit_id": row["visit_id"],
                "status": "no_clinical_record",
                "reason": f"RID {rid} not found in clinical data.",
            })
            unmatched_rows.append(row.to_dict())
            continue

        # Strategy 1: Exact visit code match
        exact_matches = [a for a in assessments if a["viscode_norm"] == visit]
        if exact_matches:
            best = exact_matches[0]  # take first (should be unique)
            cdr_value = best["cdglobal"]
            match_type = "exact_visit_code"
            matched_viscode = best["viscode_norm"]
            matched_examdate = str(best.get("examdate", ""))
            if pd.notna(best.get("examdate")) and pd.notna(scan_date):
                gap_days = abs((best["examdate"] - scan_date).days)

        # Strategy 2: Nearest assessment within max_gap_days
        elif pd.notna(scan_date) and assessments:
            candidates = []
            for a in assessments:
                ed = a.get("examdate")
                if pd.notna(ed):
                    diff = abs((ed - scan_date).days)
                    if diff <= max_gap_days:
                        candidates.append((diff, a))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_gap, best = candidates[0]
                cdr_value = best["cdglobal"]
                match_type = f"nearest_within_{max_gap_days}d"
                matched_viscode = best["viscode_norm"]
                matched_examdate = str(best.get("examdate", ""))
                gap_days = best_gap

        if cdr_value is not None:
            # Validate CDR value
            if cdr_value not in CDR_CLASS_MAP:
                match_log.append({
                    "image_id": image_id,
                    "subject_id": row["subject_id"],
                    "visit_id": row["visit_id"],
                    "status": "invalid_cdr_value",
                    "cdr_value": cdr_value,
                    "reason": (
                        f"CDR value {cdr_value} is not in the valid set "
                        f"{sorted(CDR_CLASS_MAP.keys())}. Excluding from training."
                    ),
                })
                unmatched_rows.append(row.to_dict())
                continue

            row_dict = row.to_dict()
            row_dict["cdr_score"] = cdr_value
            row_dict["label"] = CDR_CLASS_MAP[cdr_value]
            row_dict["match_type"] = match_type
            row_dict["cdr_examdate"] = matched_examdate
            row_dict["days_gap"] = gap_days
            matched_rows.append(row_dict)
            match_log.append({
                "image_id": image_id,
                "subject_id": row["subject_id"],
                "visit_id": row["visit_id"],
                "status": "matched",
                "match_type": match_type,
                "cdr_score": cdr_value,
                "label": CDR_CLASS_MAP[cdr_value],
                "days_gap": gap_days,
                "cdr_viscode": matched_viscode,
            })
        else:
            match_log.append({
                "image_id": image_id,
                "subject_id": row["subject_id"],
                "visit_id": row["visit_id"],
                "status": "no_match_within_window",
                "reason": (
                    f"No CDR assessment found with visit code '{visit}' "
                    f"or within {max_gap_days} days of scan date {scan_date}."
                ),
            })
            unmatched_rows.append(row.to_dict())

    return pd.DataFrame(matched_rows), pd.DataFrame(unmatched_rows), match_log


def main() -> None:
    args = parse_args()
    imaging_manifest = Path(args.imaging_manifest)
    out_dir          = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Check clinical file ───────────────────────────────────────────────────
    if args.clinical_csv is None:
        check_clinical_file_instructions(imaging_manifest)
        return  # unreachable — exits above

    clinical_csv = Path(args.clinical_csv)
    if not clinical_csv.exists():
        logger.error("Clinical CSV not found: %s", clinical_csv)
        check_clinical_file_instructions(imaging_manifest)
        return

    # ── Load imaging manifest ─────────────────────────────────────────────────
    if not imaging_manifest.exists():
        logger.error("Imaging manifest not found: %s", imaging_manifest)
        sys.exit(1)

    imaging_df = pd.read_csv(imaging_manifest)
    logger.info("Imaging manifest: %d rows, %d subjects", len(imaging_df), imaging_df["subject_id"].nunique())

    # ── Load clinical data ────────────────────────────────────────────────────
    clinical_df = load_clinical_data(clinical_csv)

    # ── Match ─────────────────────────────────────────────────────────────────
    matched_df, unmatched_df, match_log = match_to_imaging(
        imaging_df, clinical_df, max_gap_days=args.max_gap_days
    )

    # ── Statistics ────────────────────────────────────────────────────────────
    n_total    = len(imaging_df)
    n_matched  = len(matched_df)
    n_unmatched = len(unmatched_df)

    cdr_dist = {}
    label_dist = {}
    if not matched_df.empty:
        for cdr_val, cnt in matched_df["cdr_score"].value_counts().items():
            cdr_dist[str(cdr_val)] = int(cnt)
        for lbl, cnt in matched_df["label"].value_counts().items():
            label_dist[str(int(lbl))] = {
                "count": int(cnt),
                "label_name": CDR_CLASS_LABELS.get(int(lbl), "Unknown"),
            }

    # ── Write outputs ─────────────────────────────────────────────────────────
    if not matched_df.empty:
        out_path = out_dir / "manifest_with_labels.csv"
        matched_df.to_csv(out_path, index=False)
        logger.info("Labelled manifest written: %s (%d rows)", out_path, n_matched)

    if not unmatched_df.empty:
        unmatched_path = out_dir / "manifest_unmatched_labels.csv"
        unmatched_df.to_csv(unmatched_path, index=False)
        logger.info("Unmatched entries: %s (%d rows)", unmatched_path, n_unmatched)

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "imaging_manifest": str(imaging_manifest),
        "clinical_csv": str(clinical_csv),
        "max_gap_days": args.max_gap_days,
        "matching_rules": {
            "primary": "Exact visit code match (subject_id + normalised VISCODE)",
            "fallback": f"Nearest CDR assessment within {args.max_gap_days} days of scan date",
            "invalid_cdr": f"CDR values not in {sorted(CDR_CLASS_MAP.keys())} are excluded",
            "no_match": "Entries with no valid CDR match are placed in manifest_unmatched_labels.csv",
        },
        "cdr_class_mapping": {str(k): v for k, v in CDR_CLASS_MAP.items()},
        "cdr_class_labels": CDR_CLASS_LABELS,
        "stats": {
            "total_imaging_entries": n_total,
            "matched_with_label": n_matched,
            "unmatched": n_unmatched,
            "match_rate_pct": round(100 * n_matched / max(n_total, 1), 1),
        },
        "cdr_score_distribution": cdr_dist,
        "label_distribution": label_dist,
        "limitations": [
            "CDR is a global clinical rating, not an imaging biomarker.",
            "CDR scores matched within 6 months of scan date may reflect disease state "
            "at a slightly different timepoint.",
            "This dataset alone cannot validate clinical progression claims.",
            "CDR class 3 includes both CDR=2.0 and CDR=3.0 (merged for small cohort). "
            "See cdr_class_mapping for details.",
        ],
        "match_log_sample": match_log[:30],
    }

    report_path = out_dir / "label_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Label alignment report written: %s", report_path)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ADNI Clinical Label Alignment Summary")
    logger.info("=" * 60)
    logger.info("  Imaging entries:           %d", n_total)
    logger.info("  Matched with CDR label:    %d (%.1f%%)", n_matched, 100 * n_matched / max(n_total, 1))
    logger.info("  Unmatched / no label:      %d", n_unmatched)
    logger.info("  CDR distribution:")
    for cdr_val, cnt in sorted(cdr_dist.items()):
        logger.info("    CDR %s: %d scans", cdr_val, cnt)
    logger.info("  Label distribution:")
    for lbl, info in sorted(label_dist.items()):
        logger.info("    Class %s (%s): %d scans", lbl, info["label_name"], info["count"])

    if n_matched < 20:
        logger.warning(
            "Only %d labelled scans — too few for reliable model training and evaluation. "
            "Minimum recommended: 50 per class for classification.",
            n_matched,
        )

    logger.info("-" * 60)
    logger.info(
        "NEXT STEP: Run the training smoke test:\n"
        "  python scripts/gpu_smoke_test.py \\\n"
        "    --manifest data/raw/mri/ADNI/manifest_with_labels.csv \\\n"
        "    --nifti-dir data/raw/mri/ADNI/nifti"
    )
    logger.info("=" * 60)

    if n_matched == 0:
        logger.error("No labels aligned. Training cannot proceed.")
        sys.exit(4)


if __name__ == "__main__":
    main()
