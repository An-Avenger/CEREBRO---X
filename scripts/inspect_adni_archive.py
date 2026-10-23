#!/usr/bin/env python
"""
scripts/inspect_adni_archive.py
--------------------------------
Inspect the ADNI DICOM ZIP archive WITHOUT extracting it.

Reads the ZIP central directory to produce a structural inventory
of subjects, scan series, image IDs, dates, and file counts.
Outputs a machine-readable JSON report and a CSV summary.

The ADNI ZIP path structure is:
    ADNI/{subject_id}/{series_name}/{date_time}/{image_id}/ADNI_*.dcm

Usage:
    python scripts/inspect_adni_archive.py \\
        --archive "e:/PROJECTS/CEREBRO-X/Clinical_T1w_Imaging_Cohort_MRI_2.zip" \\
        --out-dir data/raw/mri/ADNI/inspection

Outputs:
    data/raw/mri/ADNI/inspection/archive_inventory.csv
    data/raw/mri/ADNI/inspection/archive_report.json
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import zipfile
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("inspect_adni_archive")

# ADNI path depth: ADNI / subject / series / datetime / image_id / file
ADNI_PREFIX = "ADNI"
IDX_SUBJECT  = 1
IDX_SERIES   = 2
IDX_DATETIME = 3
IDX_IMAGE_ID = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the ADNI DICOM ZIP archive without extracting it."
    )
    parser.add_argument(
        "--archive",
        required=True,
        type=str,
        help="Absolute path to the ADNI ZIP file (e.g. Clinical_T1w_Imaging_Cohort_MRI_2.zip).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/mri/ADNI/inspection",
        type=str,
        help="Directory to write reports to (created if missing).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of sample file paths to include per subject in the report.",
    )
    return parser.parse_args()


def normalize_path(raw: str) -> str:
    """Normalize ZIP entry paths to forward-slash form."""
    return raw.replace("\\", "/")


def inspect_archive(zip_path: Path, sample_size: int) -> dict:
    """
    Read the ZIP central directory and produce a structural inventory.

    Returns a dict with:
        total_files, subjects (dict), series (Counter), extensions (Counter),
        image_ids (set), sample_paths (list)
    """
    logger.info("Opening ZIP archive: %s", zip_path)
    logger.info("Archive size: %.2f GB", zip_path.stat().st_size / 1e9)

    subjects: dict[str, dict] = {}   # subject_id -> {series: Counter, images: set, dates: set}
    series_counter: collections.Counter = collections.Counter()
    ext_counter: collections.Counter = collections.Counter()
    image_ids: set[str] = set()
    total_files = 0
    parse_errors = 0
    sample_paths: list[str] = []

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        all_names = zf.namelist()
        total_entries = len(all_names)
        logger.info("Total ZIP entries: %d", total_entries)

        for raw_name in all_names:
            name = normalize_path(raw_name)
            parts = name.rstrip("/").split("/")

            # Skip directories
            if name.endswith("/"):
                continue

            total_files += 1

            # Extension counting
            fname = parts[-1]
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "no_ext"
            ext_counter[ext] += 1

            # Expect: ADNI / subject / series / datetime / image_id / file
            if len(parts) < 6 or parts[0] != ADNI_PREFIX:
                parse_errors += 1
                continue

            subject_id  = parts[IDX_SUBJECT]
            series_name = parts[IDX_SERIES]
            date_str    = parts[IDX_DATETIME]
            image_id    = parts[IDX_IMAGE_ID]

            # Accumulate subject record
            if subject_id not in subjects:
                subjects[subject_id] = {
                    "series": collections.Counter(),
                    "image_ids": set(),
                    "dates": set(),
                    "file_count": 0,
                    "sample_paths": [],
                }

            s = subjects[subject_id]
            s["series"][series_name] += 1
            s["image_ids"].add(image_id)
            s["dates"].add(date_str)
            s["file_count"] += 1

            if len(s["sample_paths"]) < sample_size:
                s["sample_paths"].append(name)

            series_counter[series_name] += 1
            image_ids.add(image_id)

        if total_files % 10000 == 0:
            logger.info("  Processed %d files...", total_files)

    logger.info(
        "Scan complete: %d files | %d subjects | %d unique series | %d image IDs | %d parse_errors",
        total_files, len(subjects), len(series_counter), len(image_ids), parse_errors,
    )

    return {
        "archive_path": str(zip_path),
        "archive_size_gb": round(zip_path.stat().st_size / 1e9, 3),
        "total_entries": total_entries,
        "total_files": total_files,
        "parse_errors": parse_errors,
        "subjects": subjects,
        "series_counter": dict(series_counter),
        "ext_counter": dict(ext_counter),
        "unique_image_ids": len(image_ids),
    }


def build_csv(subjects: dict) -> list[dict]:
    """Convert subject dict to flat rows for CSV export."""
    rows = []
    for subj_id, info in subjects.items():
        for series_name, file_count in info["series"].items():
            for image_id in sorted(info["image_ids"]):
                # We can only associate a series with image_ids approximately
                # (all image_ids under a subject are listed)
                break  # just one row per series for summary
            rows.append({
                "subject_id": subj_id,
                "series_name": series_name,
                "dicom_file_count": file_count,
                "unique_image_ids": len(info["image_ids"]),
                "scan_dates": "|".join(sorted(info["dates"])),
                "sample_path": info["sample_paths"][0] if info["sample_paths"] else "",
            })
    return rows


def main() -> None:
    args = parse_args()
    zip_path = Path(args.archive)
    out_dir  = Path(args.out_dir)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not zip_path.exists():
        logger.error("Archive not found: %s", zip_path)
        sys.exit(1)

    if not zipfile.is_zipfile(str(zip_path)):
        logger.error("Not a valid ZIP file: %s", zip_path)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run inspection ────────────────────────────────────────────────────────
    inventory = inspect_archive(zip_path, sample_size=args.sample_size)

    # ── Serialise subjects dict (sets → lists for JSON) ───────────────────────
    subjects_serializable = {}
    for subj, info in inventory["subjects"].items():
        subjects_serializable[subj] = {
            "series": dict(info["series"]),
            "image_ids": sorted(info["image_ids"]),
            "dates": sorted(info["dates"]),
            "file_count": info["file_count"],
            "sample_paths": info["sample_paths"],
        }

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "archive_path": inventory["archive_path"],
        "archive_size_gb": inventory["archive_size_gb"],
        "total_entries": inventory["total_entries"],
        "total_files": inventory["total_files"],
        "parse_errors": inventory["parse_errors"],
        "unique_subjects": len(inventory["subjects"]),
        "unique_series_types": len(inventory["series_counter"]),
        "unique_image_ids": inventory["unique_image_ids"],
        "file_type_counts": inventory["ext_counter"],
        "series_type_counts": inventory["series_counter"],
        "subjects": subjects_serializable,
        "warnings": [],
    }

    # ── Warnings ──────────────────────────────────────────────────────────────
    ext_counter = inventory["ext_counter"]
    nifti_count = ext_counter.get("nii", 0) + ext_counter.get("gz", 0)
    dcm_count   = ext_counter.get("dcm", 0)

    if nifti_count == 0 and dcm_count > 0:
        msg = (
            f"Archive contains {dcm_count} DICOM (.dcm) files and ZERO NIfTI files. "
            "You must convert DICOM series to NIfTI using dcm2niix before training. "
            "Command: dcm2niix -o <output_dir> -f %i_%s_%d -z y <dicom_series_dir>"
        )
        report["warnings"].append(msg)
        logger.warning(msg)

    if inventory["parse_errors"] > 0:
        msg = f"{inventory['parse_errors']} entries could not be parsed into the expected path structure."
        report["warnings"].append(msg)
        logger.warning(msg)

    # ── Write JSON report ─────────────────────────────────────────────────────
    json_path = out_dir / "archive_report.json"
    json_path.write_text(json.dumps(report, indent=2))
    logger.info("JSON report written to: %s", json_path)

    # ── Write CSV summary ─────────────────────────────────────────────────────
    try:
        import pandas as pd
        rows = build_csv(inventory["subjects"])
        csv_path = out_dir / "archive_inventory.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info("CSV inventory written to: %s", csv_path)
    except ImportError:
        logger.warning("pandas not available — skipping CSV export.")

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ADNI Archive Inspection Summary")
    logger.info("=" * 60)
    logger.info("  Archive:           %s", inventory["archive_path"])
    logger.info("  Size:              %.2f GB", inventory["archive_size_gb"])
    logger.info("  Total DICOM files: %d", dcm_count)
    logger.info("  NIfTI files:       %d", nifti_count)
    logger.info("  Unique subjects:   %d", len(inventory["subjects"]))
    logger.info("  Unique image IDs:  %d", inventory["unique_image_ids"])
    logger.info("  Series types:      %d", len(inventory["series_counter"]))
    logger.info("-" * 60)
    logger.info("Top 10 series types by file count:")
    for series, count in sorted(
        inventory["series_counter"].items(), key=lambda x: -x[1]
    )[:10]:
        logger.info("    %-50s  %d files", series, count)

    if report["warnings"]:
        logger.info("-" * 60)
        logger.warning("WARNINGS:")
        for w in report["warnings"]:
            logger.warning("  !! %s", w)

    logger.info("=" * 60)
    logger.info("Next step: Convert DICOM to NIfTI, then run:")
    logger.info("  python scripts/build_adni_manifest.py --nifti-dir <output_dir> ...")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
