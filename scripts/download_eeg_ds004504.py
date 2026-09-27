#!/usr/bin/env python
"""
scripts/download_eeg_ds004504.py
---------------------------------
Download OpenNeuro ds004504 EEG dataset (Miltiadous et al., 2023).

Dataset: "A Dataset of Scalp EEG Recordings of Alzheimer's Disease,
          Frontotemporal Dementia and Healthy Subjects"
Source:  https://openneuro.org/datasets/ds004504
Access:  Public S3 bucket — no login required.

Subjects: 88 (36 AD, 23 FTD, 29 Controls)
Format:   EEGLAB .set files, 19-channel resting-state EEG, eyes-closed.

This script:
    1. Downloads participants.tsv (metadata)
    2. Downloads all .set EEG files to data/raw/eeg/
    3. Saves a download manifest

Usage:
    python scripts/download_eeg_ds004504.py [--max-subjects N]

Note: Total dataset is ~2GB. Expect 10-30 minutes depending on connection.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("download_eeg_ds004504")

BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds004504"
OUTPUT_DIR = Path("data/raw/eeg")
N_TOTAL_SUBJECTS = 88  # Known from dataset description


def download_file(url: str, dest: Path, overwrite: bool = False) -> bool:
    """Download a single file. Returns True if successful."""
    if dest.exists() and not overwrite:
        logger.info("Already exists: %s", dest.name)
        return True
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1e6
        logger.info("Downloaded: %s (%.1f MB)", dest.name, size_mb)
        return True
    except Exception as e:
        logger.warning("FAILED: %s -> %s", url, e)
        if dest.exists():
            dest.unlink()
        return False


def get_subject_ids(n_max: int = N_TOTAL_SUBJECTS) -> list[str]:
    return [f"sub-{i:03d}" for i in range(1, min(n_max, N_TOTAL_SUBJECTS) + 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-subjects", type=int, default=N_TOTAL_SUBJECTS,
                        help="Maximum number of subjects to download (default: all 88)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download metadata files
    logger.info("Downloading metadata files...")
    for fname in ["participants.tsv", "dataset_description.json"]:
        download_file(f"{BASE_URL}/{fname}", OUTPUT_DIR / fname, overwrite=args.overwrite)

    # 2. Download EEG files for each subject
    subject_ids = get_subject_ids(args.max_subjects)
    logger.info("Downloading EEG data for %d subjects...", len(subject_ids))

    download_tasks = []
    for sid in subject_ids:
        # Main .set file
        set_url = f"{BASE_URL}/{sid}/eeg/{sid}_task-eyesclosed_eeg.set"
        set_dest = OUTPUT_DIR / sid / f"{sid}_task-eyesclosed_eeg.set"
        download_tasks.append((set_url, set_dest))

        # Channels TSV (small metadata)
        ch_url = f"{BASE_URL}/{sid}/eeg/{sid}_task-eyesclosed_channels.tsv"
        ch_dest = OUTPUT_DIR / sid / f"{sid}_task-eyesclosed_channels.tsv"
        download_tasks.append((ch_url, ch_dest))

    results = {"success": [], "failed": []}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_file, url, dest, args.overwrite): (url, dest)
            for url, dest in download_tasks
        }
        for future in as_completed(futures):
            url, dest = futures[future]
            ok = future.result()
            if ok:
                results["success"].append(str(dest))
            else:
                results["failed"].append(url)

    # Save manifest
    manifest = {
        "dataset": "ds004504",
        "source": "https://openneuro.org/datasets/ds004504",
        "citation": (
            "Miltiadous A, et al. (2023). A Dataset of Scalp EEG Recordings of "
            "Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects. "
            "Data, 8(6), 95."
        ),
        "subjects_attempted": len(subject_ids),
        "files_downloaded": len(results["success"]),
        "files_failed": len(results["failed"]),
        "failed_urls": results["failed"],
    }

    manifest_path = OUTPUT_DIR / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("Subjects:   %d", len(subject_ids))
    logger.info("Downloaded: %d files", len(results["success"]))
    logger.info("Failed:     %d files", len(results["failed"]))
    logger.info("Manifest:   %s", manifest_path)
    logger.info("=" * 60)

    if results["failed"]:
        logger.warning("Some files failed to download. Run with --overwrite to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
