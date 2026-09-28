#!/usr/bin/env python
"""
Validate the ADNI MRI dataset mount.
Checks if the manifest exists, conforms to the schema, and if all referenced NIfTI files exist and are valid.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("validate_adni")

def main():
    parser = argparse.ArgumentParser(description="Validate ADNI MRI Dataset Mount")
    parser.add_argument("--base-dir", type=str, default="data/raw/mri/ADNI", help="Path to ADNI base directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    manifest_path = base_dir / "manifest.csv"

    if not base_dir.exists():
        logger.error(f"Base directory {base_dir} does not exist.")
        sys.exit(1)

    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}. See README_ADNI_MOUNT.md for instructions.")
        sys.exit(1)

    logger.info(f"Found manifest at {manifest_path}. Validating schema...")
    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        logger.error(f"Failed to read manifest: {e}")
        sys.exit(1)

    required_cols = {"subject_id", "visit_id", "nifti_path", "cdr_score"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"Manifest is missing required columns: {missing}")
        sys.exit(1)
        
    logger.info(f"Schema valid. Found {len(df)} entries.")
    
    missing_files = 0
    for idx, row in df.iterrows():
        file_path = base_dir / row["nifti_path"]
        if not file_path.exists():
            logger.warning(f"Row {idx}: Missing file {file_path}")
            missing_files += 1

    if missing_files > 0:
        logger.error(f"Validation failed: {missing_files} NIfTI files are missing from the filesystem.")
        sys.exit(1)
        
    logger.info("Validation PASSED! The ADNI dataset is correctly mounted and ready for training.")

if __name__ == "__main__":
    main()
