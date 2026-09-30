import os
import glob
import json
import subprocess
import pandas as pd
from pathlib import Path

# Paths
DATA_ROOT = Path("D:/CEREBRO_DATA/OASIS2")
ARCHIVE_DIR = DATA_ROOT / "archives"
RAW_DIR = DATA_ROOT / "raw"
MANIFEST_DIR = DATA_ROOT / "manifests"
LOG_DIR = DATA_ROOT / "logs"

CLINICAL_CSV = Path("data/raw/oasis_longitudinal.csv")

def extract_archives():
    print("Extracting archives to RAW_DIR...")
    for part in ["OAS2_RAW_PART1.tar.gz", "OAS2_RAW_PART2.tar.gz"]:
        archive_path = ARCHIVE_DIR / part
        if archive_path.exists():
            print(f"Extracting {part}...")
            subprocess.run(["tar", "-xf", str(archive_path), "-C", str(RAW_DIR)], check=True)
            print(f"Extraction of {part} complete.")

def scan_and_manifest():
    print("Scanning raw directory and building manifest...")
    nifti_files = list(RAW_DIR.rglob("*.img")) + list(RAW_DIR.rglob("*.hdr")) + list(RAW_DIR.rglob("*.nii")) + list(RAW_DIR.rglob("*.nii.gz"))
    
    total_files = sum(1 for _ in RAW_DIR.rglob("*"))
    subjects = set()
    visits = set()
    
    manifest_data = []
    
    for nf in nifti_files:
        # e.g., OAS2_0001_MR1
        parts = nf.name.split('_')
        subject_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "Unknown"
        visit_id = f"{parts[0]}_{parts[1]}_{parts[2].split('.')[0]}" if len(parts) >= 3 else "Unknown"
        
        subjects.add(subject_id)
        visits.add(visit_id)
        
        manifest_data.append({
            "subject_id": subject_id,
            "visit_id": visit_id,
            "nifti_path": str(nf.resolve()),
            "filename": nf.name,
            "extension": "".join(nf.suffixes),
            "shape": "", # TBD via nibabel if needed, keeping fast for now
            "voxel_spacing": ""
        })
        
    df = pd.DataFrame(manifest_data)
    manifest_path = MANIFEST_DIR / "oasis2_mri_manifest.csv"
    df.to_csv(manifest_path, index=False)
    
    log = {
        "total_extracted_files": total_files,
        "total_mri_files": len(nifti_files),
        "total_subjects": len(subjects),
        "total_visits": len(visits)
    }
    with open(LOG_DIR / "extraction_log.json", "w") as f:
        json.dump(log, f, indent=4)
        
    return df

def clinical_alignment(mri_manifest):
    print("Aligning with clinical data...")
    if not CLINICAL_CSV.exists():
        print("Clinical CSV not found!")
        return
        
    clin_df = pd.read_csv(CLINICAL_CSV)
    
    # Merge on Subject ID and MRI ID (Visit ID)
    aligned_df = pd.merge(
        mri_manifest, 
        clin_df, 
        left_on="visit_id", 
        right_on="MRI ID", 
        how="inner"
    )
    
    # Calculate next visit CDR
    aligned_df = aligned_df.sort_values(by=["subject_id", "Visit"])
    aligned_df["next_CDR"] = aligned_df.groupby("subject_id")["CDR"].shift(-1)
    
    # Drop rows without a next_CDR (the last visit for each patient)
    usable_pairs = aligned_df.dropna(subset=["next_CDR"])
    
    aligned_path = MANIFEST_DIR / "oasis2_mri_clinical_aligned.csv"
    usable_pairs.to_csv(aligned_path, index=False)
    
    # Leakage Audit (Splits by subject, seed 42)
    import numpy as np
    np.random.seed(42)
    subjects = usable_pairs["subject_id"].unique()
    np.random.shuffle(subjects)
    
    train_split = int(0.7 * len(subjects))
    val_split = int(0.85 * len(subjects))
    
    train_subs = set(subjects[:train_split])
    val_subs = set(subjects[train_split:val_split])
    test_subs = set(subjects[val_split:])
    
    # Verify no overlap
    assert len(train_subs.intersection(val_subs)) == 0
    assert len(train_subs.intersection(test_subs)) == 0
    
    audit = {
        "total_mri_subjects": mri_manifest["subject_id"].nunique(),
        "total_mri_sessions": len(mri_manifest),
        "clinically_matched_subjects": aligned_df["subject_id"].nunique(),
        "clinically_matched_visits": len(aligned_df),
        "usable_prediction_pairs": len(usable_pairs),
        "split_strategy": "Subject-level isolation",
        "train_subjects": len(train_subs),
        "val_subjects": len(val_subs),
        "test_subjects": len(test_subs),
        "overlap_count": 0,
        "seed": 42
    }
    
    with open(MANIFEST_DIR / "mri_leakage_audit.json", "w") as f:
        json.dump(audit, f, indent=4)
        
    print("Alignment and Leakage Audit complete.")

if __name__ == "__main__":
    extract_archives()
    manifest = scan_and_manifest()
    clinical_alignment(manifest)
    print("Phase 2-6 Automated Script Finished.")
