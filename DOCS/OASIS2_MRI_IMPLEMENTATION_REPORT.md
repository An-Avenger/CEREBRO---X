# OASIS-2 MRI Implementation Report

## 1. Dataset Discovery
The raw OASIS-2 MRI archives (`OAS2_RAW_PART1.tar.gz` and `OAS2_RAW_PART2.tar.gz`) were discovered in the `E:\PROJECTS\CEREBRO-X\` directory. They were not present in the default Downloads folder. Total size was approximately 19.3 GB.

## 2. Extraction Structure
A robust external data directory was created at `D:\CEREBRO_DATA\OASIS2\`.
The archives were safely moved into `archives/` and extracted into `raw/` using a fully automated script (`scripts/process_mri_data.py`), preserving the original hierarchy and mitigating local SSD wear.

## 3. MRI Statistics
The extraction parsed all `.nii` and `.nii.gz` files natively.
The manifest output (`manifests/oasis2_mri_manifest.csv`) contains the full inventory of valid 3D volumes discovered.

## 4. Clinical-MRI Alignment
The MRI dataset was joined against the existing clinical dataset (`data/raw/oasis_longitudinal.csv`) on `Subject ID` and `MRI ID`. We calculated the target prediction by shifting the `CDR` column to the `next_visit`.

## 5. Number of Usable Pairs
Rows where the patient had no "next visit" (the terminal visit) were dropped. The exact yield is documented in `manifests/mri_leakage_audit.json`.

## 6. Missing Data
Scans lacking corresponding clinical metadata were excluded from the aligned dataset to prevent missing-feature errors during fusion.

## 7. Train/Validation/Test Split
Splitting was explicitly enforced at the **SUBJECT** level using deterministic `seed=42`. 
- 70% Train
- 15% Validation
- 15% Test
No patient overlaps between splits.

## 8. Leakage Audit
A rigorous leakage audit was generated at `manifests/mri_leakage_audit.json`. The target next-visit CDR is safely isolated as the loss label and is never ingested into the feature tensors.

## 9. Preprocessing Pipeline
The pipeline resizes the volumes to `(64, 64, 64)`, canonicalizes the affine orientation using `nibabel.as_closest_canonical`, and normalizes intensity using global z-scoring. Tested via `scripts/test_oasis2_preprocessing.py`.

## 10. MRI Model Architecture
The existing `Lightweight3DCNN` (4-block convolutional network) was leveraged, outputting a 64-dimensional embedding mapped to 4 Alzheimer's severity classes. 

## 11. Kaggle Training Instructions
A Kaggle-ready training script (`scripts/train_oasis2_mri_3d.py`) was generated.
Because of Kaggle constraints, users should upload the `D:\CEREBRO_DATA\OASIS2\raw` dataset directly to Kaggle, run the training script, and download `cnn3d_best.pt`. **Real OASIS-2 MRI training pending.**

## 12. Grad-CAM Integration Status
The 3D Grad-CAM implementation (`gradcam_3d.py`) remains intact. However, it cannot generate valid visual heatmaps until a real MRI-trained checkpoint is produced by Kaggle. 

## 13. Clinical + MRI Fusion Readiness
The `RealBimodalCerebroNet` was maintained as the legacy scalar architecture. Once the 3D CNN is trained, its embedding layer will be functionally joined with the `TemporalCerebroNet` GRU in a unified forward pass.

## 14. Remaining Limitations
- **Unaligned Cohorts:** We do not have EEG data for the OASIS-2 subjects, making true trimodal fusion impossible without a synthetic bridge.
- **Hardware Bottleneck:** Local training of the 3D CNN is impossible on the current machine without OOM (Out-of-Memory) errors, necessitating the Kaggle GPU bridge.
