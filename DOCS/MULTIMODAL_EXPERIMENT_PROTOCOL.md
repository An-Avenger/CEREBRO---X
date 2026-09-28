# Cerebro X Multimodal Experiment Protocol

This document defines the strict, reproducible methodology used to evaluate multimodal combinations for predicting Alzheimer's disease progression.

## 1. Modality Definition
- **Clinical (M1):** 19 longitudinal features (demographic, cognitive, volumetric scalars).
- **MRI (M2):** T1-weighted NIfTI volumes processed into 3D tensors.
- **EEG (M3):** Raw/spectral temporal sequences.

No artificial fourth modalities (e.g. arbitrarily splitting "demographics" into a new modality) are permitted.

## 2. Target Definition
The standard target is **Next-visit Clinical Dementia Rating (CDR)** (4 classes: 0.0, 0.5, 1.0, 2.0). 
Binary classification tasks (e.g. AD vs Control in EEG) are treated as strictly independent experiments and their accuracies cannot be directly compared to 4-class CDR models.

## 3. Data Splitting (Anti-Leakage)
- **Subject-Level Isolation:** All datasets are split entirely by `Subject ID`. No patient is allowed to span across training and test subsets.
- **Proportions:** Train (70%), Validation (15%), Test (15%).
- **Random State:** Seed 42 is enforced across all `train_test_split` algorithms.
- **Cross-Contamination Prevention:** Scalers (StandardScaler) and Imputers (SimpleImputer) are fit **only** on the training split, and applied forward to the validation/test splits.

## 4. Unaligned Cohorts Protocol
If modalities do not belong to the same patients (e.g. OASIS-2 vs OpenNeuro ds004504), they **must not** be falsely concatenated.
- **Action:** Any fusion experiment requiring mismatched datasets is marked `UNAVAILABLE`.
- **Reasoning:** A model trained on a clinical record from Patient A and an EEG record from Patient B does not learn biological reality, and reporting its accuracy constitutes academic misconduct.

## 5. Evaluation Protocol
- **Metrics:** Accuracy, Balanced Accuracy, and Macro F1 are mandatory.
- **Importance of Balanced Accuracy:** Because early-stage neurodegeneration datasets are highly skewed toward healthy/mild classes, standard Accuracy is highly misleading. Balanced Accuracy and Macro F1 provide the true indicator of clinical utility.
- **Tuning:** Hyperparameters must be tuned solely on the validation split. The test split is evaluated exactly once at the end of the pipeline.

## 6. Real vs Synthetic Guardrails
- Scaffolding utilizing `torch.randn()` or synthetic labels is explicitly blocked from emitting final metrics.
- All experimental outputs must trace back to a mathematically verifiable feature matrix.
