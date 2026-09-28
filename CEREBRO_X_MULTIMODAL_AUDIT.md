# CEREBRO X: MULTIMODAL AUDIT

## 1. What was already implemented
- **Clinical Model**: `TemporalCerebroNet` (GRU) on 19 clinical features, evaluated on OASIS-2 longitudinal records.
- **Legacy Bimodal Fusion**: A scalar-based `BimodalCerebroNet` combining clinical data and 3 basic MRI scalar volumes (nWBV, eTIV, ASF) extracted manually.
- **EEG Model**: An independent `EEGSpectralEncoder` pipeline operating on the OpenNeuro ds004504 dataset.
- **Explainability**: Real `GradientExplainer` (SHAP) for the clinical model, and `Grad-CAM` for the CNN.
- **Digital Brain Twin (Z_t)**: A 64-dimensional clinical latent state `Z_c` exists and works.

## 2. What was newly implemented
- **MRI 3D CNN Pipeline**: A functional PyTorch pipeline utilizing `nibabel` to ingest real NIfTI volumes, replacing a synthetic artifact-generating scaffold.
- **Tri-Modal Fusion Scaffold**: `TriModalCerebroNet` (Clinical + MRI + EEG) architecture implemented in `fusion.py`, explicitly designed to prevent execution on unaligned synthetic data.
- **Honest Metrics Manifest**: Wiped out fabricated 97.5% MRI synthetic accuracy and >77% leaked EEG accuracy in favor of verifiable real-world metrics.

## 3. Dataset used for each experiment
- **M1 (Clinical)**: OASIS-2 Longitudinal (150 subjects, 223 samples)
- **M2 (MRI)**: ADNI (NIfTI awaiting mount by authorized user)
- **M3 (EEG)**: OpenNeuro ds004504 (87 subjects)
- **M4 (Clinical + MRI Scalar)**: OASIS-2
- **M5 (Clinical + EEG)**: N/A (Unaligned)
- **M6 (MRI + EEG)**: N/A (Unaligned)
- **M7 (Clinical + MRI + EEG)**: N/A (Unaligned)

## 4-12. Metrics & Configurations
| Config | Target | Architecture | Split | Acc | B-Acc | F1 |
|--------|--------|--------------|-------|-----|-------|----|
| **M1** | 4-class CDR | GRU (19 feat) -> Linear | Subject (70/15/15) | 75.0% | 55.7% | 55.1% |
| **M3** | Binary AD/FTD | CNN/Spectral | Subject | 50.0% | 47.8% | 47.6% |
| **M4** | 4-class CDR | GRU + MLP (Scalar) | Subject (70/15/15) | 71.4% | 52.9% | N/A |

*Note: Confusion matrices for M1 and M3 indicate a strong bias toward the majority class (CDR=0 for OASIS-2, Control for ds004504).*

## 13. Data Leakage Audit
- **Status**: CLEARED. 
- A rigorous subject-level split script (`splits.py`) strictly isolates patients. Preprocessor scalers and imputers are strictly fitted on the train split.
- Previous EEG test-set contamination (scaling before split) was resolved.

## 14. Real vs Synthetic Status
- **Real**: M1, M3, M4.
- **Synthetic/Removed**: M2 (previously used Gaussian noise volumes; stripped to await real ADNI data), M5-M7 (Tri-modal scaffold previously trained on synthetic data).

## 15. Missing Modality Limitations
- The underlying constraint is that OASIS-2 (Clinical), ADNI (MRI), and OpenNeuro ds004504 (EEG) contain entirely disjoint patient cohorts. Real trimodal fusion is technically impossible without creating "Frankenstein" patients.

## 16. Final Cerebro X Model
- Currently, **M1 (Clinical TemporalCerebroNet)** serves as the mathematically sound core of Cerebro X.

## 17. Whether multimodal fusion improved performance
- Comparing M1 (Clinical: 75.0% Acc, 55.7% B-Acc) with M4 (Legacy Clinical+MRI: 71.4% Acc, 52.9% B-Acc) indicates that simple scalar MRI fusion **reduced** performance, likely due to added noise/dimensionality without corresponding informative embeddings. 

## 18 & 19. >90% Objective
- **Achieved?** No.
- **Reason**: We strictly enforced subject-level splitting and eliminated fabricated synthetic MRI tests. Achieving >90% on OASIS-2 4-class CDR prediction without leakage or massive class imbalance exploiting is highly unrealistic for standard temporal models. Real medical AI evaluates on balanced accuracy and F1.

## 20. Recommended Next Research Step
- Secure access to a unified dataset containing multimodal temporal records for the *same* subjects (e.g., ADNI comprehensive longitudinal dataset containing Clinical, NIfTI MRI, and genetic panels) to operationalize the scaffolded `TriModalCerebroNet`.
