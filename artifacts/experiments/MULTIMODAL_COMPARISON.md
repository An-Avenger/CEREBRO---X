# Modality Comparison

| Experiment | Modalities | Accuracy | Balanced Acc | Macro F1 |
|------------|------------|----------|--------------|----------|
| M1 | Clinical | 75.0% | 55.7% | 55.1% |
| M2 | MRI | N/A | N/A | N/A |
| M3 | EEG | 50.0% | 47.8% | 47.6% |
| M4 | Clinical + MRI | 71.4%* | 52.9%* | N/A |
| M5 | Clinical + EEG | N/A | N/A | N/A |
| M6 | MRI + EEG | N/A | N/A | N/A |
| M7 | Clinical + MRI + EEG | N/A | N/A | N/A |

*\* M4 denotes the legacy scalar MRI fusion experiment. It uses structured volumetric data, NOT the 3D CNN NIfTI embeddings.*

## Scientific Limitations
- **Unaligned Cohorts:** The experiments marked `N/A` currently have no patient-aligned cohort. Clinical data relies on OASIS-2, EEG on OpenNeuro ds004504, and MRI requires ADNI. A multimodal fusion of unaligned subjects is scientifically invalid, hence they have been left strictly as architecture scaffolds.
- **MRI NIfTI Availability:** 3D CNN MRI models (M2) require the ADNI dataset to be securely mounted. Synthetic Gaussian results (formerly 97.5%) have been purged as they misrepresent medical accuracy.
