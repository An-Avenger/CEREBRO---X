# Cerebro X — Dataset and Modality Rationale

*Document Version: 1.0 (Phase 1)*

This document explicitly defines the datasets required for Cerebro X, the rules for temporal and patient-level alignment, and the strict scientific limitations of our multimodal integration strategy.

---

## 1. Required Modalities & Target Representation

To construct the patient-specific **Brain State Vector**, Cerebro X requires:
1. **Clinical / Cognitive:** Age, sex, education, MMSE, and current CDR.
2. **Structural MRI:** 3D structural representations (T1w) processed into learned embeddings.
3. **EEG (Functional):** Temporal/spectral features representing functional state.

### The Scientific Limitation
A true, perfectly matched, longitudinal public dataset containing Clinical, MRI, *and* EEG data for the exact same patient cohort and aligned time points **does not exist** (e.g., ADNI and OASIS do not mandate synchronized longitudinal EEG for all MRI cohorts). 

**CRITICAL RULE ENFORCEMENT:** We absolutely **will not** naively merge unrelated EEG datasets with MRI/Clinical datasets and falsely claim them as patient-level multimodal data.

### The Alternative Architecture Strategy
Because a tri-modal matched dataset is unavailable:
1. **Clinical + MRI Fusion:** This will form the core longitudinal architecture, sourced from OASIS or ADNI, where patient and time-point alignment *is* valid.
2. **Independent Modality Branches:** All modalities (including EEG) will be pre-trained and evaluated as independent encoders on respective valid cohorts.
3. **Simulated Missingness:** During development and architectural scaffolding, if real raw data (e.g., 3D MRI volumes) is pending institutional access, the pipeline will generate properly shaped simulated tensors to verify architectural shape and gradient flow, explicitly marking them as `SCAFFOLDED` until real data replaces them.

---

## 2. Primary Dataset: OASIS-3 (Current Fallback: OASIS-2)

### Source & Access
- **Source:** Washington University (Open Access Series of Imaging Studies).
- **Status:** Cerebro X currently utilizes the Kaggle OASIS-2 subset (Clinical/Volumetric only). We are blocked on raw OASIS-3 MRI access.

### Dataset Profile (OASIS-2 subset)
- **Subjects:** 150 unique patients.
- **Visits:** 373 total clinical/scanning sessions (2–5 longitudinal visits per patient).
- **Modalities Available:** Clinical metadata, derived scalar MRI volumetric data (nWBV, eTIV, ASF).
- **Longitudinal Info:** Yes (Follow-up days mapped).
- **Missing Modalities:** Raw 3D MRI volumes and EEG are missing in this subset.

### Preprocessing Requirements
- **Clinical:** Imputation (median), scaling, categorical one-hot encoding, temporal delta computation.
- **MRI (Future OASIS-3):** Brain extraction (skull stripping), spatial registration (MNI152), intensity normalization, and 3D tensor conversion.

---

## 3. Secondary Dataset: ADNI (External Validation)

### Source & Access
- **Source:** Alzheimer's Disease Neuroimaging Initiative (LIDA).
- **Status:** Requires explicit application and data-use agreements. Not currently integrated.

### Purpose
ADNI will be used exclusively as a cross-dataset validation benchmark to measure domain shift. It will not be blindly concatenated into the training data.

---

## 4. Patient and Time-Point Alignment Rules

To prevent data leakage and false multimodal assumptions, Cerebro X enforces the following matching rules:

### A. Patient-ID Alignment
- Modalities will only be fused if they carry an identical, cryptographically hashed, or strictly verified Patient ID (e.g., `OAS2_0001`).
- **Zero-overlap constraint:** Train, Validation, and Test splits are strictly partitioned by Patient ID. A patient cannot cross splits.

### B. Time-Point Alignment (Temporal Matching)
- For a Clinical Visit at time $T$, an MRI scan is only considered a valid multimodal pair if the scan occurred within $T \pm 90$ days. 
- Any scan outside this tolerance is treated as a separate unaligned event or discarded.

### C. Missing Modality Handling
- If a patient has a valid clinical visit but no MRI within the time-point tolerance, the architecture will utilize missing-modality imputation (e.g., zero-masking the MRI encoder output and setting a binary `missing_mri` indicator flag for the fusion layer).

---

## 5. Ethical & Licensing Considerations

- **OASIS / ADNI Data Use Agreements:** Explicitly forbid redistribution of raw datasets. No raw images or clinical CSVs will be committed to the Cerebro X repository (`.gitignore` enforces this).
- **De-identification:** All patient IDs must remain pseudonymous as provided by the original dataset.
- **Medical Disclaimer:** Cerebro X is a non-clinical research prototype. Predictions must not be used for medical decisions.
