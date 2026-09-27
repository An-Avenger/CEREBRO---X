# Cerebro X — MRI Integration (Future Phase)

> **Status: NOT IMPLEMENTED**
>
> MRI integration is deferred until real MRI image files are verified and available.
> Do not implement fake MRI inference. Do not use placeholder image tensors.

---

## Why MRI is Deferred

The current verified dataset (`oasis_longitudinal.csv`) contains only MRI-**derived** scalar
features: `eTIV` (estimated total intracranial volume), `nWBV` (normalized whole brain volume),
and `ASF` (atlas scaling factor). These are numbers, not images.

Raw OASIS-2 MRI files (T1-weighted `.nii.gz`) are a separate download from the OASIS repository
and are NOT included in the Kaggle `jboysen/mri-and-alzheimers` dataset.

---

## Prerequisites Before MRI Integration

1. **Obtain raw MRI files** from `https://sites.wustl.edu/oasisbrains/request-access/`
2. **Verify file formats** — expected: T1-weighted NIfTI (`.nii` or `.nii.gz`)
3. **Match MRI sessions to CSV rows** using `MRI ID` and `Subject ID`
4. **Verify session coverage** — not every CSV row may have a corresponding MRI scan
5. **Document MRI acquisition parameters** — field strength, voxel size, scanner

---

## Planned MRI Pipeline

```
Raw T1w NIfTI
    ↓
Format/metadata validation (nibabel)
    ↓
Orientation normalization (RAS)
    ↓
Quality control (SNR check, visual inspection)
    ↓
Brain extraction / skull stripping
    ↓
Spatial normalization (MNI152 registration)
    ↓
Resampling (1mm isotropic)
    ↓
Intensity normalization (z-score or percentile)
    ↓
Save preprocessed tensor
    ↓
MRI Encoder (3D CNN or volumetric features)
```

---

## Source Structure (Planned, Not Yet Created)

```
src/cerebro_x/
└── imaging/
    ├── __init__.py
    ├── loaders/
    │   └── nifti_loader.py      # TODO: load .nii/.nii.gz, verify subject match
    ├── preprocessing/
    │   ├── orientation.py       # TODO: reorient to RAS
    │   ├── skull_strip.py       # TODO: brain extraction
    │   ├── normalize.py         # TODO: intensity normalization
    │   └── qc.py                # TODO: quality control checks
    ├── models/
    │   └── mri_encoder.py       # TODO: 3D CNN encoder (justified after baselines)
    └── inference/
        └── extract_embedding.py # TODO: extract MRI embedding for a given session
```

---

## Split Rules for MRI

When MRI data is available, splitting rules become even more critical:

- **Never split MRI slices from the same subject across train/test.** Slices are not independent.
- **Split must be at the subject level**, not at the session, slice, or patch level.
- If a subject has multiple MRI sessions, all sessions for that subject stay in the same split.
- MRI preprocessing must be fitted/calibrated on training subjects only.

---

## Fusion with Clinical Data

Once MRI encoder is trained independently:

```
MRI Embedding (from MRI Encoder)
         +
Clinical Representation (from clinical pipeline)
         ↓
Fusion Layer (concatenate → MLP, or attention-based)
         ↓
Next-state Prediction
```

This fusion is implemented ONLY after both individual modality models are evaluated separately.

---

## Reference

OASIS MRI data access: https://sites.wustl.edu/oasisbrains/home/oasis-2/
