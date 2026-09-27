# Cerebro X — Dataset Card: OASIS-2 Longitudinal

**Last verified:** August 2026  
**Verified by:** Manual inspection of Kaggle dataset (jboysen/mri-and-alzheimers)

---

## Identity

| Field | Value |
|---|---|
| **Dataset name** | OASIS-2 Longitudinal MRI and Alzheimer's (Kaggle mirror) |
| **Kaggle slug** | `jboysen/mri-and-alzheimers` |
| **File used** | `oasis_longitudinal.csv` |
| **Original source** | Open Access Series of Imaging Studies (OASIS) — oasis-brains.org |
| **Dataset type** | Longitudinal clinical + derived MRI metadata (CSV only) |
| **NOT** | Raw MRI images, OASIS-3, ADNI |

---

## Verified Schema (15 columns)

| Column | Type | Description |
|---|---|---|
| `Subject ID` | string | Unique participant identifier |
| `MRI ID` | string | Unique MRI session identifier |
| `Group` | categorical | Nondemented / Demented / Converted |
| `Visit` | integer | Visit number (1, 2, 3, …) |
| `MR Delay` | integer | Days from first MRI to current MRI |
| `M/F` | categorical | Sex (M/F) |
| `Hand` | categorical | Handedness |
| `Age` | integer | Age at visit |
| `EDUC` | integer | Years of education |
| `SES` | integer | Socioeconomic status (1–5, missing in some rows) |
| `MMSE` | float | Mini-Mental State Examination score (0–30) |
| `CDR` | float | Clinical Dementia Rating (0.0, 0.5, 1.0, 2.0) |
| `eTIV` | float | Estimated Total Intracranial Volume (mm³) |
| `nWBV` | float | Normalized Whole Brain Volume |
| `ASF` | float | Atlas Scaling Factor |

---

## Verified Cohort Facts

| Fact | Value |
|---|---|
| Total rows | 373 |
| Total columns | 15 |
| Unique subjects | 150 |
| Subjects with 2 visits | 94 |
| Subjects with 3 visits | 43 |
| Subjects with 4 visits | 9 |
| Subjects with 5 visits | 4 |
| Group: Nondemented | 190 |
| Group: Demented | 146 |
| Group: Converted | 37 |
| CDR = 0.0 | 206 |
| CDR = 0.5 | 123 |
| CDR = 1.0 | 41 |
| CDR = 2.0 | 3 |
| MMSE count (non-null) | 371 |
| MMSE mean | 27.34 |
| MMSE std | 3.68 |
| MMSE min | 4 |
| MMSE max | 30 |
| Subjects with changing CDR | 34 |

---

## Longitudinal Pair Count (Derived)

For next-visit CDR prediction:
- Each subject contributes (N_visits − 1) pairs
- 94 subjects × 1 + 43 × 2 + 9 × 3 + 4 × 4 = 94 + 86 + 27 + 16 = **223 pairs**
- **The code must calculate this — do not hard-code 223.**

---

## What This Dataset DOES NOT Contain

- Raw T1-weighted MRI image files (`.nii`, `.dcm`)
- FreeSurfer parcellation outputs
- PET imaging
- Genetic/biomarker data
- EEG data
- ADNI subjects
- OASIS-3 subjects

---

## Access and Terms

This Kaggle mirror is publicly available.
The original OASIS-2 data requires compliance with the OASIS data-use agreement.
Do not re-identify participants.
Do not redistribute raw data.

---

## Known Limitations

1. **Small sample size:** 150 subjects / 223 longitudinal pairs is too small for deep learning without strong regularization or pretrained features. Tabular baselines are the appropriate first approach.
2. **CDR class imbalance:** CDR=2.0 has only 3 records. Evaluating per-class performance is necessary.
3. **Missing SES values:** SES has documented missingness. Strategy: flag + impute with training-set median.
4. **Missing MMSE (2 rows):** Document and handle explicitly.
5. **CSV-only:** eTIV, nWBV, ASF are MRI-derived scalars, not raw MRI tensors. MRI encoder cannot be built from this CSV alone.
6. **Single-site, single-era data:** Generalizability is unknown and likely limited.
