# Cerebro X — Research Limitations

> **Required reading before interpreting any Cerebro X output.**

---

## Dataset Limitations

### 1. Small sample size
The current dataset (OASIS-2 Kaggle CSV) contains 150 subjects and 223 longitudinal pairs.
This is far too small for deep learning without substantial regularization or pretrained features.
All accuracy estimates have wide confidence intervals at this scale.

### 2. CDR class imbalance
CDR = 2.0 appears in only 3 records (out of 373).
Per-class metrics for CDR = 2.0 are not reliable.
Balanced accuracy and macro-averaged metrics are more appropriate than raw accuracy.

### 3. CSV-only MRI features
`eTIV`, `nWBV`, and `ASF` are MRI-derived scalar summaries.
They do not contain raw voxel-level spatial information.
A full CNN-based MRI encoder cannot be built from this CSV.
Conclusions about "MRI contribution" are limited to these three derived scalars.

### 4. Single-site, single-era data
OASIS-2 was collected at a single institution.
Models trained on OASIS-2 may not generalize to different scanners, protocols, or demographics.
External validation (ADNI) is needed to assess generalizability.

### 5. Missing data
- `SES` has documented missing values in some rows.
- `MMSE` has 2 missing values.
- Imputation introduces uncertainty. Imputation strategy is documented in `configs/datasets/oasis2_kaggle.yaml`.

---

## Modeling Limitations

### 6. Longitudinal pairs are not independent
Consecutive visits from the same subject are correlated.
Standard cross-validation (ignoring subject identity) would leak information.
Subject-level splitting reduces but does not eliminate all correlation.

### 7. Short follow-up for some subjects
94 of 150 subjects have only 2 visits.
For these subjects only one training pair is available.
Long-term progression modeling is limited.

### 8. Prediction horizon is visit-based, not time-based
The model predicts the "next visit" CDR, not CDR at a specific future time point.
Visit spacing varies (see `MR Delay` column).
Time-to-event analysis would require a different formulation.

### 9. Baselines may outperform complex models
With 150 subjects, a well-tuned logistic regression may match or outperform more complex models.
Complexity must be justified by a measurable improvement on validation data.

---

## Explainability Limitations

### 10. Feature importance ≠ causation
Model feature importance indicates what the model used, not what causes cognitive decline.
Hippocampal atrophy is associated with Alzheimer's disease, but this model cannot
establish or confirm that relationship causally.

### 11. SHAP is model-specific
SHAP values computed for a Random Forest have a different interpretation than SHAP for a linear model.
Do not compare SHAP values across model types without careful qualification.

---

## Clinical Disclaimer

### 12. NOT a clinical device
Cerebro X is an academic research prototype.
Its predictions must not be used for medical diagnosis, prognosis, or treatment planning.
No clinical validation has been performed.
The system has not been assessed for clinical safety.

---

## Future Work Required

- External validation on ADNI
- Raw MRI integration (currently only derived scalars)
- Larger dataset (OASIS-3)
- Calibration analysis
- Confidence intervals via bootstrap
- Time-indexed (not visit-indexed) prediction horizon
- Prospective validation
