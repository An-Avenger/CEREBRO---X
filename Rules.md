# Cerebro X — Engineering and AI Rules

**Version:** 1.0  
**Purpose:** Prevent hallucinated data, silent leakage, uncontrolled architecture changes, and unreproducible experiments.

---

# 1. Prime Directive

> **Never invent a dataset variable, patient record, diagnosis, visit, scan, metric, or research result.**

If the data do not contain something, the system must report that it is unavailable.

If a variable is uncertain, inspect the official dataset documentation or the downloaded file before coding against it.

---

# 2. Dataset Rules

## R-001 — OASIS-3 is primary

OASIS-3 is the initial longitudinal dataset for Cerebro X.

## R-002 — ADNI is external/secondary

ADNI is initially used for independent validation, transferability, or comparison where compatible variables and access permit.

## R-003 — Never blindly merge OASIS-3 and ADNI

Do not concatenate the datasets and pretend they form one homogeneous cohort.

## R-004 — Preserve dataset identity

Every record must include a `dataset_id`.

## R-005 — Never fabricate missing values

Missing clinical or imaging values must remain missing or be handled by a documented imputation strategy.

## R-006 — Never fabricate modality pairing

Do not attach an EEG record to an MRI subject unless the source dataset explicitly establishes the correspondence.

## R-007 — Do not assume a target exists

Before implementing a prediction head, verify the target variable in the actual dataset release.

---

# 3. Data Access Rules

## R-008 — Restricted data stay private

Raw OASIS-3 and ADNI data must not be committed to GitHub.

## R-009 — Credentials stay private

Never commit:

- API keys;
- access tokens;
- NITRC credentials;
- ADNI credentials;
- database passwords;
- cloud credentials.

## R-010 — Respect dataset terms

All access and use must comply with the official data-use agreements.

## R-011 — Do not attempt re-identification

Never attempt to identify research participants.

---

# 4. Longitudinal Leakage Rules

This is one of the most important parts of the entire project.

## R-012 — Split by patient, not by image

If a patient has five visits, those visits must not be randomly distributed across training and testing.

Bad:

```text
Patient A T0 → train
Patient A T1 → test
```

Good:

```text
Patient A T0/T1/T2 → same split
```

## R-013 — Future information cannot enter baseline features

For a prediction from `T0` to `T1`, information from `T1` or later must not be used to construct the `T0` feature vector.

## R-014 — Fit preprocessing on training data only

Scalers, imputers, feature selectors and learned preprocessing components must be fitted using training data only.

---

# 5. Clinical Variable Rules

## R-015 — Keep raw and derived variables separate

Example:

```text
MMSE_raw
MMSE_delta
MMSE_normalized
```

must not be confused with one another.

## R-016 — Document every derived feature

Every engineered clinical variable must have:

- source variables;
- formula;
- time window;
- missingness behavior.

## R-017 — Avoid target leakage

Do not use a variable as an input if it directly encodes the future target.

---

# 6. MRI Rules

## R-018 — Preserve original provenance

Every processed MRI must be traceable to its source session.

## R-019 — No silent preprocessing

Every preprocessing transformation must be recorded.

## R-020 — Quality control before training

Corrupt, incomplete, incorrectly oriented or otherwise invalid scans must be detected before training.

## R-021 — Consistent preprocessing

Train, validation and test images must undergo the same preprocessing pipeline, except for training-only augmentation.

---

# 7. Model Development Rules

## R-022 — Baselines before complexity

The minimum progression is:

```text
Clinical baseline
      ↓
MRI baseline
      ↓
MRI + Clinical baseline
      ↓
Longitudinal baseline
      ↓
Cerebro X advanced model
```

## R-023 — Every advanced model must have a reason

Do not add a Transformer, attention layer, graph network, diffusion model or other component without a research motivation and measurable hypothesis.

## R-024 — No metric cherry-picking

The primary metric must be selected before comparing final models.

## R-025 — Report more than accuracy

Depending on the task, evaluate suitable metrics such as:

- balanced accuracy;
- precision;
- recall;
- F1;
- AUROC;
- AUPRC;
- MAE;
- RMSE;
- calibration;
- concordance index for appropriate time-to-event tasks.

Not every metric applies to every task.

---

# 8. Explainability Rules

## R-026 — Explainability is not causality

Never write:

> "The model proved that hippocampal atrophy causes Alzheimer's disease."

Instead:

> "The model assigned high attribution to features/regions associated with the prediction."

## R-027 — Use model-compatible explanations

Grad-CAM requires compatible model activations. SHAP requires an appropriate explanation setup. Do not attach an explanation algorithm just because its name appears in a paper.

## R-028 — Explanations must be reproducible

Store:

- model version;
- input preprocessing version;
- explanation method;
- relevant parameters.

---

# 9. Code Rules

## R-029 — Python style

Use:

- type hints;
- small functions;
- descriptive names;
- docstrings for non-obvious research logic.

## R-030 — Configuration over hardcoding

Dataset paths, seeds, hyperparameters and experiment settings belong in configuration files.

## R-031 — No absolute personal paths

Never commit paths such as:

```text
C:\Users\Aryan\...
D:\dataset\...
/home/user/...
```

## R-032 — No giant notebooks as the final implementation

Notebooks are for exploration.

Reusable research logic belongs under `src/`.

## R-033 — Tests for critical logic

At minimum test:

- dataset parsing;
- visit matching;
- patient-level splitting;
- target generation;
- preprocessing;
- output shapes.

---

# 10. AI Coding Assistant Rules

AI coding assistants may:

- generate boilerplate;
- refactor code;
- create tests;
- explain APIs;
- suggest implementation strategies.

AI coding assistants must not:

- invent dataset columns;
- fabricate research metrics;
- claim that code was executed when it was not;
- claim a model is state-of-the-art without verification;
- silently change the research target;
- download restricted datasets without explicit user authorization;
- remove safety/data-provenance checks to make code run.

When uncertain:

```text
STOP → INSPECT → VERIFY → IMPLEMENT
```

---

# 11. Error Handling

Errors must be explicit.

Bad:

```text
if column_missing:
    use_random_value
```

Good:

```text
raise a clear validation error explaining:
- missing column
- expected source
- dataset
- remediation
```

Never hide data problems merely to make the pipeline execute.

---

# 12. Experiment Rules

Every experiment requires:

```text
Experiment ID
Dataset
Data version/download date
Split strategy
Random seed
Preprocessing version
Model version
Hyperparameters
Metrics
Checkpoint
Git commit
```

A model without provenance is an anecdote wearing a lab coat.

---

# 13. Git Rules

Never commit:

```text
data/raw/
data/interim/
data/processed/
*.nii
*.nii.gz
*.dcm
*.zip
*.tar
*.pt
*.pth
.env
credentials/
```

Model checkpoints may be stored separately according to repository size and data policy.

---

# 14. API Rules

The API must distinguish:

```text
research prediction
```

from:

```text
medical diagnosis
```

The frontend must not display research predictions as clinical diagnoses.

---

# 15. Frontend Rules

The interface must show:

- model version;
- dataset/source where appropriate;
- prediction type;
- uncertainty/confidence when available;
- research disclaimer.

Do not use UI language such as:

```text
"You have Alzheimer's."
```

Prefer:

```text
"Model prediction: Alzheimer's disease category"
```

with the appropriate research disclaimer.

---

# 16. Definition of Done

A feature is not complete until:

- code works;
- tests pass where applicable;
- provenance is documented;
- configuration is reproducible;
- no restricted data is committed;
- research assumptions are documented;
- the implementation matches the PRD.

---

# 17. Change Control

Any change to one of these requires updating the relevant documentation:

- prediction target;
- primary dataset;
- external validation dataset;
- model architecture;
- major preprocessing step;
- train/test strategy;
- evaluation metric;
- clinical interpretation.

Do not let the code silently become a different thesis.
