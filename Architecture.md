# Cerebro X — System Architecture

**Version:** 1.0  
**Status:** Initial architecture  
**Primary data:** OASIS-3  
**External validation:** ADNI  
**Initial modality scope:** Structural T1 MRI + clinical/cognitive data

---

## 1. Architecture Principle

Cerebro X is divided into five layers:

1. Data
2. Processing
3. Representation learning
4. Prediction and explainability
5. Application/API

The architecture deliberately separates research code from application code.

---

## 2. High-Level Architecture

```text
                         CEREBRO X
                            |
        +-------------------+-------------------+
        |                                       |
        v                                       v
   OASIS-3 DATA                              ADNI DATA
        |                                       |
        +-------------------+-------------------+
                            |
                     Dataset Adapters
                            |
                            v
                Data Validation + Provenance
                            |
             +--------------+--------------+
             |                             |
             v                             v
        MRI Pipeline                Clinical Pipeline
             |                             |
             v                             v
      MRI Representation          Clinical Representation
             |                             |
             +--------------+--------------+
                            |
                            v
                 Longitudinal Alignment
                            |
                            v
                  Multimodal Fusion
                            |
                            v
                  Patient Brain State Z_t
                            |
             +--------------+--------------+
             |                             |
             v                             v
       Current State                 Future State
       Prediction                    Prediction
             |                             |
             +--------------+--------------+
                            |
                            v
                       Explainability
                            |
                            v
                  Evaluation + Reports
                            |
                            v
                    FastAPI Research API
                            |
                            v
                     React Frontend
```

---

## 3. Repository Structure

```text
cerebro-x/
|
├── README.md
├── PRD.md
├── Architecture.md
├── Rules.md
├── Phases.md
├── Design.md
|
├── configs/
│   ├── base.yaml
│   ├── datasets/
│   │   ├── oasis3.yaml
│   │   └── adni.yaml
│   └── experiments/
│
├── data/
│   ├── raw/                 # NEVER COMMIT
│   ├── interim/             # NEVER COMMIT
│   ├── processed/           # NEVER COMMIT
│   └── metadata/
│
├── notebooks/
│   ├── 01_dataset_audit/
│   ├── 02_mri_qc/
│   ├── 03_clinical_audit/
│   ├── 04_baselines/
│   └── 05_longitudinal/
│
├── src/
│   └── cerebro_x/
│       ├── data/
│       │   ├── oasis3/
│       │   ├── adni/
│       │   ├── schemas.py
│       │   └── provenance.py
│       │
│       ├── preprocessing/
│       │   ├── mri.py
│       │   ├── clinical.py
│       │   ├── longitudinal.py
│       │   └── qc.py
│       │
│       ├── features/
│       │   ├── imaging.py
│       │   ├── clinical.py
│       │   └── temporal.py
│       │
│       ├── models/
│       │   ├── baselines/
│       │   ├── encoders/
│       │   ├── fusion/
│       │   ├── temporal/
│       │   └── cerebro_twin.py
│       │
│       ├── explainability/
│       │   ├── mri.py
│       │   ├── clinical.py
│       │   └── latent.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── splits.py
│       │   ├── calibration.py
│       │   └── cross_dataset.py
│       │
│       └── utils/
│
├── experiments/
│   ├── configs/
│   ├── runs/
│   └── reports/
│
├── api/
│   └── app/
│       ├── main.py
│       ├── routes/
│       ├── schemas/
│       └── services/
│
├── frontend/
│   └── react-app/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── data/
│
├── scripts/
│   ├── audit_oasis3.py
│   ├── audit_adni.py
│   ├── preprocess.py
│   └── train.py
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── .gitignore
└── requirements.txt / pyproject.toml
```

---

## 4. Data Layer

### OASIS-3 Adapter

Responsibilities:

- load official OASIS-3 metadata;
- identify subjects and sessions;
- load demographic/clinical/cognitive tables;
- identify MRI sessions;
- preserve original identifiers in protected/local metadata;
- expose a common internal schema.

### ADNI Adapter

Responsibilities:

- load authorized ADNI exports;
- identify subject/visit/image relationships;
- load compatible clinical/cognitive variables;
- identify MRI scans;
- preserve ADNI provenance;
- expose the same internal schema where possible.

The adapters must not silently rename incompatible variables to make datasets appear identical.

---

## 5. Canonical Internal Schema

A common internal representation may contain:

```text
dataset_id
subject_id
visit_id
visit_days
mri_session_id
mri_path
mri_sequence
diagnosis
cognitive_variables
demographics
preprocessing_version
source_metadata
```

Not every field will be populated for every dataset.

The schema must support missing values explicitly rather than inventing data.

---

## 6. MRI Pipeline

Initial MRI target:

**T1-weighted structural MRI**

Pipeline:

```text
Raw MRI
  ↓
Format / metadata validation
  ↓
Orientation normalization
  ↓
Quality control
  ↓
Brain extraction (if required)
  ↓
Spatial normalization / registration
  ↓
Resampling
  ↓
Intensity normalization
  ↓
Tensor conversion
  ↓
MRI Encoder
```

Potential MRI representations:

### Option A — 2D slices

Lower computational cost, weaker 3D anatomical context.

### Option B — 2.5D

Multiple neighboring slices as channels.

### Option C — 3D CNN

Best anatomical context but significantly higher compute/memory requirements.

### Option D — Precomputed regional/volumetric features

Useful baseline and computationally efficient.

The final choice must be based on the actual available data and hardware budget.

---

## 7. Clinical Pipeline

```text
Clinical tables
      ↓
Variable audit
      ↓
Missingness analysis
      ↓
Temporal alignment
      ↓
Leakage detection
      ↓
Normalization / encoding
      ↓
Clinical Encoder
```

Candidate variables may include cognitive and clinical assessments available in the selected dataset.

No variable is automatically assumed to exist. The dataset audit phase must establish exact names, visit coverage, missingness and meaning.

---

## 8. Longitudinal Representation

For a patient with observations:

```text
T0 → T1 → T2 → T3
```

each time point produces:

```text
MRI_t + Clinical_t
        ↓
      Encoder
        ↓
      Z_t
```

The sequence:

```text
Z0, Z1, Z2, Z3
```

can then be processed by a temporal model.

Candidate temporal models:

- GRU/LSTM;
- temporal Transformer;
- temporal attention;
- simple delta/change features as a baseline.

The simplest valid baseline should always be implemented before the more complex model.

---

## 9. Cerebro X Latent Brain State

The core latent state is:

```text
Z_t = f(MRI_t, Clinical_t)
```

For longitudinal modeling:

```text
H_t = TemporalModel(Z_0 ... Z_t)
```

Future prediction:

```text
Y_(t+h) = PredictionHead(H_t)
```

Possible outputs:

- future cognitive score;
- cognitive decline/change;
- future diagnostic category;
- progression/conversion event.

The exact target is a Phase 1.1 decision.

---

## 10. Fusion Strategy

Initial fusion baseline:

```text
MRI Encoder ──┐
              ├── Concatenate → MLP → Prediction
Clinical MLP ─┘
```

Advanced Cerebro X fusion:

```text
MRI Embedding ─────┐
                   │
Clinical Embedding ├──► Attention / Gated Fusion
                   │
Temporal Context ──┘
                         ↓
                    Brain State Z_t
```

The advanced architecture must only be evaluated against strong baselines.

---

## 11. Dataset Separation

OASIS-3 and ADNI will not be blindly merged.

Recommended evaluation structure:

```text
Experiment A:
OASIS-3 train → OASIS-3 validation → OASIS-3 test

Experiment B:
ADNI-compatible cohort → independent evaluation

Experiment C:
Train on OASIS-3 → evaluate on ADNI
```

The exact direction may change after dataset audit.

Cross-dataset experiments are intended to measure generalization/domain shift.

---

## 12. Explainability Layer

### MRI

Potential techniques:

- Grad-CAM for CNN-based architectures;
- Integrated Gradients;
- occlusion analysis;
- attention visualization where applicable.

### Clinical

Potential techniques:

- SHAP;
- permutation importance;
- feature attribution.

### Important rule

An explanation indicates model sensitivity/attribution. It does **not** establish that a brain region or clinical feature causes the disease.

---

## 13. API Architecture

```text
React Frontend
      ↓
FastAPI
      ↓
Service Layer
      ↓
Model Registry / Inference
      ↓
Preprocessing
      ↓
Prediction
      ↓
Explanation
      ↓
Result JSON
```

Example conceptual response:

```json
{
  "dataset": "research_case",
  "prediction": {
    "current_state": "...",
    "future_outcome": "...",
    "confidence": 0.0
  },
  "longitudinal": {
    "time_points": []
  },
  "explanation": {
    "clinical_features": [],
    "mri_regions": []
  },
  "model_version": "..."
}
```

The actual schema will be finalized after the research target is locked.

---

## 14. Storage Architecture

PostgreSQL may store:

- research-case metadata;
- experiment metadata;
- prediction results;
- model versions;
- explanation metadata;
- application users if authentication is later required.

Raw OASIS-3/ADNI imaging should not be placed into PostgreSQL.

Object/file storage should hold derived research artifacts where permitted.

---

## 15. MLOps / Experiment Tracking

Each experiment should record:

```text
experiment_id
dataset
dataset_version_or_download_date
split_seed
preprocessing_version
model_version
hyperparameters
training_environment
metrics
checkpoint
git_commit
```

MLflow may be introduced after the first baseline is working.

---

## 16. Technology Stack

### Research / ML

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- nibabel
- SimpleITK
- MONAI where useful
- matplotlib
- SHAP
- Captum where appropriate

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Database

- PostgreSQL
- SQLAlchemy

### Frontend

- React
- TypeScript
- Vite
- a charting/visualization library selected during implementation

### Infrastructure

- Docker
- Git
- GitHub
- GPU environment such as Kaggle/Lightning AI subject to current availability and project limits

Do not add a dependency merely because it is fashionable. Dependencies are not Pokémon.

---

## 17. Architecture Decision Records

Every major architectural decision should be documented under:

```text
docs/adr/
```

Examples:

```text
ADR-001-primary-dataset.md
ADR-002-prediction-target.md
ADR-003-mri-representation.md
ADR-004-temporal-model.md
ADR-005-cross-dataset-validation.md
```

---

## 18. First Implementation Boundary

The first executable milestone is **not the web app**.

It is:

```text
OASIS-3 metadata
      ↓
Dataset audit
      ↓
Subject/visit table
      ↓
MRI↔clinical alignment
      ↓
Clean longitudinal dataset
      ↓
Baseline experiment
```

Only after this works should model development proceed.
