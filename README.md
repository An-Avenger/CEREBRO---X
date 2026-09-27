# Cerebro-X — Longitudinal Digital Brain Twin

> Predicting cognitive decline from multi-visit clinical data using temporal deep learning.

[![Tests](https://img.shields.io/badge/tests-114%20passing-brightgreen)]()
[![Phase](https://img.shields.io/badge/phase-14%20complete-blue)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)]()

---

## What is Cerebro-X?

Cerebro-X is an end-to-end ML research system that builds a **Longitudinal Digital Brain Twin** — a model that learns how a patient's brain health evolves over time and predicts future cognitive decline.

Unlike snapshot-based models, Cerebro-X processes the patient's **entire visit history** through a temporal GRU (Gated Recurrent Unit) network, capturing how symptoms change between visits.

---

## Key Results

| Model | Balanced Accuracy | Hardware |
|---|---|---|
| Dummy Classifier | 25.0% | CPU |
| Random Forest | 45.7% | CPU |
| Logistic Regression (Baseline) | 72.0% | CPU |
| Last-Visit Baseline | 53.7% | CPU |
| **Temporal GRU (Cerebro-X)** | **55.7%** | **Kaggle T4 GPU** |

> **Full report:** [`artifacts/FINAL_REPORT.md`](artifacts/FINAL_REPORT.md)

---

## Project Structure

```
CEREBRO-X/
├── src/cerebro_x/
│   ├── data/           # Data loading, OASIS-2 loader, sequence dataset
│   ├── features/       # Clinical feature engineering
│   ├── models/
│   │   ├── baselines.py        # Logistic Regression, Random Forest
│   │   └── deep/               # MLP, CNN3D, GRU, Multimodal Fusion
│   ├── explainability/ # SHAP temporal explainer, Grad-CAM scaffold
│   └── evaluation/     # Metrics, train/val/test splits
├── scripts/            # Training & reporting scripts
├── configs/            # YAML experiment configs
├── data/processed/     # Built pair datasets
├── artifacts/          # All experiment outputs (models, plots, metrics)
├── tests/              # 75 unit tests
└── notebooks/          # Kaggle cloud training notebooks
```

---

## Quick Start

### 1. Install
```bash
pip install -e .
```

### 2. Run Tests
```bash
python -m pytest tests/ -v
# Expected: 75 passed
```

### 3. Build Dataset Pairs
```bash
python scripts/build_pairs.py --config configs/base.yaml
```

### 4. Train Baselines (CPU)
```bash
python scripts/train_baselines.py
```

### 5. Train Temporal GRU (Kaggle GPU)
Upload `cerebro_x_codebase.zip` as a Kaggle Dataset and run `notebooks/02_cloud_training/kaggle_runner.ipynb`.

### 6. Generate SHAP Explanations
```bash
python scripts/run_explainability.py
```

### 7. Generate Final Report
```bash
python scripts/generate_final_report.py
# Output: artifacts/FINAL_REPORT.md
```

---

## Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Project Setup & Architecture | Done |
| 2 | Dataset Curation (OASIS-2) | Done |
| 3 | Traditional ML Baselines | Done |
| 4 | Multimodal Scaffold (MRI + Clinical) | Done |
| 5 | Longitudinal Temporal GRU | Done |
| 6 | Explainability (SHAP + Grad-CAM) | Done |
| 7 | Final Review & Delivery | Done |
| 8 | Research API (FastAPI) | Done |
| 9 | Frontend Dashboard (Next.js) | Done |
| 10 | Database & Experiment Management | Done |
| 11 | Validation & Robustness (39 tests) | Done |
| 12 | Thesis / Paper Preparation | Done |
| 13 | Deployment (Docker) | Done |
| 14 | Final Research Audit | Done |

---

## Docker Quick Start

```bash
# Build and launch both API + Frontend
docker compose up --build

# API:      http://localhost:8000/docs
# Frontend: http://localhost:3000
```

---

## Local Quick Start

## Dataset

**OASIS-2** (Open Access Series of Imaging Studies)
- 150 patients, 373 sessions
- 223 longitudinal visit pairs built
- Clinical features: Age, MMSE, CDR, brain volumes (nWBV, eTIV), socioeconomic status

---

*M.Tech Research Project — Cerebro-X Digital Brain Twin*


> [!CAUTION]
> **Research Prototype Only.** This system is not a clinical diagnostic device. Model outputs are research predictions and must not be interpreted as medical diagnoses or treatment recommendations.
