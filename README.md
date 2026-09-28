# Cerebro-X — AI-Based Digital Brain Twin for Predicting Neurological Disease Progression

> Predicting cognitive decline from multi-visit Clinical, MRI, and EEG data using temporal deep learning.

[![Tests](https://img.shields.io/badge/tests-240%20passing-brightgreen)]()
[![Phase](https://img.shields.io/badge/phase-15%20complete-blue)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)]()

---

## What is Cerebro-X?

Cerebro-X is an end-to-end ML research system designed to build a **Longitudinal Digital Brain Twin**. It explores the predictive power of fusing multiple data modalities (Clinical records, NIfTI MRI, and EEG) to forecast Alzheimer's Disease progression.

The core validated architecture utilizes a temporal GRU (Gated Recurrent Unit) network to process a patient's **entire visit history** (Clinical Z_t state), capturing how symptoms and volumetric scalars change over time.

---

## Modality Comparison & Current Results

The experimental protocol implements a strict subject-level split (70/15/15) to prevent data leakage. Synthetic data scaffolds are strictly marked as unavailable for final evaluation.

| Experiment | Modalities | Accuracy | Balanced Acc | Macro F1 | Dataset | Status |
|------------|------------|----------|--------------|----------|---------|--------|
| **M1** | Clinical | 75.0% | 55.7% | 55.1% | OASIS-2 | **Validated Core** |
| **M2** | MRI (3D CNN) | N/A | N/A | N/A | ADNI | Awaiting Data |
| **M3** | EEG (Spectral) | 50.0% | 47.8% | 47.6% | ds004504 | Standalone (AD/FTD) |
| **M4** | Clinical + MRI | 71.4%* | 52.9%* | N/A | OASIS-2 | Legacy (Scalars Only) |
| **M5** | Clinical + EEG | N/A | N/A | N/A | Unaligned | Unavailable |
| **M6** | MRI + EEG | N/A | N/A | N/A | Unaligned | Unavailable |
| **M7** | Clinical + MRI + EEG | N/A | N/A | N/A | Unaligned | Scaffold Built |

*\* M4 uses structural scalar data, NOT 3D NIfTI embeddings. M2 requires authorized ADNI access. Multimodal fusions involving EEG or NIfTI MRI are currently unavailable due to dataset cohort misalignment (OASIS-2 vs ADNI vs OpenNeuro), preserving strict scientific validity.*

> **Experimental Protocol:** [`docs/MULTIMODAL_EXPERIMENT_PROTOCOL.md`](docs/MULTIMODAL_EXPERIMENT_PROTOCOL.md)
> **Multimodal Audit:** [`CEREBRO_X_MULTIMODAL_AUDIT.md`](CEREBRO_X_MULTIMODAL_AUDIT.md)

---

## Project Structure

```
CEREBRO-X/
├── src/cerebro_x/
│   ├── data/           # Dataset loaders (OASIS, ADNI, OpenNeuro)
│   ├── features/       # Feature engineering & preprocessing
│   ├── models/
│   │   ├── baselines.py        
│   │   └── deep/               # Temporal GRU, CNN3D, EEGNet, Fusion
│   ├── explainability/ # Real Temporal SHAP, 3D Grad-CAM
│   └── evaluation/     # Subject-level splitting, metrics
├── scripts/            # Training, ablation, and report generation
├── docs/               # Scientific protocols and methodology
├── data/raw/           # ADNI, OpenNeuro, OASIS-2 (mount locations)
├── artifacts/          # Validated checkpoints, metrics, comparisons
├── tests/              # 240 strict unit tests (anti-leakage)
```

---

## Quick Start

### 1. Install & Test
```bash
pip install -e .
python -m pytest tests/ -v
# Expected: 240 passed
```

### 2. Clinical Core Training (M1)
```bash
python scripts/build_pairs.py
python scripts/train_longitudinal.py
```

### 3. API & Dashboard (Docker)
```bash
docker compose up --build
# API:      http://localhost:8000/docs
# Frontend: http://localhost:3000
```

---

## Explainability (XAI)
Cerebro-X enforces genuine explainability over synthetic fallbacks:
- **Clinical (SHAP):** Computes exact temporal feature attributions using `shap.GradientExplainer` against the longitudinal Z_t state.
- **MRI (Grad-CAM):** Hook-based 3D Grad-CAM overlay, ready for activation upon authorized ADNI dataset integration.

---

*M.Tech Research Project — Cerebro-X Digital Brain Twin*

> [!CAUTION]
> **Research Prototype Only.** This system is not a clinical diagnostic device. Model outputs are research predictions and must not be interpreted as medical diagnoses or treatment recommendations.
