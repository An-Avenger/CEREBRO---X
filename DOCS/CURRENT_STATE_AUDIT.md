# Cerebro X — Current State Audit (Updated)

*Last updated: 2026-08-25 — Post Phase 3-7 Fix*

## 1. Component Status Audit

| Component | Actual Status | Evidence | Required Work |
| --- | --- | --- | --- |
| **OASIS-2 Clinical** | ✅ IMPLEMENTED | `data/raw/oasis_longitudinal.csv` (150 subjects, 373 visits). Loader, pairs builder, feature engineering all working. | None. |
| **Clinical features** | ✅ IMPLEMENTED | `src/cerebro_x/features/clinical.py` — 19 features including MRI-derived scalars and temporal deltas. Target leakage checks pass. | None. |
| **Longitudinal GRU (Phase 2)** | ✅ IMPLEMENTED & TRAINED | `artifacts/EXP-LONGITUDINAL-001/temporal_gru_cpu.pt`. Accuracy=75.0%, B-Acc=55.7%, F1=0.55. Subject-level split enforced. | None. Serves as Phase 2 baseline. |
| **MRI Branch (Phase 3)** | ✅ IMPLEMENTED & TRAINED | `src/cerebro_x/models/deep/mri_scalar.py`. MRI-derived scalars (nWBV, eTIV, ASF, nwbv_delta). `artifacts/EXP-MRI-SCALAR-001/metrics.json`. Acc=35.7%, B-Acc=27.6%. | Note: scalars only, NOT raw 3D volumes. Raw MRI requires OASIS-3 access. |
| **EEG (Phase 4)** | ❌ NOT IMPLEMENTED | `src/cerebro_x/models/deep/eeg_net.py` is PLANNED architecture only. No EEG dataset with patient-level OASIS-2 alignment exists. See `DOCS/EEG_LIMITATION.md`. | Requires EEG dataset acquisition. Architecture ready. |
| **Multimodal fusion (Phase 5)** | ✅ IMPLEMENTED & TRAINED (Bimodal) | `src/cerebro_x/models/deep/bimodal_fusion.py`. `artifacts/EXP-FUSION-BIMODAL-001/metrics.json`. Clinical+MRI: Acc=71.4%, B-Acc=52.9%, F1=0.54. Ablation complete (3 conditions). | Tri-modal (with EEG) blocked until EEG data available. |
| **Digital Brain Twin (Phase 6)** | ✅ IMPLEMENTED | `src/cerebro_x/brain_twin/extractor.py`. Z_t trajectories extracted per patient. PCA + heatmap plots generated. `artifacts/EXP-BRAIN-TWIN-001/`. Specification in `DOCS/DIGITAL_BRAIN_TWIN_SPECIFICATION.md`. | Z_t is 64-dim GRU hidden state (clinical-only) or 96-dim bimodal. |
| **SHAP (Phase 8)** | ✅ IMPLEMENTED (Clinical) | `src/cerebro_x/explainability/shap_temporal.py`. SHAP plots in `artifacts/EXP-EXPLAIN-001/`. | Extend to bimodal when needed. MRI Grad-CAM requires raw 3D volumes. |
| **Grad-CAM** | ⚠️ SCAFFOLDED | `src/cerebro_x/explainability/gradcam.py`. Cannot be tested without trained MRI CNN on real volumes. | Blocked pending raw MRI data. |
| **Longitudinal Progression (Phase 7)** | ✅ IMPLEMENTED | `artifacts/EXP-PROGRESSION-001/`. Per-class metrics, calibration plots, CDR trajectory plots. Horizon: 1 visit ahead. | None. Limitation documented. |
| **Backend API** | ❌ NOT IMPLEMENTED | No FastAPI/Flask app exists. | Phase 11 work — after AI pipeline complete. |
| **Frontend** | ❌ NOT IMPLEMENTED | No React/HTML dashboard. | Phase 12 work. |
| **Database** | ❌ NOT IMPLEMENTED | File-based artifacts only. | Phase 13 work. |

---

## 2. Fake/Scaffolded Items Removed or Labelled

| Item | Previous State | Current State |
| --- | --- | --- |
| `artifacts/EXP-MULTIMODAL-001/multimodal_cpu.pt` | Fake (synthetic tensors) | Kept but labelled INVALID in `fusion.py` docstring |
| `src/cerebro_x/models/deep/fusion.py` | Claimed as working | Now has `STATUS: SCAFFOLDED — NOT FOR ACTIVE USE` header |
| `src/cerebro_x/models/deep/eeg_net.py` | Claimed as working | Now has `STATUS: PLANNED — NOT CURRENTLY TRAINABLE` header |
| `scripts/run_phase4_multimodal.py` | Used `is_synthetic_eeg=True` | Still present but the new `train_bimodal_fusion.py` is the active script |

---

## 3. Data Leakage Risk Audit

### 3.1 Subject-Level Leakage
* **Status:** ✅ MITIGATED across all experiments.
* **Evidence:** `subject_level_split()` with `_verify_no_subject_overlap()` enforced in every training script. All new scripts (Phase 3, 5, 6, 7) use the same seed=42, 70/15/15 split.

### 3.2 Temporal Leakage
* **Status:** ✅ MITIGATED.
* **Evidence:** `build_longitudinal_pairs.py` builds causal (t → t+1) pairs. `SequenceDataset` builds prefixes [V1], [V1,V2], ... never using future visits.

### 3.3 Target Leakage
* **Status:** ✅ MITIGATED.
* **Evidence:** `next_CDR` and `next_MMSE` explicitly excluded from feature matrix. `_verify_no_target_leakage()` raises if violated.

### 3.4 MRI Feature Leakage Risk
* **Status:** ✅ LOW RISK.
* **Evidence:** MRI scalars (nWBV, eTIV, ASF) from the CURRENT visit only. `nwbv_delta` uses previous visit value. No future MRI info used.

---

## 4. Experiments Summary

| Experiment | Status | Key Result |
| --- | --- | --- |
| EXP-BASELINE-001 | ✅ Done | LogReg: ~72% Bal-Acc |
| EXP-LONGITUDINAL-001 | ✅ Done | GRU: 75.0% Acc, 55.7% B-Acc |
| EXP-MRI-SCALAR-001 | ✅ Done | MRI-only: 35.7% Acc, 27.6% B-Acc |
| EXP-FUSION-BIMODAL-001 | ✅ Done | Clinical+MRI: 71.4% Acc, 52.9% B-Acc |
| EXP-BRAIN-TWIN-001 | ✅ Done | Z_t (64-dim) trajectories for all patients |
| EXP-PROGRESSION-001 | ✅ Done | Per-class + calibration + CDR trajectory plots |
| EXP-EXPLAIN-001 | ✅ Done | SHAP temporal plots for clinical GRU |

---

## 5. What Remains Before Full System (Phases 8–14)

- Phase 8: Research API (FastAPI)
- Phase 9: Frontend dashboard
- Phase 10: Database
- Phase 11: Validation & robustness
- Phase 12: Thesis/paper preparation
- Phase 13: Deployment/Docker
- Phase 14: Final research audit
