# Cerebro X — EEG Modality Limitation

**Status:** NOT IMPLEMENTED  
**Classification:** DOCUMENTED LIMITATION (not a scaffolded placeholder)

---

## Summary

The EEG (Electroencephalography) branch of Cerebro X is **not implemented** because no suitable EEG dataset with patient-level alignment to the existing OASIS-2 clinical cohort exists. This document explains what is missing, why it is necessary, what alternatives exist, and what scientific claims cannot yet be made.

---

## 1. What Is Missing

### 1.1 Dataset

No EEG dataset with the following properties is publicly available:

| Requirement | Status |
|---|---|
| EEG recordings for Alzheimer's Disease patients | ⚠️ Some exist (see §3), but not matched to OASIS-2 |
| Longitudinal EEG across multiple visits | ❌ Rare |
| Patient ID alignment with OASIS-2 subjects | ❌ Not available |
| Time-point alignment with clinical visits (±90 days) | ❌ Not available |
| Open access with compatible licensing | ⚠️ Partial |

### 1.2 Code Status

The following EEG-related files exist as **planned architecture only**:

| File | Status |
|---|---|
| `src/cerebro_x/models/deep/eeg_net.py` | Architecture code (EEGNet, Lawhern et al. 2018) — **NEVER TRAINED** |
| `src/cerebro_x/models/deep/fusion.py` | TriModalCerebroNet — **SCAFFOLDED, uses synthetic tensors** |
| `scripts/run_phase4_multimodal.py` | Training script with `is_synthetic_eeg=True` — **INVALID FOR RESEARCH** |

**CRITICAL:** The `multimodal_cpu.pt` checkpoint in `artifacts/EXP-MULTIMODAL-001/` was trained on randomly generated EEG tensors (`torch.randn`). Its metrics are meaningless and it does NOT represent a trained EEG model. This checkpoint must not be used in any evaluation or demonstration.

---

## 2. Why EEG Is Necessary for the Full Vision

The full Cerebro X specification requires EEG because:

1. **Functional complement to structural MRI:** MRI captures structural atrophy; EEG captures functional brain dynamics (synchrony, spectral power changes, coherence).
2. **Early biomarker potential:** EEG abnormalities (e.g., slowing of alpha rhythms, increased delta power) have been documented in pre-clinical Alzheimer's stages before significant structural change.
3. **Longitudinal sensitivity:** EEG may capture faster-changing functional states than structural MRI.
4. **Cost and accessibility:** EEG is cheaper than MRI and could increase deployment reach.

Without EEG, Cerebro X is a **Clinical + Structural MRI** system — which is still scientifically valid but not the full tri-modal vision.

---

## 3. Alternatives That Could Enable EEG Integration

### Option A: Separate EEG Cohort (Independent Encoder)

Use an existing EEG-Alzheimer's dataset to pre-train the EEGNet encoder independently:

| Dataset | Subjects | Availability |
|---|---|---|
| OpenNeuro ds003670 (EEG Alzheimer's) | ~88 | Public |
| CAUEEG (Seoul National University) | ~1,378 | Request-based |
| TUH EEG Corpus (Temple University) | Large | Free registration |
| ADNI EEG (limited) | ~40 | ADNI access required |

**Limitation:** Pre-training on a different cohort and then fusion with OASIS-2 clinical data is architecturally possible but **scientifically problematic** — the patient populations are different, defeating the purpose of patient-level multimodal fusion.

### Option B: Simulated EEG Features (Clearly Labeled)

Generate biologically-informed synthetic EEG band-power features correlated with CDR labels for architecture validation. **Must be clearly labeled as SYNTHETIC in all results.**

### Option C: Wait for Dataset Access

Apply for access to ADNI EEG data or OpenNeuro datasets, establish a proper patient-level alignment pipeline, then implement Phase 4 with real data.

---

## 4. Scientific Claims That Cannot Be Made Without EEG

The following claims are PROHIBITED until real EEG is integrated:

- ❌ "Cerebro X uses EEG-based functional brain monitoring"
- ❌ "The tri-modal fusion combines clinical, MRI, and EEG information"
- ❌ "EEG features improve prediction over clinical-only baseline"
- ❌ Any metric from the `multimodal_cpu.pt` checkpoint

The following claims remain VALID with the current implementation:

- ✅ "Cerebro X integrates longitudinal clinical data and MRI-derived structural features"
- ✅ "The bimodal fusion (Clinical + MRI scalars) improves over single-modality baselines" *(if ablation confirms this)*
- ✅ "The system is architecturally designed to incorporate EEG when data becomes available"

---

## 5. Decision

EEG integration is **deferred** pending dataset access. The current architecture (`eeg_net.py`, `fusion.py`) is kept as planned code but is **explicitly not active** in any training or evaluation pipeline.

All active experiments use only `Clinical` and `MRI-derived scalars` modalities.
