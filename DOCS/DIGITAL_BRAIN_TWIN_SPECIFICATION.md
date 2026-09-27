# Cerebro X — Digital Brain Twin Specification

**Version:** 1.0  
**Phase:** 6  
**Status:** IMPLEMENTED (Clinical-only Z_t from GRU) / PLANNED (bimodal Z_t from BimodalCerebroNet)

---

## 1. Formal Definition

The **Digital Brain Twin** of patient P at visit time t is defined as the latent vector:

$$Z_t^{(P)} \in \mathbb{R}^d$$

that summarizes all available multimodal information about patient P's brain state as of visit t.

### 1.1 Current Implementation (Phase 6, Clinical-Only)

$$Z_t^{(P)} = h_t \in \mathbb{R}^{64}$$

where $h_t$ is the final hidden state of the trained GRU after processing the clinical visit history:

$$h_t = \text{GRU}(x_1, x_2, \ldots, x_t)$$

and each $x_i \in \mathbb{R}^{19}$ is the 19-dimensional clinical feature vector at visit $i$:

$$x_i = [\text{age}_i, \text{sex}, \text{educ}, \text{ses}, \text{MMSE}_i, \text{CDR}_i, \text{nWBV}_i, \text{eTIV}_i, \text{ASF}_i, \Delta\text{MMSE}_i, \Delta\text{CDR}_i, \Delta\text{nWBV}_i, \ldots]$$

**Causal constraint:** The GRU is unidirectional. $Z_t$ uses only visits $\{x_1, \ldots, x_t\}$. It has no access to any future visit $x_{t+1}, \ldots, x_T$.

### 1.2 Extended Definition (Phase 5, Bimodal Clinical + MRI)

$$Z_t^{(P)} = [h_t^{\text{clin}} \| v_t^{\text{MRI}}] \in \mathbb{R}^{96}$$

where:
- $h_t^{\text{clin}} \in \mathbb{R}^{64}$ — GRU hidden state on clinical sequence
- $v_t^{\text{MRI}} \in \mathbb{R}^{32}$ — MRI scalar encoder output: $\text{MLP}(\text{nWBV}_t, \text{eTIV}_t, \text{ASF}_t, \Delta\text{nWBV}_t)$
- $\|$ denotes vector concatenation

---

## 2. Temporal Evolution

The twin evolves as the patient accumulates visits:

$$Z_1^{(P)} \to Z_2^{(P)} \to Z_3^{(P)} \to \ldots \to Z_T^{(P)}$$

This trajectory represents the **longitudinal brain state evolution** of patient P.

The GRU architecture ensures that each $Z_t$ is conditioned on all prior information:

$$Z_t = f_\theta(Z_{t-1}, x_t)$$

where $f_\theta$ is the GRU transition function with learned parameters $\theta$.

---

## 3. Prediction from Z_t

Given $Z_t$, the prediction head maps to a CDR class distribution:

$$\hat{y}_{t+1} = \text{softmax}(\text{MLP}(Z_t))$$

$$\hat{y}_{t+1} \in \Delta^3 \subset \mathbb{R}^4 \quad (\text{CDR classes: } 0.0, 0.5, 1.0, 2.0)$$

**Prediction horizon:** 1 visit ahead only (next-visit CDR).  
The inter-visit interval in OASIS-2 averages ~732 days (~2 years).  
We do NOT claim multi-step or continuous-time disease trajectory prediction with this dataset.

---

## 4. What Enters Z_t

### Clinical features (all 19, standardized):

| Feature | Type | Leakage-safe? |
|---|---|---|
| curr_age | Static/slow | ✅ Yes |
| sex, hand, educ, ses | Static demographic | ✅ Yes |
| curr_mmse | Current visit | ✅ Yes |
| curr_cdr | Current visit (INPUT, not target) | ✅ Yes |
| curr_nWBV, curr_eTIV, curr_ASF | Current MRI scalars | ✅ Yes |
| current_mr_delay, days_between_visits | Timing | ✅ Yes |
| n_prior_visits | Visit count | ✅ Yes |
| prev_mmse, prev_cdr, prev_nWBV | Prior visit values | ✅ Yes (only i-1) |
| mmse_delta, cdr_delta, nwbv_delta | Rate of change | ✅ Yes (only i-1) |

**next_CDR** and **next_MMSE** are NEVER included in features. They are targets only.

---

## 5. What Z_t Does NOT Currently Capture

| Modality | Status | Reason |
|---|---|---|
| Raw 3D MRI volumes | ❌ Not captured | No NIfTI files in OASIS-2 Kaggle dataset |
| EEG signals | ❌ Not captured | No EEG data exists for this cohort |
| Genetic/biomarker data | ❌ Not captured | Not available in OASIS-2 |
| Medication/treatment history | ❌ Not captured | Not recorded in OASIS-2 |

These limitations are documented in `DOCS/EEG_LIMITATION.md` and `DOCS/RESEARCH_LIMITATIONS.md`.

---

## 6. How New Observations Update the Twin

When patient P arrives for a new visit $t+1$ with new observations $x_{t+1}$:

$$Z_{t+1} = \text{GRU}(Z_t, x_{t+1})$$

The GRU update step is:

$$r_t = \sigma(W_r x_t + U_r h_{t-1})  \quad \text{(reset gate)}$$
$$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}))$$
$$z_t = \sigma(W_z x_t + U_z h_{t-1})  \quad \text{(update gate)}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

---

## 7. Artifact Evidence

The following artifacts demonstrate actual implementation:

| Artifact | Status |
|---|---|
| `artifacts/EXP-LONGITUDINAL-001/temporal_gru_cpu.pt` | ✅ Trained GRU checkpoint |
| `artifacts/EXP-LONGITUDINAL-001/metrics.json` | ✅ Real test metrics |
| `artifacts/EXP-BRAIN-TWIN-001/per_patient/*.npy` | ✅ Z_t trajectories |
| `artifacts/EXP-BRAIN-TWIN-001/plots/brain_twin_Z_trajectories_PCA.png` | ✅ PCA visualization |
| `artifacts/EXP-BRAIN-TWIN-001/plots/brain_twin_Z_heatmap.png` | ✅ Heatmap visualization |
| `artifacts/EXP-FUSION-BIMODAL-001/metrics.json` | ✅ Bimodal Z_t (96-dim) ablation |

---

## 8. Known Limitations

1. **Z_t is 64-dimensional** — interpretability requires dimensionality reduction (PCA, UMAP).
2. **Clinical-only Z_t** does not encode raw brain imaging — it encodes derived MRI scalars only.
3. **Prediction is 1 visit ahead** — not continuous-time or multi-step.
4. **150 patients, 373 visits** — Z_t representation may not generalize beyond OASIS-2 without retraining.
5. **Z_t is not clinically validated** — this is a research prototype.
