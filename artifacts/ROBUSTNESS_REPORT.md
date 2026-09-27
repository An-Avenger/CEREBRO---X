# Cerebro-X — Phase 11 Robustness Report
*Generated: 2026-08-27 17:24*

---

## Summary

| Metric | Value |
|---|---|
| Total tests run | 39 |
| Passed | 39 |
| Failed | 0 |
| Overall status | [PASS] ALL PASS |
| Test file | `tests/test_robustness.py` |
| Runtime | `======================= 39 passed, 5 warnings in 1.48s ========================` |

---

## Scenario Results

| ID | Scenario | Tests | Result |
|---|---|---|---|
| R-001 | Missing Clinical Values (NaN Injection) | 4/4 | [PASS] |
| R-002 | Incomplete Longitudinal Visits (Single Visit) | 4/4 | [PASS] |
| R-003 | Corrupted / Zero MRI Scalars | 4/4 | [PASS] |
| R-004 | Extreme Outlier MRI Feature Values | 5/5 | [PASS] |
| R-005 | Class Imbalance Simulation | 6/6 | [PASS] |
| R-006 | Temporal Gaps (Long Sequences & Large Time Jumps) | 4/4 | [PASS] |
| R-007 | Dataset / Distribution Shift | 4/4 | [PASS] |
| R-008 | Random Seed Sensitivity / Determinism | 3/3 | [PASS] |

---

## Scenario Details

### R-001 -- Missing Clinical Values (NaN Injection)

**Status:** [PASS]  
**Tests:** 4/4 passed

**Description:** Injects NaN and None values into individual visit fields to confirm the feature builder's `fillna(0.0)` path handles missing data gracefully without producing NaN tensors or model crashes.

### R-002 -- Incomplete Longitudinal Visits (Single Visit)

**Status:** [PASS]  
**Tests:** 4/4 passed

**Description:** Feeds sequences of length 1 — a patient with only one recorded visit. The GRU must handle sequences shorter than training data without error. Tests both clinical and bimodal models.

### R-003 -- Corrupted / Zero MRI Scalars

**Status:** [PASS]  
**Tests:** 4/4 passed

**Description:** Passes zero-filled and negative MRI scalar values to simulate a failed acquisition or missing MRI session. The bimodal model must produce valid probability distributions even on degenerate input.

### R-004 -- Extreme Outlier MRI Feature Values

**Status:** [PASS]  
**Tests:** 5/5 passed

**Description:** Sends physiologically impossible values (eTIV=9999, nWBV=0.4, age=120) to verify that z-score normalization produces finite tensors and the model returns structurally valid predictions despite out-of-distribution input.

### R-005 -- Class Imbalance Simulation

**Status:** [PASS]  
**Tests:** 6/6 passed

**Description:** Feeds sequences where all visits share the same CDR value, simulating highly imbalanced or homogeneous patient cohorts. Tests all four CDR classes (0.0, 0.5, 1.0, 2.0) independently.

### R-006 -- Temporal Gaps (Long Sequences & Large Time Jumps)

**Status:** [PASS]  
**Tests:** 4/4 passed

**Description:** Tests long visit sequences (5–7 visits) and extreme time gaps between visits (20-year age jump). Also validates sequences with zero time gap. The GRU packed-sequence mechanism must handle all lengths cleanly.

### R-007 -- Dataset / Distribution Shift

**Status:** [PASS]  
**Tests:** 4/4 passed

**Description:** Scales all feature values by 2x or 0.5x to simulate a different scanner or acquisition protocol (domain shift). The model must return valid predictions, and tensor shape must remain (1, T, 19) regardless of scale.

### R-008 -- Random Seed Sensitivity / Determinism

**Status:** [PASS]  
**Tests:** 3/3 passed

**Description:** Verifies that inference is fully deterministic: the same input always produces identical class probabilities (to 1e-6 precision) across repeated calls when the model is in eval() mode with no_grad().

---

## Key Findings

1. **Missing values handled safely.** The `fillna(0.0)` path in `visits_to_feature_tensor`
   prevents NaN propagation into the feature tensor. All NaN/None injection tests pass.

2. **Single-visit inference works.** The packed-sequence GRU correctly handles sequences
   of length 1 via `lengths.clamp(min=1)` — critical for first-visit patients.

3. **Zero/corrupted MRI scalars produce valid outputs.** After z-score normalization,
   zero-filled MRI values become finite negative scalars, not NaN or Inf.

4. **Outlier features do not crash inference.** Extreme values (age=120, eTIV=9999)
   produce large but finite normalized values. The GRU remains numerically stable.

5. **Inference is fully deterministic.** With `model.eval()` + `torch.no_grad()`,
   repeated identical calls produce bit-identical probability vectors (within 1e-6).

6. **Distribution shift degrades gracefully.** 2x/0.5x scaled features produce valid
   predictions — though accuracy is not guaranteed on OOD data, no crashes occur.

---

## Limitations

- Robustness tests use **untrained (randomly initialized) models**, not production
  checkpoints, to isolate structural/numerical robustness from accuracy concerns.
- Accuracy under dataset shift (R-007) was not evaluated — this would require a
  held-out external cohort (e.g., ADNI) with known labels.
- Class imbalance tests (R-005) test inference robustness, not calibration. A
  calibration analysis under imbalance would require labelled test data.

---

## Conclusion

All 39 robustness tests pass. The Cerebro-X inference pipeline is numerically
stable, crash-free, and deterministic across all 8 adversarial stress scenarios
defined in Phase 11. The system is ready for Phase 13 (Deployment).

---

*Cerebro-X Phase 11 — M.Tech Research Project*
