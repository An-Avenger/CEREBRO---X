# ADR-002 — Prediction Target: Next-Visit CDR

**Date:** 2026-08-18  
**Status:** Accepted (provisional — must be confirmed after dataset audit)  
**Deciders:** Aryan Sharma (M.Tech student)

---

## Context

Cerebro X requires a concrete prediction target before model development can begin.
The Master Prompt (Phase 1.1) lists five candidate targets:

- A: Future MMSE score
- B: MMSE change
- C: Future CDR / CDR-SB
- D: Future diagnostic state (CN/MCI/AD)
- E: Conversion event (MCI → AD)

---

## Decision

**Primary target: next-visit CDR (Clinical Dementia Rating)**

CDR is treated as an ordinal multiclass problem with classes: {0.0, 0.5, 1.0, 2.0}.

**Secondary target (optional, regression): next-visit MMSE**

---

## Rationale

| Criterion | CDR | Group (Nondemented/Demented/Converted) | MMSE |
|---|---|---|---|
| Available in verified CSV | ✅ | ✅ | ✅ |
| Clinically meaningful | ✅ | Moderate | ✅ |
| Longitudinal coverage | ✅ | ✅ | ✅ (2 missing) |
| Leakage risk | Low if handled correctly | Low | Low |
| Ordinal structure | ✅ (enables ordinal metrics) | ❌ | Continuous |
| Research precedent | Strong (CDR widely used) | Present | Strong |
| Subjects with changing target | 34 of 150 | Present | Not audited yet |

CDR is the most commonly used severity staging tool in Alzheimer's research.
Using CDR as the target aligns with clinical practice and research norms.

The `Group` column (Nondemented/Demented/Converted) is a different variable —
"Converted" mixes visit-level and subject-level semantics and requires additional
care to use without leakage. It is deferred to a secondary experiment.

---

## Leakage Risk and Mitigation

**Risk:** CDR at visit T+1 (the target) must never appear in the feature matrix for visit T.

**Mitigation implemented in `build_longitudinal_pairs.py`:**
- Input features come exclusively from visit T rows (and earlier visits for temporal features).
- `next_CDR` is the CDR value from visit T+1, added as a separate target column.
- A test `test_next_cdr_not_in_input_features` explicitly verifies this.

---

## Consequences

- Evaluation must use ordinal-sensitive metrics: MAE (CDR), quadratic weighted kappa.
- CDR = 2.0 has only 3 records — per-class metrics for this class are unreliable.
- Balanced accuracy is preferred over raw accuracy.
- If CDR baseline results are poor, Group-based classification or MMSE regression may be explored as alternatives.

---

## Review Trigger

This decision must be reviewed if:
- The dataset audit reveals CDR is too sparse for the selected subjects.
- The split results in CDR = 2.0 being absent from validation/test splits.
- A stronger research argument for MMSE regression or Group classification is made.
