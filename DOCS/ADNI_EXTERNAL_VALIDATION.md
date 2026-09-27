# Cerebro X — ADNI External Validation Protocol

> **Status: NOT IMPLEMENTED**
>
> ADNI is a planned external validation dataset. No ADNI data is present or integrated.
> Do not fabricate ADNI records or blindly merge ADNI rows into OASIS-2 training data.

---

## What ADNI Is

The **Alzheimer's Disease Neuroimaging Initiative (ADNI)** is a longitudinal, multi-center
observational study providing clinical, imaging, genetic, biomarker, and cognitive data.

Official website: https://adni.loni.usc.edu/

Access requires:
1. Approved ADNI account at https://ida.loni.usc.edu/
2. Signed ADNI Data Use Agreement
3. Institutional approval (where required)

---

## Why ADNI Cannot Be Naively Merged with OASIS-2

| Difference | Impact |
|---|---|
| Different recruitment protocols | Cohort selection bias |
| Different MRI acquisition sites and scanners | Domain shift in volumetric features |
| Different cognitive assessment batteries | Some variables map imperfectly |
| Different visit schedules | Longitudinal alignment required |
| Different subject ID spaces | IDs will collide if naively concatenated |
| Different CDR/MMSE administration context | Label distribution may differ |

---

## Planned Variable Mapping (TODO — requires ADNI access to verify)

| OASIS-2 Column | Candidate ADNI Equivalent | Notes |
|---|---|---|
| `CDR` | `CDGLOBAL` or `CDMEMORY` | Must verify exact ADNI column name |
| `MMSE` | `MMSCORE` | Verify |
| `Age` | `AGE` | Likely direct |
| `M/F` | `PTGENDER` | Recode |
| `EDUC` | `PTEDUCAT` | Verify scale |
| `SES` | `PTMARRY` / derived | ADNI does not directly measure SES the same way |
| `eTIV` | Via FreeSurfer: `ICV` | Verify ADNI FreeSurfer output table |
| `nWBV` | Via FreeSurfer: derived | Verify |
| `ASF` | Not standard in ADNI | May not be directly available |

**All mappings are hypothetical until actual ADNI data is inspected.**

---

## External Validation Protocol

When ADNI access is obtained:

1. **Do NOT mix ADNI subjects into OASIS-2 training data.**
2. **Load ADNI as a separate dataset** using a new adapter: `src/cerebro_x/data/adni/`
3. **Audit ADNI variables** — verify CDR, MMSE, volumetric features are present and compatible.
4. **Map variables explicitly** — document every mapping decision.
5. **Apply OASIS-2 trained preprocessor** to ADNI data (do NOT refit on ADNI).
6. **Evaluate OASIS-2 trained model on ADNI** — this measures generalization.
7. **Report:**
   - performance degradation vs. OASIS-2 test set
   - distribution shift in features
   - distribution shift in CDR labels
   - calibration shift

---

## Files to Create (When ADNI Access Is Ready)

```
configs/datasets/adni.yaml
src/cerebro_x/data/adni/
    __init__.py
    loader.py
    audit.py
    variable_map.py
docs/adr/ADR-003-adni-variable-mapping.md
```

---

## Record to Update

After completing ADNI integration, add an entry to `docs/MEMORY.md` and update this document
with actual variable mappings, cohort facts, and evaluation results.
