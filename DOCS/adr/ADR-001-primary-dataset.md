# ADR-001 — Primary Dataset Selection

**Date:** 2026-08-18  
**Status:** Accepted  
**Deciders:** Aryan Sharma (M.Tech student)

---

## Context

Cerebro X originally specified OASIS-3 as the primary dataset (see `PRD.md` and `Architecture.md`,
written during Phase 0 planning). OASIS-3 is a large, rich multimodal dataset requiring an
institutional access request and downloading tens of gigabytes of MRI data.

Before model development can begin, a verified, accessible dataset must be selected for the
initial baseline pipeline.

---

## Decision

**The current primary dataset for Cerebro X Phase 0–3 is the OASIS-2 longitudinal metadata
available on Kaggle** (`jboysen/mri-and-alzheimers`, file: `oasis_longitudinal.csv`).

This is a CSV file containing clinical and MRI-derived scalar data for 150 longitudinal
subjects (373 visit records, 15 columns).

---

## Rationale

| Factor | OASIS-3 | OASIS-2 Kaggle CSV |
|---|---|---|
| Availability right now | Requires access request + large download | Immediately available on Kaggle |
| Format | Raw NIfTI MRI + clinical tables | CSV only |
| Size | ~10+ GB imaging | ~50 KB CSV |
| Subjects | 1378 | 150 |
| MRI images | Yes | No (derived scalars only) |
| Suitable for Phase 0-1 baselines | Overkill before data audit | Appropriate |

The OASIS-2 CSV allows the full longitudinal tabular pipeline to be built, tested,
and evaluated before committing to the larger dataset infrastructure.

---

## Consequences

- The existing `PRD.md` and `Architecture.md` describe OASIS-3. They are kept for reference
  but describe the long-term vision, not the current implementation.
- All code in `src/cerebro_x/data/oasis2/` is OASIS-2 specific.
- When OASIS-3 access is granted, a new adapter `src/cerebro_x/data/oasis3/` will be created.
- ADNI remains the planned external validation dataset, unchanged.

---

## What This Is NOT

This decision does NOT mean OASIS-3 is abandoned. It means:

> Build the longitudinal pipeline with the currently available verified dataset,
> then migrate to OASIS-3 when it is downloaded and verified.
