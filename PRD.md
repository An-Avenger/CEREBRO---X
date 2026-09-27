# Cerebro X — Product Requirements Document (PRD)

**Project:** Cerebro X  
**Full Title:** AI-Based Digital Brain Twin for Predicting Neurological Disease Progression Using Longitudinal MRI and Clinical Data  
**Disease Focus:** Alzheimer's disease and related cognitive decline  
**Document Version:** 1.0  
**Status:** Project foundation / Phase 0–1  
**Last Updated:** 2026-08-18

---

## 1. Executive Summary

Cerebro X is a research-oriented AI platform designed to construct a patient-specific computational representation of neurological state from longitudinal brain MRI and clinical/cognitive data.

The first implementation will focus on Alzheimer's disease and related cognitive decline. The system will not be treated as a clinical diagnostic device. Its purpose is research, experimentation, visualization, longitudinal prediction, and explainability.

The central research idea is to move beyond a one-time diagnosis:

> **Baseline and longitudinal observations → patient-specific latent brain state → future cognitive/disease-state prediction → explainable evidence.**

The initial system will use:

1. **OASIS-3** as the primary longitudinal research dataset.
2. **ADNI** as an independently sourced dataset for validation/transferability experiments where access and compatible variables permit.

The datasets will **not be naively merged row-by-row**. Dataset identity will remain explicit, and cross-dataset evaluation will be treated as a separate experiment.

EEG is **not part of the initial data contract** because OASIS-3 and ADNI do not provide a guaranteed, synchronized MRI+EEG cohort suitable for the proposed longitudinal patient-level fusion. EEG may be added later as a separate extension if a compatible dataset is identified.

---

## 2. Problem Statement

Traditional machine-learning approaches to Alzheimer's disease often frame the task as a static classification problem:

`MRI → CN/MCI/AD`

Cerebro X instead investigates whether longitudinal multimodal observations can be represented as a dynamic patient-specific state that is useful for predicting future cognitive or diagnostic outcomes.

### Core problem

Given a patient's MRI and clinical/cognitive observations at one or more time points:

- estimate the current neurological state;
- predict a future cognitive or disease-state outcome;
- quantify longitudinal change;
- identify which input features or regions contribute to the prediction;
- evaluate whether the learned representation generalizes across independent datasets.

---

## 3. Research Objectives

### O1 — Current-state estimation

Learn a representation of the patient's current neurological state from structural MRI and selected clinical/cognitive variables.

### O2 — Diagnostic-state prediction

Evaluate prediction of clinically defined categories such as cognitively normal (CN), mild cognitive impairment (MCI), and Alzheimer's disease/dementia, subject to the exact labels available in the selected dataset.

### O3 — Longitudinal progression prediction

Predict a future clinical/cognitive outcome from information available at an earlier visit.

Possible targets include:

- future MMSE score or change;
- future CDR/CDR-SB or change;
- future diagnostic state;
- conversion/progression events.

The exact target and prediction horizon must be selected only after inspecting the actual released variables and visit coverage.

### O4 — Patient-specific brain state

Construct a latent representation `Z_t` for each patient at time `t`.

Conceptually:

`Z_t = f(MRI_t, Clinical_t, History_≤t)`

### O5 — Explainability

Provide evidence for model predictions using appropriate techniques for the model and modality, including MRI saliency/attention methods and feature-level attribution for tabular clinical inputs.

### O6 — Generalization

Test whether the model trained on the primary dataset retains useful performance on an independently sourced dataset without assuming that the two datasets have identical acquisition protocols or distributions.

---

## 4. Target Users

Cerebro X is intended for:

- M.Tech/research students;
- academic supervisors;
- neuroscience/neuroinformatics researchers;
- machine-learning researchers;
- reviewers evaluating the methodology;
- developers operating the research prototype.

It is **not** intended for unsupervised clinical use by patients or for medical decision-making.

---

## 5. Dataset Strategy

### 5.1 Primary dataset — OASIS-3

OASIS-3 is a longitudinal multimodal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer's disease.

The official OASIS documentation describes:

- 1378 participants;
- 2842 MR sessions;
- longitudinal data collected across multiple projects;
- T1w, T2w, FLAIR, ASL, SWI, TOF, resting-state BOLD and DTI MRI sequences;
- clinical and cognitive/psychometric files;
- FreeSurfer-derived outputs for many MR sessions;
- PET and other imaging data.

Cerebro X will initially prioritize **structural T1-weighted MRI + clinical/cognitive data**.

OASIS-3 access requires a request and NITRC account. The project must comply with OASIS data-use terms.

### 5.2 Secondary / external dataset — ADNI

ADNI is a longitudinal, multi-center observational study containing clinical, imaging, genetic, biomarker and cognitive information.

The official ADNI documentation confirms availability of:

- structural MRI and other MRI modalities;
- clinical and cognitive assessments;
- demographic information;
- longitudinal visits;
- MRI-derived numerical outputs;
- multiple ADNI phases and acquisition protocols.

ADNI data are accessed through the LONI Image and Data Archive and require approved access and compliance with the ADNI Data Use Agreement.

ADNI will initially be treated as an **external validation / transferability dataset**, not as a blindly concatenated training source.

### 5.3 EEG

EEG is intentionally deferred from Cerebro X v1.

Reason:

- the primary OASIS-3 and ADNI strategy is based on longitudinal MRI + clinical/cognitive data;
- a valid multimodal EEG experiment requires patient-level modality correspondence and compatible visit timing;
- an unrelated EEG dataset must not be artificially paired with MRI patients.

EEG becomes a future research branch only after a defensible dataset and matching strategy are established.

---

## 6. Functional Requirements

### FR-01 — Dataset ingestion

The system shall support controlled ingestion of OASIS-3 and ADNI-derived research data.

### FR-02 — Data provenance

Every processed record shall retain provenance fields sufficient to identify:

- dataset;
- subject identifier/hash;
- visit/session;
- acquisition information where available;
- preprocessing version;
- source file or accession reference where permitted.

### FR-03 — Longitudinal alignment

The system shall align MRI and clinical observations using explicit visit/time rules.

The matching tolerance shall be configurable and documented.

### FR-04 — MRI preprocessing

The pipeline shall support:

- format conversion where necessary;
- orientation/metadata normalization;
- quality-control checks;
- resampling;
- intensity normalization;
- optional skull stripping/brain extraction;
- spatial standardization;
- generation of model-ready tensors.

Preprocessing choices must be recorded as experiment configuration.

### FR-05 — Clinical preprocessing

The pipeline shall support:

- missing-value analysis;
- appropriate imputation or missingness indicators;
- normalization/standardization;
- categorical encoding;
- longitudinal feature construction;
- leakage checks.

### FR-06 — Baseline models

The system shall implement unimodal and simple baselines before advanced fusion.

Minimum baseline set:

1. clinical-only;
2. MRI-only;
3. MRI + clinical fusion;
4. longitudinal baseline.

### FR-07 — Cerebro X model

The main research model shall learn a patient-specific latent representation and support future-state prediction.

### FR-08 — Explainability

The system shall provide model-specific explanations without presenting them as causal evidence.

### FR-09 — Evaluation

The system shall report appropriate metrics for the selected prediction task and include confidence intervals where practical.

### FR-10 — Reproducibility

Experiments shall record:

- random seed;
- dataset release/download date;
- preprocessing configuration;
- model configuration;
- training configuration;
- evaluation split;
- software environment;
- model checkpoint identifier.

### FR-11 — Web research interface

A later application layer shall provide:

- patient/research-case selection;
- current-state visualization;
- longitudinal timeline;
- prediction results;
- model explanation;
- experiment metadata.

The UI must clearly indicate that Cerebro X is a research prototype.

---

## 7. Non-Functional Requirements

### NFR-01 — Reproducibility

A complete experiment should be reproducible from versioned configuration and documented dataset access.

### NFR-02 — Privacy

No attempt shall be made to identify research participants.

### NFR-03 — Data security

Raw restricted-access datasets shall not be committed to Git or distributed through the application repository.

### NFR-04 — Modularity

Dataset adapters, preprocessing, models, evaluation, explainability and API layers shall remain independently testable.

### NFR-05 — Scalability

The architecture should support local CPU development and GPU training on approved cloud environments.

---

## 8. Research Constraints

1. OASIS-3 and ADNI are different studies with different cohorts, protocols and distributions.
2. Dataset shift must be measured rather than ignored.
3. Longitudinal leakage is a major risk. Patient-level separation must be enforced.
4. Future information must never be used to construct baseline features.
5. Model explanations are not causal explanations.
6. Clinical labels must be taken from documented dataset variables, not invented mappings.
7. The final prediction target must be locked only after inspecting actual dataset availability.
8. Restricted-access datasets must not be redistributed.

---

## 9. Success Criteria

Cerebro X v1 is considered successful when it can:

- ingest a documented subset of OASIS-3;
- create a clean longitudinal patient table;
- pair MRI and clinical observations using an explicit rule;
- train reproducible MRI-only and clinical-only baselines;
- train an MRI + clinical model;
- evaluate a longitudinal prediction target;
- produce patient-specific latent representations;
- generate interpretable model evidence;
- evaluate on an independent ADNI cohort when variable compatibility permits;
- expose results through a research-oriented API.

---

## 10. Out of Scope for v1

- clinical diagnosis;
- treatment recommendation;
- autonomous medical decision-making;
- fabricated MRI/EEG pairing;
- direct patient identification;
- deployment as a medical device;
- claiming that the latent vector is a biological digital replica of the brain;
- claiming causal explanations from saliency maps.

---

## 11. Research Questions

### RQ1
Does combining MRI and clinical information improve prediction over either modality alone?

### RQ2
Does longitudinal information improve future-state prediction over single-visit classification?

### RQ3
Can a patient-specific latent state represent clinically meaningful variation across visits?

### RQ4
How robust is the model under cross-dataset evaluation?

### RQ5
Which MRI regions and clinical variables contribute most strongly to predictions?

---

## 12. Initial Hypotheses

**H1:** MRI + clinical fusion will outperform at least one unimodal baseline on the selected prediction task.

**H2:** Longitudinal modeling will improve future-outcome prediction compared with a single-visit baseline.

**H3:** Patient-specific latent representations will contain information predictive of future clinical/cognitive state.

**H4:** Cross-dataset performance will decrease relative to in-dataset testing because of cohort and acquisition shift, and explicit domain-shift analysis will help characterize that degradation.

---

## 13. Medical/Research Disclaimer

Cerebro X is an academic research system. Its outputs are predictions generated by machine-learning models and must not be interpreted as a medical diagnosis, prognosis, treatment recommendation, or substitute for qualified clinical judgment.

---

## 14. Official Dataset References

### OASIS-3

Official project page:
https://sites.wustl.edu/oasisbrains/home/oasis-3/

Official access page:
https://sites.wustl.edu/oasisbrains/request-access/

Official FAQ:
https://sites.wustl.edu/oasisbrains/home/oasis-resources-and-faq/

### ADNI

Official project page:
https://adni.loni.usc.edu/

Official data page:
https://adni.loni.usc.edu/data-samples/adni-data/

Official MRI data page:
https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/

---

## 15. Decision Gate Before Model Development

Before implementing the final prediction model, the following must be completed and documented:

- [ ] Obtain approved access to OASIS-3.
- [ ] Obtain approved access to ADNI if required.
- [ ] Inspect the exact released tables/files.
- [ ] Identify subject, visit and scan identifiers.
- [ ] Determine available diagnostic labels.
- [ ] Determine available longitudinal cognitive variables.
- [ ] Determine MRI sequences and quality constraints.
- [ ] Define MRI↔clinical visit matching rule.
- [ ] Select the prediction target.
- [ ] Select the prediction horizon.
- [ ] Define patient-level train/validation/test splits.
- [ ] Freeze the preprocessing specification.

Only after these decisions are locked should the main model architecture be finalized.
