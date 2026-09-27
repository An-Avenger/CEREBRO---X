# Cerebro X — Product and Research Interface Design

**Version:** 1.0  
**Design direction:** Scientific, futuristic, restrained, high-credibility  
**Primary purpose:** Research visualization, not clinical diagnosis

---

# 1. Design Philosophy

Cerebro X should feel like a serious neuroinformatics research platform rather than a generic hospital dashboard.

Visual language:

```text
Scientific
+
Futuristic
+
Minimal
+
Data-dense
+
Readable
```

Avoid:

- excessive neon;
- fake medical holograms;
- decorative brain animations that convey no information;
- "AI magic" language;
- consumer-health styling.

The interface should communicate:

> "There is serious computation happening here."

Not:

> "Someone discovered CSS gradients."

---

# 2. Theme

Primary theme:

**Dark scientific interface**

Background layers:

- near-black base;
- dark slate surfaces;
- slightly lighter cards;
- subtle borders.

Accent system:

- cyan/blue for neutral AI information;
- violet for model/latent-state information;
- amber for warnings/uncertainty;
- red reserved for critical system errors, not disease labels.

The exact color tokens should be centralized and configurable.

---

# 3. Typography

Recommended:

### Primary

Inter

### Monospace / technical

JetBrains Mono

Use monospace sparingly for:

- model IDs;
- experiment IDs;
- dataset IDs;
- technical metadata.

Typography hierarchy:

```text
Page title
  ↓
Section title
  ↓
Metric
  ↓
Supporting text
  ↓
Metadata
```

Avoid tiny text for important scientific values.

---

# 4. Main Application Layout

```text
+------------------------------------------------------------+
| CEREBRO X                         Model: CX-001   Research |
+----------------------+-------------------------------------+
|                      |                                     |
|  OVERVIEW            |         MAIN CONTENT                |
|                      |                                     |
|  PATIENTS            |                                     |
|  BRAIN STATE         |                                     |
|  LONGITUDINAL        |                                     |
|  PREDICTIONS         |                                     |
|  EXPLAINABILITY      |                                     |
|                      |                                     |
|  EXPERIMENTS         |                                     |
|  MODELS              |                                     |
|                      |                                     |
|  DATASETS            |                                     |
|                      |                                     |
+----------------------+-------------------------------------+
```

---

# 5. Overview Screen

Purpose:

Provide a high-level research-system overview.

Components:

### Model status

```text
Model
Cerebro X v0.x
```

### Dataset status

```text
Primary: OASIS-3
External: ADNI
```

### Latest experiment

Show:

- experiment ID;
- model;
- dataset;
- metric;
- timestamp.

### Longitudinal summary

Show aggregate:

- number of research subjects;
- number of visits;
- number of MRI sessions;
- missingness;
- selected target.

Only show values actually computed by the system.

---

# 6. Research Case Screen

A "research case" is the application's abstraction over a participant.

Display:

```text
Research Case
CX-XXXX

Dataset
OASIS-3 / ADNI

Visit timeline
T0 ─── T1 ─── T2 ─── T3
```

Never expose unnecessary identifying information.

---

# 7. Brain State Screen

Main visualization:

```text
             MRI VIEWER
      +-----------------------+
      |                       |
      |     MRI slice         |
      |                       |
      |                       |
      +-----------------------+

Latent Brain State
[ embedding summary ]

State trajectory
T0 ───── T1 ───── T2 ───── T3
```

Possible views:

- axial;
- sagittal;
- coronal;
- selected saliency overlay.

---

# 8. Longitudinal Screen

Primary visualization:

```text
Clinical score
 ^
 | |  |   |   \____
 |
 +------------------------> Time
   T0    T1    T2    T3
```

Additional tracks:

- diagnostic state;
- cognitive score;
- model latent state;
- MRI-derived measurements.

The interface should distinguish:

```text
Observed
```

from:

```text
Predicted
```

clearly.

---

# 9. Prediction Screen

Display:

### Current-state prediction

```text
Predicted state
[model output]
```

### Future prediction

```text
Prediction horizon
12 months

Predicted outcome
[model output]
```

### Confidence / uncertainty

Use uncertainty when the model supports it.

Do not display arbitrary percentages as "confidence" if they are merely raw softmax probabilities without calibration.

---

# 10. Explainability Screen

### MRI explanation

```text
Original MRI
      +
Attribution / saliency overlay
```

### Clinical explanation

```text
Feature             Contribution
----------------------------------
Feature A           +++++
Feature B           +++
Feature C           --
```

The interface should explain:

> "Model attribution"

not:

> "Cause of disease"

---

# 11. Experiment Screen

Display:

```text
Experiment ID
Dataset
Split
Seed
Model
Preprocessing
Metrics
Checkpoint
Git Commit
```

Example:

```text
EXP-00017
Dataset: OASIS-3
Model: CX-Temporal-01
Seed: 42
MRI preprocessing: v0.3
Clinical preprocessing: v0.2
```

---

# 12. Model Registry Screen

Display:

| Model | Version | Dataset | Status |
|---|---|---|---|
| MRI Baseline | v0.1 | OASIS-3 | Complete |
| Clinical Baseline | v0.1 | OASIS-3 | Complete |
| Fusion Baseline | v0.1 | OASIS-3 | Complete |
| Cerebro X | v0.1 | OASIS-3 | Research |

Never label a model "best" unless the evaluation protocol supports that conclusion.

---

# 13. Dataset Screen

Display metadata only.

Example:

```text
OASIS-3
Type: Longitudinal multimodal
Role: Primary
Access: Restricted
Status: Connected / Pending

ADNI
Type: Longitudinal observational
Role: External validation
Access: Approved / Pending
```

Do not expose restricted raw data.

---

# 14. UX Rules

## D-001

Every prediction must display its model version.

## D-002

Every research visualization must indicate whether data are observed or predicted.

## D-003

Uncertainty must not be hidden.

## D-004

No "diagnosis" language for model output.

## D-005

No fake real-time claims.

## D-006

Loading states must explain what is being processed.

## D-007

Errors must be understandable to a researcher.

---

# 15. Accessibility

Minimum:

- sufficient contrast;
- keyboard navigation;
- readable font sizes;
- labels for charts;
- non-color-only indicators;
- screen-reader-friendly controls where practical.

---

# 16. Responsive Design

Primary target:

**Desktop/laptop research workstation**

Secondary:

**Tablet**

Mobile is not a primary target for the research interface.

---

# 17. Visual Identity

Brand:

```text
CEREBRO X
```

Suggested visual motif:

- stylized neural network;
- subtle cortical contour;
- circular state/timeline motif.

Avoid using recognizable copyrighted Marvel/X-Men logos or artwork. The name can have thematic inspiration, but the product should maintain an original research identity.

---

# 18. Design Implementation Rules

Create centralized design tokens:

```text
theme/
  colors
  typography
  spacing
  radii
  shadows
```

Do not hardcode styling throughout components.

Recommended frontend structure:

```text
frontend/
└── src/
    ├── components/
    ├── pages/
    ├── layouts/
    ├── charts/
    ├── brain-viewer/
    ├── services/
    ├── hooks/
    ├── types/
    └── theme/
```

---

# 19. Research Credibility Rule

Every visual element must answer at least one of these:

1. What happened?
2. What is predicted?
3. How confident is the model?
4. What evidence influenced the model?
5. How did the state change over time?
6. Which experiment produced this result?

If it answers none of them, it is probably decoration.

And Cerebro X has enough brains involved already.
