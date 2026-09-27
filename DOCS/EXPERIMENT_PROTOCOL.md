# Cerebro X — Experiment Protocol

Every experiment must follow this protocol. Deviating without documentation is a protocol violation.

---

## 1. Before Running Any Experiment

### 1.1 Register the experiment

Create a config file in `configs/experiments/`:

```
configs/experiments/exp_<name>_<nnn>.yaml
```

The config must contain ALL of:

- `experiment_id` — unique, never reused
- `description`
- `dataset` and `dataset_version_note`
- `seed`
- `split` — strategy, sizes, group column
- `target` — name, type, classes, note
- `features` — which groups are active
- `models` — list with params
- `metrics` — primary metric declared BEFORE running
- `artifacts` — what to save

### 1.2 Verify the dataset

The dataset must be present and pass schema validation before any model runs:

```bash
python scripts/audit_oasis2.py --config configs/datasets/oasis2_kaggle.yaml
```

### 1.3 Build longitudinal pairs

```bash
python scripts/build_pairs.py --config configs/base.yaml
```

Inspect `data/processed/next_visit_pairs.csv` before continuing.

---

## 2. Running the Experiment

```bash
python scripts/train_baselines.py --experiment configs/experiments/exp_baseline_001.yaml
```

The script must:
1. Load config
2. Set all random seeds
3. Load and validate dataset
4. Build features
5. Build splits (subject-level)
6. Run cross-validation
7. Select best model
8. Evaluate on final test set ONCE
9. Save all artifacts

**The final test set must be used exactly once.**

---

## 3. Required Artifacts

After running, `artifacts/<experiment_id>/` must contain:

```
model/
    model.pkl           (or model.pt for neural models)
    preprocessor.pkl
    feature_schema.json
    label_mapping.json

metrics/
    cv_metrics.json
    test_metrics.json
    predictions.csv

plots/
    confusion_matrix.png
    feature_importance.png

config/
    experiment.yaml     (copy of the config used)
    environment.txt     (pip freeze output)

README.md               (auto-generated summary)
```

If any file is missing, the experiment is INCOMPLETE.

---

## 4. Metrics Requirements

### Classification (next-visit CDR)

Must report ALL of:
- accuracy
- balanced accuracy (primary)
- macro precision, recall, F1
- weighted F1
- Cohen's kappa
- mean absolute error (ordinal-sensitive)
- confusion matrix
- per-class precision/recall/F1

Must NOT report:
- Only accuracy
- Fabricated confidence intervals
- AUC if the task formulation does not support it

### Regression (next-visit MMSE, secondary)

Must report ALL of:
- MAE
- RMSE
- R²

---

## 5. Anti-Patterns (Forbidden)

| Anti-pattern | Consequence |
|---|---|
| Cherry-picking the primary metric AFTER seeing results | Protocol violation |
| Using the test set during model selection | Data leakage |
| Splitting rows from the same subject into different folds | Longitudinal leakage |
| Fitting scaler/imputer on all data (including test) | Data leakage |
| Reporting accuracy without balanced accuracy on imbalanced data | Misleading result |
| Overwriting a previous experiment's artifacts | Loss of reproducibility |
| Claiming training ran when it did not | Scientific misconduct |

---

## 6. Recording in MEMORY.md

After the experiment completes, add an entry to `docs/MEMORY.md`:

```markdown
## YYYY-MM-DD — EXP-XXX — <description>
- Config: configs/experiments/exp_xxx.yaml
- Dataset: oasis2_kaggle, 373 rows, 150 subjects
- Split: 128 train subjects / 22 val subjects (from CV) / 22 test subjects
- Pairs used: 223
- Models run: [list]
- Primary metric (balanced accuracy): [actual values]
- Test result: [actual values]
- Artifacts: artifacts/EXP-XXX/
- Issues: [any unresolved]
```

Only record ACTUAL results. Never record aspirational results.
