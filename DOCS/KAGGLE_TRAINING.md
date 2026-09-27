# Cerebro X — Kaggle Training Guide

---

## Dataset

Kaggle dataset slug: `jboysen/mri-and-alzheimers`

Mount path inside Kaggle kernel:
```
/kaggle/input/mri-and-alzheimers/oasis_longitudinal.csv
```

Always discover the path programmatically:
```python
import glob
matches = glob.glob("/kaggle/input/**/oasis_longitudinal.csv", recursive=True)
if not matches:
    raise FileNotFoundError("oasis_longitudinal.csv not found in /kaggle/input. "
                            "Please attach the dataset 'jboysen/mri-and-alzheimers'.")
csv_path = matches[0]
print(f"Found dataset at: {csv_path}")
```

---

## Accelerator Policy

Do NOT use GPU for:
- Pandas loading
- sklearn preprocessing
- sklearn baseline models
- Exploratory analysis

Use GPU only when:
- Training a PyTorch neural model (Phase 4+)
- MRI model training begins (Phase 9+)

For Phase 0–3, set Kaggle accelerator to **None (CPU)** to avoid wasting GPU quota.

---

## Notebook Structure

The Kaggle notebook (`notebooks/01_dataset_audit/oasis2_audit.ipynb`) is organized as:

| Cell Group | Purpose |
|---|---|
| 1. Environment verification | Python version, packages, device check |
| 2. Dataset path discovery | glob search, not hard-coded path |
| 3. Data loading | Load CSV, validate schema |
| 4. Dataset audit | Shape, dtypes, missingness, distributions |
| 5. Longitudinal pair construction | Build next-visit pairs |
| 6. Subject-level splitting | Train/val/test subject lists |
| 7. Preprocessing | Scale, impute (fit on train only) |
| 8. Baseline models | Majority, LR, RF |
| 9. Cross-validation | GroupKFold, report class dist per fold |
| 10. Final test evaluation | ONE time, final test subjects |
| 11. Explainability | Feature importance |
| 12. Artifact export | Save metrics, plots, model |
| 13. Inference example | Show prediction on sample |
| 14. Reproducibility summary | Seeds, versions, paths |

---

## Artifact Export

At the end of the notebook:
```python
import shutil
shutil.make_archive("/kaggle/working/cerebro_x_artifacts", "zip",
                    "/kaggle/working/artifacts")
```
Download from Kaggle → **Output** tab.

---

## Environment Logging

Always log:
```python
import subprocess, sys
print(sys.version)
result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
with open("artifacts/EXP-BASELINE-001/config/environment.txt", "w") as f:
    f.write(result.stdout)
```

---

## Common Kaggle Pitfalls

| Pitfall | Fix |
|---|---|
| Hard-coded path `/kaggle/input/datasets/...` | Use glob discovery |
| Running GPU cells on CPU notebook | Check `torch.cuda.is_available()` |
| Kernel dies on OOM during sklearn RF | Reduce `n_jobs` or `n_estimators` |
| Artifact directory not found | Create it with `os.makedirs(..., exist_ok=True)` |
| Importing from `src/` fails | Add `sys.path.insert(0, "/kaggle/working/cerebro-x/src")` |
