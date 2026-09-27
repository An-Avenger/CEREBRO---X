#!/usr/bin/env python
"""
scripts/train_eeg_encoder.py
------------------------------
Phase 4: Train EEGNet encoder on ds004504 EEG band-power features.

This is a STANDALONE encoder trained on the Miltiadous et al. (2023) dataset.
It is evaluated independently. It CANNOT be patient-level fused with OASIS-2
because the patient populations are completely different cohorts.

Task: 3-class classification — AD vs FTD vs Control
Additional: Binary classification — Disease vs Control (AD+FTD vs Control)

Architecture: MLP on extracted spectral band-power features.
Note: EEGNet (CNN) would operate on raw signals. Here we use the pre-extracted
      spectral features to avoid requiring raw signal processing in the training
      loop. The EEGNet architecture in eeg_net.py is preserved for raw signal use.

Saves to: artifacts/EXP-EEG-STANDALONE-001/

Usage:
    python scripts/train_eeg_encoder.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cerebro_x.utils.io import setup_logging, make_artifact_dir

logger = logging.getLogger("train_eeg_encoder")

FEATURES_CSV = Path("data/processed/eeg_features.csv")
SEED = 42
EPOCHS = 80
PATIENCE = 15
LR = 1e-3
N_FOLDS = 5

# Label mappings
GROUP_3CLASS = {0: "Alzheimer's", 1: "FTD", 2: "Control"}
GROUP_BINARY = {0: "Disease (AD+FTD)", 1: "Control"}


class EEGSpectralEncoder(nn.Module):
    """MLP encoder for EEG band-power features.

    Input: spectral features (global bands + per-channel + ratios)
    Output: embedding vector → classification head

    This is the feature-based variant of EEGNet for use with pre-extracted
    spectral features. The raw-signal CNN EEGNet is in models/deep/eeg_net.py.
    """

    def __init__(self, input_dim: int, embed_dim: int = 64, num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def load_features(csv_path: Path):
    """Load extracted EEG features, dropping metadata columns."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d subjects, %d columns", len(df), len(df.columns))

    # Metadata columns to exclude from features
    meta_cols = ["subject_id", "group", "group_name", "age", "gender", "mmse",
                 "n_channels", "sfreq", "duration_s"]

    feature_cols = [c for c in df.columns if c not in meta_cols]
    X_raw = df[feature_cols].values.astype(float)
    y = df["group"].values.astype(int)

    return X_raw, y, feature_cols, df


def preprocess(X_train_raw, X_test_raw):
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train_raw)
    X_test_imp = imp.transform(X_test_raw)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train_imp)
    X_test = sc.transform(X_test_imp)

    return X_train, X_test, imp, sc


def run_training(X_train, y_train, X_test, y_test, num_classes, device, artifact_dir, tag):
    """Train a single model and return metrics."""
    cw_counts = np.bincount(y_train, minlength=num_classes).astype(float)
    cw_counts[cw_counts == 0] = 1.0
    cw = torch.tensor(len(y_train) / (num_classes * cw_counts), dtype=torch.float32).to(device)

    model = EEGSpectralEncoder(X_train.shape[1], embed_dim=64, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)

    best_val_loss, best_state, pat = float("inf"), None, 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl, tn = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(yb)
            tn += len(yb)
        train_loss = tl / max(tn, 1)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                vl += criterion(model(Xb.to(device)), yb.to(device)).item() * len(yb)
                vn += len(yb)
        val_loss = vl / max(vn, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info("[%s] Epoch %02d | Train: %.4f | Val: %.4f", tag, epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                logger.info("[%s] Early stopping at epoch %d.", tag, epoch)
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds = model(Xb.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    metrics = {
        "accuracy": float(accuracy_score(all_true, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_preds)),
        "f1_macro": float(f1_score(all_true, all_preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(all_true, all_preds, average="weighted", zero_division=0)),
        "n_test": len(all_true),
        "n_train": len(y_train),
    }

    # Save checkpoint
    torch.save(model.state_dict(), artifact_dir / f"{tag}_encoder.pt")

    # Loss curve
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="Train Loss")
        ax.plot(val_losses, label="Val Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"EEG Encoder Training — {tag}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(artifact_dir / f"{tag}_loss_curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        logger.warning("Loss curve failed: %s", e)

    # Confusion matrix
    try:
        import seaborn as sns
        labels = list(GROUP_3CLASS.values()) if num_classes == 3 else list(GROUP_BINARY.values())
        cm = confusion_matrix(all_true, all_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels[:num_classes],
                    yticklabels=labels[:num_classes], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"EEG — {tag} — Confusion Matrix")
        fig.tight_layout()
        fig.savefig(artifact_dir / f"{tag}_confusion_matrix.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        logger.warning("Confusion matrix failed: %s", e)

    return metrics, model


def main():
    setup_logging("INFO")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    artifact_dir = make_artifact_dir("artifacts", "EXP-EEG-STANDALONE-001")
    logger.info("Artifact dir: %s", artifact_dir)
    device = torch.device("cpu")

    if not FEATURES_CSV.exists():
        logger.error("EEG features not found at %s", FEATURES_CSV)
        logger.error("Run: python scripts/extract_eeg_features.py")
        sys.exit(1)

    X_raw, y, feature_cols, df = load_features(FEATURES_CSV)
    logger.info("Feature shape: %s | Classes: %s",
                X_raw.shape, dict(zip(*np.unique(y, return_counts=True))))

    # ─── 3-Class: AD vs FTD vs Control ───────────────────────────────────────
    logger.info("=== TASK 1: 3-Class (AD vs FTD vs Control) ===")
    from sklearn.model_selection import train_test_split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_test, imp3, sc3 = preprocess(X_train_raw, X_test_raw)

    metrics_3class, model_3class = run_training(
        X_train, y_train, X_test, y_test,
        num_classes=3, device=device, artifact_dir=artifact_dir, tag="3class"
    )

    # ─── Binary: Disease (AD+FTD) vs Control ─────────────────────────────────
    logger.info("=== TASK 2: Binary (Disease vs Control) ===")
    y_bin = (y < 2).astype(int)  # 0=AD, 1=FTD → "disease"=1, 2=Control → "healthy"=0
    y_bin_mapped = np.where(y < 2, 0, 1)  # 0=disease, 1=control

    X_tr_raw_b, X_te_raw_b, yb_train, yb_test = train_test_split(
        X_raw, y_bin_mapped, test_size=0.2, random_state=SEED, stratify=y_bin_mapped
    )
    X_tr_b, X_te_b, imp2, sc2 = preprocess(X_tr_raw_b, X_te_raw_b)

    metrics_binary, model_binary = run_training(
        X_tr_b, yb_train, X_te_b, yb_test,
        num_classes=2, device=device, artifact_dir=artifact_dir, tag="binary"
    )

    # ─── Save results ─────────────────────────────────────────────────────────
    import joblib
    joblib.dump(imp3, artifact_dir / "3class_imputer.pkl")
    joblib.dump(sc3,  artifact_dir / "3class_scaler.pkl")
    joblib.dump(imp2, artifact_dir / "binary_imputer.pkl")
    joblib.dump(sc2,  artifact_dir / "binary_scaler.pkl")

    results = {
        "experiment_id": "EXP-EEG-STANDALONE-001",
        "phase": 4,
        "dataset": "OpenNeuro ds004504 — Miltiadous et al. (2023)",
        "citation": (
            "Miltiadous A, et al. (2023). A Dataset of Scalp EEG Recordings of "
            "Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects. "
            "Data, 8(6), 95."
        ),
        "subjects": int(len(df)),
        "eeg_features": feature_cols[:10],  # first 10 for brevity
        "n_features_total": len(feature_cols),
        "critical_limitation": (
            "STANDALONE ENCODER ONLY. This model is trained on a DIFFERENT PATIENT COHORT "
            "than OASIS-2. Patient-level fusion with OASIS-2 clinical data is NOT possible "
            "without a dataset that has both EEG and OASIS-2 clinical records for the same "
            "patients. See DOCS/EEG_LIMITATION.md for full explanation."
        ),
        "3class_AD_FTD_Control": metrics_3class,
        "binary_disease_vs_control": metrics_binary,
    }

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 70)
    logger.info("PHASE 4 — EEG STANDALONE ENCODER RESULTS")
    logger.info("Dataset: OpenNeuro ds004504 (%d subjects)", len(df))
    logger.info("=" * 70)
    logger.info("TASK 1 — 3-Class (AD vs FTD vs Control):")
    logger.info("  Accuracy:          %.1f%%", metrics_3class["accuracy"] * 100)
    logger.info("  Balanced Accuracy: %.1f%%", metrics_3class["balanced_accuracy"] * 100)
    logger.info("  F1 Macro:          %.3f", metrics_3class["f1_macro"])
    logger.info("")
    logger.info("TASK 2 — Binary (Disease vs Control):")
    logger.info("  Accuracy:          %.1f%%", metrics_binary["accuracy"] * 100)
    logger.info("  Balanced Accuracy: %.1f%%", metrics_binary["balanced_accuracy"] * 100)
    logger.info("  F1 Macro:          %.3f", metrics_binary["f1_macro"])
    logger.info("=" * 70)
    logger.info("IMPORTANT: These results are from a STANDALONE EEG encoder,")
    logger.info("           NOT fused with OASIS-2 clinical data.")
    logger.info("Artifacts: %s", artifact_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
