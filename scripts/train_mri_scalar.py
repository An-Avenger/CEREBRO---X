#!/usr/bin/env python
"""
scripts/train_mri_scalar.py
---------------------------
Phase 3: Train MRI Scalar Branch independently.

Trains MRIScalarClassifier on MRI-derived scalar features:
    curr_nwbv, curr_etiv, curr_asf, nwbv_delta

This is the standalone MRI branch experiment.
Saves to: artifacts/EXP-MRI-SCALAR-001/

Usage:
    python scripts/train_mri_scalar.py
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    mean_absolute_error, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.oasis2.loader import load_oasis2_auto
from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.models.deep.mri_scalar import MRIScalarClassifier, MRI_SCALAR_FEATURES
from cerebro_x.utils.io import setup_logging, make_artifact_dir

logger = logging.getLogger("train_mri_scalar")

CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
CDR_INV = {v: k for k, v in CDR_MAP.items()}
NUM_CLASSES = 4
SEED = 42


def prepare_mri_data(df: pd.DataFrame, imputer=None, scaler=None, is_train=True):
    """Extract and preprocess MRI scalar features."""
    # Verify features present
    missing = [f for f in MRI_SCALAR_FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"MRI features missing from DataFrame: {missing}")

    X_raw = df[MRI_SCALAR_FEATURES].values.astype(float)
    y_raw = df["next_CDR"].map(CDR_MAP).values.astype(int)

    if is_train:
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X_raw)
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X_imp)
        return X_scaled, y_raw, imp, sc
    else:
        X_imp = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imp)
        return X_scaled, y_raw


def get_class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    weights = len(y) / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_n += len(y_batch)

    return total_loss / max(total_n, 1)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    y_true_cdr = np.array([CDR_INV[int(p)] for p in all_true])
    y_pred_cdr = np.array([CDR_INV[int(p)] for p in all_preds])

    metrics = {
        "accuracy": float(accuracy_score(all_true, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_preds)),
        "f1_macro": float(f1_score(all_true, all_preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(all_true, all_preds, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_true_cdr, y_pred_cdr)),
    }
    return metrics, np.array(all_true), np.array(all_preds)


def main():
    setup_logging("INFO")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    artifact_dir = make_artifact_dir("artifacts", "EXP-MRI-SCALAR-001")
    logger.info("Artifact dir: %s", artifact_dir)
    device = torch.device("cpu")

    # ── Load data ─────────────────────────────────────────────────────────────
    pairs_path = Path("data/processed/next_visit_pairs.csv")
    if not pairs_path.exists():
        logger.info("Building pairs from raw OASIS-2...")
        raw_df = load_oasis2_auto()
        pairs_df, _ = build_next_visit_pairs(raw_df)
    else:
        pairs_df = pd.read_csv(pairs_path)
        logger.info("Loaded %d pairs from %s", len(pairs_df), pairs_path)

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df, val_df, test_df, report = subject_level_split(
        pairs_df, test_size=0.15, val_size=0.15, seed=SEED
    )
    logger.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # ── Preprocess MRI scalars ─────────────────────────────────────────────────
    X_train, y_train, imputer, scaler = prepare_mri_data(train_df, is_train=True)
    X_val, y_val = prepare_mri_data(val_df, imputer=imputer, scaler=scaler, is_train=False)
    X_test, y_test = prepare_mri_data(test_df, imputer=imputer, scaler=scaler, is_train=False)

    def make_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=16, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val, y_val, shuffle=False)
    test_loader  = make_loader(X_test, y_test, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MRIScalarClassifier(
        input_dim=len(MRI_SCALAR_FEATURES),
        embed_dim=32,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    ).to(device)

    class_weights = get_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    EPOCHS, PATIENCE = 50, 10

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        logger.info("Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f",
                    epoch, EPOCHS, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    model.load_state_dict(best_state)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    test_metrics, y_true, y_pred = evaluate(model, test_loader, device)
    logger.info("Test Metrics: %s", json.dumps(test_metrics, indent=2))

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), artifact_dir / "mri_scalar_classifier.pt")

    import joblib
    joblib.dump(imputer, artifact_dir / "mri_imputer.pkl")
    joblib.dump(scaler, artifact_dir / "mri_scaler.pkl")

    results = {
        "experiment_id": "EXP-MRI-SCALAR-001",
        "phase": 3,
        "description": "MRI-Derived Scalar Branch — standalone experiment",
        "modality": "MRI-derived scalars (nWBV, eTIV, ASF, nwbv_delta)",
        "dataset": "OASIS-2 Kaggle CSV",
        "features": MRI_SCALAR_FEATURES,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "seed": SEED,
        "model": "MRIScalarClassifier (MLP)",
        "test_metrics": test_metrics,
        "note": (
            "MRI features are derived scalar measurements (FreeSurfer outputs) "
            "from OASIS-2, NOT raw 3D MRI volumes. This is the maximum MRI "
            "information available in the OASIS-2 Kaggle dataset."
        ),
    }

    metrics_path = artifact_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Confusion matrix
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0.0", "0.5", "1.0", "2.0"],
            yticklabels=["0.0", "0.5", "1.0", "2.0"],
            ax=ax
        )
        ax.set_xlabel("Predicted CDR")
        ax.set_ylabel("True CDR")
        ax.set_title("MRI Scalar Branch — Test Set Confusion Matrix")
        fig.tight_layout()
        fig.savefig(artifact_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        logger.warning("Could not save confusion matrix: %s", e)

    logger.info("=" * 60)
    logger.info("PHASE 3 — MRI SCALAR BRANCH RESULTS")
    logger.info("=" * 60)
    logger.info("Features: %s", MRI_SCALAR_FEATURES)
    logger.info("Accuracy:          %.1f%%", test_metrics["accuracy"] * 100)
    logger.info("Balanced Accuracy: %.1f%%", test_metrics["balanced_accuracy"] * 100)
    logger.info("F1 Macro:          %.3f", test_metrics["f1_macro"])
    logger.info("MAE (CDR):         %.3f", test_metrics["mae"])
    logger.info("Artifact dir: %s", artifact_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
