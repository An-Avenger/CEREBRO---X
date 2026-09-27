#!/usr/bin/env python
"""
scripts/train_longitudinal.py
-------------------------------
Train Phase 5 longitudinal temporal models on clinical visit sequences.

Trains two models and compares against Phase 3 baseline:
    1. LastVisitBaseline  — uses only the latest visit (sequence-unaware baseline)
    2. TemporalCerebroNet — GRU consuming the full visit history

Usage:
    python scripts/train_longitudinal.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.schemas import PairColumns
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.evaluation.metrics import evaluate_predictions, plot_confusion_matrix_custom
from cerebro_x.data.pytorch.sequence_dataset import (
    SequenceDataset,
    sequence_collate_fn,
    CDR_MAP,
)
from cerebro_x.models.deep.temporal import LastVisitBaseline, TemporalCerebroNet
from cerebro_x.utils.io import setup_logging, make_artifact_dir

logger = logging.getLogger("train_longitudinal")

# Inverse CDR map: class index → original CDR value
CDR_INV = {v: k for k, v in CDR_MAP.items()}


def get_class_weights(dataset: SequenceDataset, num_classes: int = 4) -> torch.Tensor:
    """Compute inverse-frequency class weights from dataset targets."""
    targets = torch.tensor([s["target"] for s in dataset.samples], dtype=torch.long)
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts[counts == 0] = 1.0
    return len(targets) / (num_classes * counts)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, targets, lengths in loader:
        features = features.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(features, lengths)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run inference and return (y_true, y_pred, avg_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_true = []
    all_pred = []
    total_loss = 0.0
    total_samples = 0

    for features, targets, lengths in loader:
        features = features.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        logits = model(features, lengths)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        all_true.extend(targets.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    # Convert class indices back to CDR values for metric compatibility
    y_true = np.array([CDR_INV[int(c)] for c in all_true])
    y_pred = np.array([CDR_INV[int(c)] for c in all_pred])
    avg_loss = total_loss / max(total_samples, 1)

    return y_true, y_pred, avg_loss


def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    artifact_dir: Path,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 7,
) -> dict:
    """Full training loop with early stopping and evaluation."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model = model.to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    logger.info("=== Training %s for %d epochs ===", model_name, epochs)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        _, _, val_loss = evaluate_model(model, val_loader, device)

        logger.info(
            "[%s] Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f",
            model_name, epoch, epochs, train_loss, val_loss,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("[%s] Early stopping at epoch %d.", model_name, epoch)
                break

    # Load best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, test_loss = evaluate_model(model, test_loader, device)
    metrics = evaluate_predictions(y_true, y_pred)
    metrics["test_loss"] = float(test_loss)

    logger.info("[%s] Test metrics: %s", model_name, json.dumps(metrics, indent=2))

    # Save model checkpoint
    save_name = model_name.lower().replace(" ", "_")
    model_path = artifact_dir / f"{save_name}_cpu.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("[%s] Model saved: %s", model_name, model_path)

    # Save confusion matrix
    try:
        cm_path = artifact_dir / f"{save_name}_confusion_matrix.png"
        plot_confusion_matrix_custom(
            y_true, y_pred,
            classes=[0.0, 0.5, 1.0, 2.0],
            title=f"{model_name} — Test Set Confusion Matrix",
            output_path=cm_path,
        )
        logger.info("[%s] Confusion matrix saved: %s", model_name, cm_path)
    except Exception as e:
        logger.warning("[%s] Could not save confusion matrix: %s", model_name, e)

    return metrics


def main():
    setup_logging("INFO")

    # ── 1. Setup outputs ──────────────────────────────────────────────────────
    artifact_dir = make_artifact_dir("artifacts", "EXP-LONGITUDINAL-001")
    logger.info("Artifacts directory: %s", artifact_dir)

    device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── 2. Load data ──────────────────────────────────────────────────────────
    pairs_path = Path("data/processed/next_visit_pairs.csv")
    if not pairs_path.exists():
        logger.error("Pairs CSV not found at %s. Run scripts/build_pairs.py first.", pairs_path)
        sys.exit(1)

    pairs_df = pd.read_csv(pairs_path)
    logger.info("Loaded %d visit pairs from %d patients.",
                len(pairs_df), pairs_df[PairColumns.SUBJECT_ID].nunique())

    # ── 3. Subject-level split ────────────────────────────────────────────────
    train_df, val_df, test_df, split_report = subject_level_split(
        pairs_df, test_size=0.15, val_size=0.15, seed=42,
        target_col=PairColumns.NEXT_CDR,
    )

    # ── 4. Build sequence datasets ────────────────────────────────────────────
    train_ds = SequenceDataset(train_df, is_train=True)
    val_ds = SequenceDataset(val_df, imputer=train_ds.imputer, scaler=train_ds.scaler, is_train=False)
    test_ds = SequenceDataset(test_df, imputer=train_ds.imputer, scaler=train_ds.scaler, is_train=False)

    # ── 4b. Save fitted preprocessor alongside model ──────────────────────────
    # This is the canonical fitted object for API inference.
    # inference.py loads this — never hardcodes normalization constants.
    from cerebro_x.features.preprocessor import ClinicalPreprocessor
    preprocessor = ClinicalPreprocessor()
    preprocessor.imputer = train_ds.imputer
    preprocessor.scaler  = train_ds.scaler
    preprocessor.is_fitted = True
    preprocessor.save(artifact_dir)
    logger.info("ClinicalPreprocessor saved to %s", artifact_dir)

    logger.info("Sequence datasets: train=%d, val=%d, test=%d",
                len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        collate_fn=sequence_collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=sequence_collate_fn, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        collate_fn=sequence_collate_fn, drop_last=False,
    )

    input_dim = train_ds.num_features
    logger.info("Input features per visit: %d", input_dim)

    class_weights = get_class_weights(train_ds)
    logger.info("Class weights: %s", class_weights.tolist())

    # ── 5. Train LastVisitBaseline ────────────────────────────────────────────
    baseline_model = LastVisitBaseline(input_dim=input_dim, hidden_dim=32)
    baseline_metrics = train_and_evaluate(
        model=baseline_model,
        model_name="last_visit_baseline",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        device=device,
        artifact_dir=artifact_dir,
        epochs=30, lr=1e-3, patience=7,
    )

    # ── 6. Train TemporalCerebroNet ───────────────────────────────────────────
    temporal_model = TemporalCerebroNet(
        input_dim=input_dim,
        gru_hidden_dim=64,
        gru_num_layers=1,
        head_hidden_dim=32,
        dropout=0.3,
    )
    temporal_metrics = train_and_evaluate(
        model=temporal_model,
        model_name="temporal_gru",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        device=device,
        artifact_dir=artifact_dir,
        epochs=30, lr=1e-3, patience=7,
    )

    # ── 7. Save combined results ──────────────────────────────────────────────
    results = {
        "experiment_id": "EXP-LONGITUDINAL-001",
        "phase": 5,
        "dataset": str(pairs_path),
        "total_pairs": len(pairs_df),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "input_features": input_dim,
        "seed": 42,
        "phase_3_baseline_balanced_accuracy": 0.72,
        "models": {
            "last_visit_baseline": baseline_metrics,
            "temporal_gru": temporal_metrics,
        },
    }

    results_path = artifact_dir / "metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", results_path)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 5 RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Phase 3 LogReg Baseline:    %.1f%% Balanced Accuracy",
                72.0)
    logger.info("Last Visit Baseline:        %.1f%% Balanced Accuracy",
                baseline_metrics["balanced_accuracy"] * 100)
    logger.info("Temporal GRU:               %.1f%% Balanced Accuracy",
                temporal_metrics["balanced_accuracy"] * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
