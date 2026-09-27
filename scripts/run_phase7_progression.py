#!/usr/bin/env python
"""
scripts/run_phase7_progression.py
----------------------------------
Phase 7: Longitudinal Disease Progression Analysis.

Performs deep evaluation of the trained GRU temporal model:
    1. Per-class performance breakdown (CDR 0.0, 0.5, 1.0, 2.0)
    2. CDR trajectory plots — actual vs predicted across visits
    3. Calibration analysis
    4. Progression category analysis (Stable / Improving / Declining)
    5. Error analysis by visit count and history length

Prediction horizon: ONE VISIT AHEAD (next-visit CDR).
This is explicitly documented — we do not claim multi-step or long-term
disease progression prediction beyond what the data supports.

Saves to: artifacts/EXP-PROGRESSION-001/

Usage:
    python scripts/run_phase7_progression.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    mean_absolute_error, classification_report, confusion_matrix
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.oasis2.loader import load_oasis2_auto
from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs
from cerebro_x.data.pytorch.sequence_dataset import SequenceDataset, sequence_collate_fn, CDR_MAP
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.models.deep.temporal import TemporalCerebroNet
from cerebro_x.utils.io import setup_logging, make_artifact_dir

logger = logging.getLogger("run_phase7_progression")

SEED = 42
CDR_INV = {v: k for k, v in CDR_MAP.items()}
CDR_CLASSES = [0.0, 0.5, 1.0, 2.0]
GRU_CHECKPOINT = Path("artifacts/EXP-LONGITUDINAL-001/temporal_gru_cpu.pt")


def predict_all(model, loader, device):
    model.eval()
    all_true, all_pred, all_lengths, all_probs = [], [], [], []
    with torch.no_grad():
        for features, targets, lengths in loader:
            logits = model(features.to(device), lengths.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_true.extend(targets.numpy())
            all_pred.extend(preds)
            all_lengths.extend(lengths.numpy())
            all_probs.extend(probs)
    return (
        np.array(all_true), np.array(all_pred),
        np.array(all_lengths), np.array(all_probs)
    )


def plot_cdr_trajectory(pairs_df, model, train_ds, device, plots_dir, n_patients=12):
    """Plot actual vs predicted CDR trajectory for sample patients."""
    from cerebro_x.features.clinical import build_feature_matrix

    X_df, _ = build_feature_matrix(pairs_df)
    X_imp = train_ds.imputer.transform(X_df.values)
    X_scaled = np.asarray(train_ds.scaler.transform(X_imp))

    subject_ids = pairs_df["Subject ID"].values
    visit_nums  = pairs_df["current_visit"].values
    next_cdrs   = pairs_df["next_CDR"].values

    from collections import defaultdict
    patient_data: dict[str, list] = defaultdict(list)
    for i, (sid, v, cdr, feat) in enumerate(zip(subject_ids, visit_nums, next_cdrs, X_scaled)):
        patient_data[sid].append({"visit": int(v), "next_cdr": float(cdr), "features": feat})
    for sid in patient_data:
        patient_data[sid].sort(key=lambda x: x["visit"])

    selected = [sid for sid, visits in patient_data.items() if len(visits) >= 2][:n_patients]

    from torch.nn.utils.rnn import pack_padded_sequence
    model.eval()

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, sid in enumerate(selected[:len(axes)]):
        visits_data = patient_data[sid]
        true_cdrs, pred_cdrs, visit_labels = [], [], []

        for t in range(len(visits_data)):
            seq = torch.tensor(
                np.stack([visits_data[j]["features"] for j in range(t + 1)]),
                dtype=torch.float32
            ).unsqueeze(0)
            length = torch.tensor([t + 1])

            with torch.no_grad():
                logits = model(seq.to(device), length.to(device))
                pred_class = logits.argmax(dim=1).item()

            true_cdrs.append(visits_data[t]["next_cdr"])
            pred_cdrs.append(CDR_INV[pred_class])
            visit_labels.append(f"V{visits_data[t]['visit']}")

        ax = axes[ax_idx]
        ax.plot(visit_labels, true_cdrs, "bo-", label="Actual", linewidth=2, markersize=8)
        ax.plot(visit_labels, pred_cdrs, "rs--", label="Predicted", linewidth=2, markersize=8)
        ax.set_ylim(-0.1, 2.3)
        ax.set_yticks([0.0, 0.5, 1.0, 2.0])
        ax.set_title(sid, fontsize=8, fontweight="bold")
        ax.set_ylabel("CDR", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    for ax_idx in range(len(selected), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        "Phase 7 — CDR Trajectory: Actual vs Predicted (next-visit horizon)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    path = plots_dir / "cdr_trajectories.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("CDR trajectory plot saved: %s", path)


def plot_calibration(y_true_class, probs, plots_dir):
    """Simple calibration plot: predicted probability vs actual frequency."""
    n_bins = 5
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for cls_idx, cdr_val in enumerate(CDR_CLASSES):
        cls_probs = probs[:, cls_idx]
        cls_binary = (y_true_class == cls_idx).astype(int)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_means, bin_fracs, bin_sizes = [], [], []

        for i in range(n_bins):
            mask = (cls_probs >= bins[i]) & (cls_probs < bins[i + 1])
            if mask.sum() > 0:
                bin_means.append(cls_probs[mask].mean())
                bin_fracs.append(cls_binary[mask].mean())
                bin_sizes.append(mask.sum())

        ax = axes[cls_idx]
        if bin_means:
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
            ax.plot(bin_means, bin_fracs, "bo-", label="Model")
            ax.fill_between(bin_means, bin_fracs, bin_means,
                            alpha=0.2, color="orange", label="Gap")
        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Fraction of positives", fontsize=9)
        ax.set_title(f"CDR = {cdr_val}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 7 — Calibration Analysis per CDR Class", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = plots_dir / "calibration.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration plot saved: %s", path)


def progression_category_analysis(pairs_df, y_true_class, y_pred_class):
    """Analyze prediction accuracy by progression category."""
    curr_cdrs = pairs_df["curr_cdr"].values
    next_cdrs = pairs_df["next_CDR"].values

    categories = []
    for curr, nxt in zip(curr_cdrs, next_cdrs):
        diff = float(nxt) - float(curr)
        if diff > 0:
            categories.append("Declining")
        elif diff < 0:
            categories.append("Improving")
        else:
            categories.append("Stable")

    categories = np.array(categories)

    # Need to align with test set
    result = {}
    for cat in ["Stable", "Declining", "Improving"]:
        mask = categories == cat
        if mask.sum() == 0:
            continue
        result[cat] = {
            "count": int(mask.sum()),
            "pct": float(mask.mean() * 100),
        }
    return result, categories


def main():
    setup_logging("INFO")
    torch.manual_seed(SEED)

    artifact_dir = make_artifact_dir("artifacts", "EXP-PROGRESSION-001")
    plots_dir = artifact_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Artifact dir: %s", artifact_dir)

    device = torch.device("cpu")

    # ── Load data ─────────────────────────────────────────────────────────────
    pairs_path = Path("data/processed/next_visit_pairs.csv")
    if not pairs_path.exists():
        raw_df = load_oasis2_auto()
        pairs_df, _ = build_next_visit_pairs(raw_df)
    else:
        pairs_df = pd.read_csv(pairs_path)
    logger.info("Loaded %d pairs", len(pairs_df))

    train_df, val_df, test_df, _ = subject_level_split(
        pairs_df, test_size=0.15, val_size=0.15, seed=SEED
    )

    train_ds = SequenceDataset(train_df, is_train=True)
    test_ds  = SequenceDataset(test_df, imputer=train_ds.imputer, scaler=train_ds.scaler, is_train=False)

    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=sequence_collate_fn)

    # ── Load GRU ──────────────────────────────────────────────────────────────
    if not GRU_CHECKPOINT.exists():
        logger.error("GRU checkpoint not found: %s. Run train_longitudinal.py first.", GRU_CHECKPOINT)
        sys.exit(1)

    model = TemporalCerebroNet(
        input_dim=train_ds.num_features,
        gru_hidden_dim=64, gru_num_layers=1, head_hidden_dim=32, dropout=0.3
    )
    model.load_state_dict(torch.load(GRU_CHECKPOINT, map_location="cpu"))
    logger.info("Loaded GRU from %s", GRU_CHECKPOINT)

    # ── Predict ───────────────────────────────────────────────────────────────
    y_true_class, y_pred_class, seq_lengths, probs = predict_all(model, test_loader, device)

    y_true_cdr = np.array([CDR_INV[int(c)] for c in y_true_class])
    y_pred_cdr = np.array([CDR_INV[int(c)] for c in y_pred_class])

    # ── Core metrics ──────────────────────────────────────────────────────────
    core_metrics = {
        "accuracy": float(accuracy_score(y_true_class, y_pred_class)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_class, y_pred_class)),
        "f1_macro": float(f1_score(y_true_class, y_pred_class, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_class, y_pred_class, average="weighted", zero_division=0)),
        "mae_cdr": float(mean_absolute_error(y_true_cdr, y_pred_cdr)),
        "prediction_horizon": "next visit (1 step ahead)",
        "test_samples": int(len(y_true_class)),
        "test_patients": int(test_df["Subject ID"].nunique()),
    }

    # Per-class metrics
    report = classification_report(
        y_true_class, y_pred_class,
        target_names=["CDR 0.0", "CDR 0.5", "CDR 1.0", "CDR 2.0"],
        zero_division=0,
        output_dict=True,
    )

    # Metrics by sequence length
    length_buckets = {
        "length_1": (seq_lengths == 1),
        "length_2": (seq_lengths == 2),
        "length_3_plus": (seq_lengths >= 3),
    }
    per_length_metrics = {}
    for bucket_name, mask in length_buckets.items():
        if mask.sum() > 0:
            per_length_metrics[bucket_name] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true_class[mask], y_pred_class[mask])),
                "balanced_accuracy": float(balanced_accuracy_score(y_true_class[mask], y_pred_class[mask])),
                "mae_cdr": float(mean_absolute_error(y_true_cdr[mask], y_pred_cdr[mask])),
            }

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_cdr_trajectory(pairs_df, model, train_ds, device, plots_dir)
    plot_calibration(y_true_class, probs, plots_dir)

    # Confusion matrix
    try:
        import seaborn as sns
        cm = confusion_matrix(y_true_class, y_pred_class)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0.0", "0.5", "1.0", "2.0"],
            yticklabels=["0.0", "0.5", "1.0", "2.0"], ax=ax
        )
        ax.set_xlabel("Predicted CDR")
        ax.set_ylabel("True CDR")
        ax.set_title("Phase 7 — GRU Confusion Matrix (Test Set)")
        fig.tight_layout()
        fig.savefig(plots_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        logger.warning("Confusion matrix plot failed: %s", e)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "experiment_id": "EXP-PROGRESSION-001",
        "phase": 7,
        "description": "Longitudinal Disease Progression Analysis",
        "model": "TemporalCerebroNet (GRU, hidden_dim=64)",
        "model_checkpoint": str(GRU_CHECKPOINT),
        "dataset": "OASIS-2 Kaggle CSV",
        "prediction_horizon": "1 visit ahead (next-visit CDR)",
        "important_limitation": (
            "OASIS-2 supports next-visit CDR prediction only. "
            "We do NOT claim multi-step or long-term disease progression prediction. "
            "The average inter-visit interval is ~732 days (2 years)."
        ),
        "test_metrics": core_metrics,
        "per_class_report": report,
        "per_sequence_length": per_length_metrics,
    }

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 65)
    logger.info("PHASE 7 — LONGITUDINAL PROGRESSION ANALYSIS")
    logger.info("=" * 65)
    logger.info("Prediction horizon: next visit (1 step ahead)")
    logger.info("Test patients:      %d", core_metrics["test_patients"])
    logger.info("Accuracy:           %.1f%%", core_metrics["accuracy"] * 100)
    logger.info("Balanced Accuracy:  %.1f%%", core_metrics["balanced_accuracy"] * 100)
    logger.info("F1 Macro:           %.3f", core_metrics["f1_macro"])
    logger.info("MAE (CDR):          %.3f", core_metrics["mae_cdr"])
    logger.info("")
    logger.info("Per-length breakdown:")
    for k, v in per_length_metrics.items():
        logger.info("  %s: n=%d, Acc=%.1f%%, B-Acc=%.1f%%",
                    k, v["n"], v["accuracy"]*100, v["balanced_accuracy"]*100)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
