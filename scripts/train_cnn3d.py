#!/usr/bin/env python
"""
scripts/train_cnn3d.py
------------------------
Train MRICerebroNet on REAL ADNI T1-weighted MRI volumes.

REQUIREMENTS (must all be satisfied before this script runs):
  1. Manifest with labels: data/raw/mri/ADNI/manifest_with_labels.csv
     (produced by scripts/prepare_adni_clinical.py)
  2. NIfTI files: referenced in the manifest nifti_path column
     (produced by dcm2niix conversion of the ADNI DICOM archive)
  3. CUDA GPU: required unless --no-gpu is passed explicitly

This script will NOT fall back to synthetic data.
If any requirement is missing, the script exits with a clear error.

Subject-level splitting:
  - Participants are split by subject_id, never by individual scans.
  - All scans from a given subject are kept in a single split.
  - Train: 70%, Validation: 15%, Test: 15% (of subjects).

Usage:
    python scripts/train_cnn3d.py \\
        --manifest  data/raw/mri/ADNI/manifest_with_labels.csv \\
        --nifti-dir data/raw/mri/ADNI/nifti \\
        --out-dir   artifacts/EXP-MRI-ADNI-001 \\
        --epochs 50 --batch-size 4 --seed 42

    # CPU-only (for smoke testing on local machine without GPU):
    python scripts/train_cnn3d.py \\
        --manifest ... --nifti-dir ... --no-gpu --epochs 2 --batch-size 2

Outputs (in --out-dir):
    best_checkpoint.pt      — full checkpoint (model + optimizer + config)
    cnn3d_config.json       — architecture and preprocessing config
    training_history.csv    — per-epoch metrics
    metrics.json            — final evaluation metrics
    split_report.json       — subject-level split assignment
    env_info.json           — Python/PyTorch/CUDA environment
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset, CDR_CLASS_LABELS
from cerebro_x.models.deep.cnn3d import MRICerebroNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_cnn3d")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MRICerebroNet on REAL ADNI data.")
    parser.add_argument("--manifest",   required=True, help="Path to manifest_with_labels.csv")
    parser.add_argument("--nifti-dir",  required=True, help="Root directory of NIfTI files")
    parser.add_argument("--out-dir",    default="artifacts/EXP-MRI-ADNI-001", help="Output directory")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-workers",   type=int, default=0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--patience",   type=int,   default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--val-frac",   type=float, default=0.15)
    parser.add_argument("--test-frac",  type=float, default=0.15)
    parser.add_argument("--resume",     default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU training. Without this flag, CUDA is required.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Apply random flips during training.",
    )
    return parser.parse_args()


# ── Device setup ──────────────────────────────────────────────────────────────

def setup_device(no_gpu: bool) -> torch.device:
    if no_gpu:
        logger.warning("--no-gpu flag set. Training on CPU. This will be very slow.")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available on this machine. "
            "GPU training is required for real ADNI MRI data (29.4 GB archive). "
            "Run on a GPU-equipped environment (Kaggle, Colab, cloud VM) or pass --no-gpu "
            "explicitly for a CPU smoke-test only."
        )
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info("GPU: %s | VRAM: %.1f GB", gpu_name, vram_gb)
    return device


# ── Subject-level split ───────────────────────────────────────────────────────

def subject_level_split(
    manifest: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict]:
    """
    Split manifest rows by participant, not by individual scans.

    All rows belonging to a subject appear in exactly one split.
    Returns (train_indices, val_indices, test_indices, report).
    """
    rng = np.random.default_rng(seed)

    subjects = np.array(manifest["subject_id"].unique())
    rng.shuffle(subjects)
    n = len(subjects)

    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    n_train = n - n_test - n_val

    if n_train <= 0:
        raise ValueError(
            f"Too few subjects ({n}) for a 3-way split "
            f"(train={n_train}, val={n_val}, test={n_test}). "
            f"Minimum required: 3 subjects."
        )

    test_subjects  = set(subjects[:n_test])
    val_subjects   = set(subjects[n_test : n_test + n_val])
    train_subjects = set(subjects[n_test + n_val :])

    # Verify no overlap
    assert len(train_subjects & val_subjects) == 0, "Train/Val subject overlap!"
    assert len(train_subjects & test_subjects) == 0, "Train/Test subject overlap!"
    assert len(val_subjects & test_subjects) == 0, "Val/Test subject overlap!"

    train_idx = manifest.index[manifest["subject_id"].isin(train_subjects)].tolist()
    val_idx   = manifest.index[manifest["subject_id"].isin(val_subjects)].tolist()
    test_idx  = manifest.index[manifest["subject_id"].isin(test_subjects)].tolist()

    report = {
        "seed": seed,
        "n_subjects_total": n,
        "n_subjects_train": len(train_subjects),
        "n_subjects_val": len(val_subjects),
        "n_subjects_test": len(test_subjects),
        "n_scans_train": len(train_idx),
        "n_scans_val": len(val_idx),
        "n_scans_test": len(test_idx),
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
        "label_dist_train": dict(Counter(manifest.loc[train_idx, "label"].tolist())),
        "label_dist_val":   dict(Counter(manifest.loc[val_idx,   "label"].tolist())),
        "label_dist_test":  dict(Counter(manifest.loc[test_idx,  "label"].tolist())),
    }

    logger.info(
        "Subject-level split: %d train / %d val / %d test subjects | "
        "%d / %d / %d scans",
        len(train_subjects), len(val_subjects), len(test_subjects),
        len(train_idx), len(val_idx), len(test_idx),
    )
    return train_idx, val_idx, test_idx, report


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(
    manifest: pd.DataFrame,
    train_idx: list[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the ACTUAL training label distribution.
    Never uses hard-coded weights from a different dataset.
    """
    train_labels = manifest.loc[train_idx, "label"].tolist()
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = []
    for cls in range(num_classes):
        cnt = counts.get(cls, 0)
        if cnt == 0:
            logger.warning(
                "Class %d (%s) has 0 training samples. "
                "Weight set to 0 — this class will not contribute to the loss. "
                "Consider obtaining more data.",
                cls, CDR_CLASS_LABELS.get(cls, "?"),
            )
            weights.append(0.0)
        else:
            weights.append(total / cnt)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info("Class weights (from training data): %s", {i: round(w, 3) for i, w in enumerate(weights)})
    return weight_tensor


# ── Training and evaluation steps ─────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for vols, labels in loader:
        vols, labels = vols.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(vols)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for vols, labels in loader:
        vols, labels = vols.to(device), labels.to(device)
        logits = model(vols)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        all_preds,
        all_labels,
    )


# ── Checkpoint save/load ──────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    config: dict,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_acc": best_val_acc,
        "config": config,
    }
    torch.save(checkpoint, str(path))


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
) -> tuple[int, float]:
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_val_acc"]


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: list[int],
    all_labels: list[int],
    num_classes: int,
) -> dict:
    label_names = [CDR_CLASS_LABELS.get(i, f"class_{i}") for i in range(num_classes)]
    metrics = {
        "accuracy": round(float(accuracy_score(all_labels, all_preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(all_labels, all_preds)), 4),
        "macro_f1": round(float(f1_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))).tolist(),
        "per_class_report": classification_report(
            all_labels, all_preds, labels=list(range(num_classes)),
            target_names=label_names, output_dict=True, zero_division=0,
        ),
    }
    return metrics


# ── Env info ──────────────────────────────────────────────────────────────────

def get_env_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Fix random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(no_gpu=args.no_gpu)

    logger.info("=" * 60)
    logger.info("MRICerebroNet Training — REAL ADNI DATA")
    logger.info("=" * 60)
    logger.info("Manifest:  %s", args.manifest)
    logger.info("NIfTI dir: %s", args.nifti_dir)
    logger.info("Output:    %s", out_dir)
    logger.info("Device:    %s", device)

    # ── Load full dataset (for splitting) ─────────────────────────────────────
    full_dataset = RealADNIMRIDataset(
        manifest_path=args.manifest,
        nifti_base_dir=args.nifti_dir,
        augment=False,
        target_shape=(64, 64, 64),
    )

    manifest = full_dataset.manifest
    num_classes = full_dataset.num_classes

    # ── Subject-level split ───────────────────────────────────────────────────
    train_idx, val_idx, test_idx, split_report = subject_level_split(
        manifest=manifest,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    (out_dir / "split_report.json").write_text(json.dumps(split_report, indent=2))

    # Build augmented training dataset
    train_dataset_aug = RealADNIMRIDataset(
        manifest_path=args.manifest,
        nifti_base_dir=args.nifti_dir,
        augment=args.augment,
        target_shape=(64, 64, 64),
    )

    train_ds  = Subset(train_dataset_aug, train_idx)
    val_ds    = Subset(full_dataset, val_idx)
    test_ds   = Subset(full_dataset, test_idx)

    logger.info(
        "Dataset sizes: train=%d val=%d test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    if len(train_ds) == 0:
        logger.error("Training set is empty. Check manifest and split fractions.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MRICerebroNet(
        in_channels=1,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: MRICerebroNet | Parameters: %d | Num classes: %d", n_params, num_classes)

    # ── Class weights from ACTUAL training data ───────────────────────────────
    class_weights = compute_class_weights(manifest, train_idx, num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Config ────────────────────────────────────────────────────────────────
    config = {
        "experiment_id": out_dir.name,
        "model_class": "cerebro_x.models.deep.cnn3d.MRICerebroNet",
        "dataset": "Real ADNI T1w NIfTI",
        "manifest": str(args.manifest),
        "nifti_dir": str(args.nifti_dir),
        "in_channels": 1,
        "embedding_dim": args.embedding_dim,
        "num_classes": num_classes,
        "cdr_class_labels": CDR_CLASS_LABELS,
        "input_shape": [1, 1, 64, 64, 64],
        "target_shape": [64, 64, 64],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "patience": args.patience,
        "seed": args.seed,
        "augment": args.augment,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_parameters": n_params,
        "device": str(device),
        "preprocessing": {
            "target_shape": [64, 64, 64],
            "resize_mode": "trilinear",
            "clip_percentiles": [0.5, 99.5],
            "normalization": "z_score_per_sample",
            "orientation": "RAS_canonical",
        },
        "clinically_validated": False,
        "warning": (
            "This model is a research tool trained on a single clinical dataset. "
            "Outputs are RESEARCH PREDICTIONS, NOT medical diagnoses. "
            "Performance has not been externally validated."
        ),
    }
    (out_dir / "cnn3d_config.json").write_text(json.dumps(config, indent=2))

    # ── Resume training ───────────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = -1.0
    best_state = None
    if args.resume and Path(args.resume).exists():
        start_epoch, best_val_acc = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler
        )
        start_epoch += 1
        logger.info("Resumed from %s at epoch %d | best_val_acc=%.4f", args.resume, start_epoch - 1, best_val_acc)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    patience_counter = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
        }
        history.append(entry)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_checkpoint(
                out_dir / "best_checkpoint.pt",
                model, optimizer, scheduler, epoch, best_val_acc, config,
            )
            patience_counter = 0
        else:
            patience_counter += 1

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f%s | best=%.3f",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
            " *" if improved else "",
            best_val_acc,
        )

        if patience_counter >= args.patience:
            logger.info(
                "Early stopping: val_acc has not improved for %d epochs.", args.patience
            )
            break

    elapsed = time.time() - t0
    logger.info("Training complete in %.1f s | Best val_acc=%.4f", elapsed, best_val_acc)

    # ── Save training history ─────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    # ── Final evaluation on test set ──────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_preds, test_labels, num_classes)

    logger.info("Test set evaluation:")
    logger.info("  Accuracy:          %.4f", test_metrics["accuracy"])
    logger.info("  Balanced accuracy: %.4f", test_metrics["balanced_accuracy"])
    logger.info("  Macro F1:          %.4f", test_metrics["macro_f1"])

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_out = {
        "experiment_id": out_dir.name,
        "model": "MRICerebroNet",
        "dataset": "Real ADNI T1w NIfTI",
        "n_train_scans": len(train_ds),
        "n_val_scans": len(val_ds),
        "n_test_scans": len(test_ds),
        "n_train_subjects": split_report["n_subjects_train"],
        "n_val_subjects": split_report["n_subjects_val"],
        "n_test_subjects": split_report["n_subjects_test"],
        "best_val_accuracy": round(best_val_acc, 4),
        "best_epoch": max(history, key=lambda x: x["val_acc"])["epoch"],
        "test_metrics": test_metrics,
        "training_time_s": round(elapsed, 1),
        "device": str(device),
        "seed": args.seed,
        "training_history": history,
        "limitations": [
            "Single-site, single-cohort evaluation. Results may not generalize.",
            "CDR is a clinical rating, not an imaging biomarker.",
            "This is a research prototype. Outputs are NOT medical diagnoses.",
        ],
        "clinically_validated": False,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    # ── Env info ──────────────────────────────────────────────────────────────
    (out_dir / "env_info.json").write_text(json.dumps(get_env_info(device), indent=2))

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Artifacts saved to: %s", out_dir)
    logger.info("  best_checkpoint.pt    — best model checkpoint")
    logger.info("  cnn3d_config.json     — architecture config")
    logger.info("  training_history.csv  — per-epoch metrics")
    logger.info("  metrics.json          — final metrics")
    logger.info("  split_report.json     — subject-level split")
    logger.info("  env_info.json         — environment")
    logger.info("=" * 60)
    logger.warning(
        "IMPORTANT: These are RESEARCH PREDICTIONS. "
        "Do NOT use as medical diagnoses. See metrics.json for limitations."
    )


if __name__ == "__main__":
    main()
