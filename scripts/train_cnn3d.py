#!/usr/bin/env python
"""
scripts/train_cnn3d.py
-----------------------
Train the MRICerebroNet (Lightweight3DCNN + classifier head) for MRI-based
CDR prediction.

DATASET CHOICE:
  The full OASIS-2 NIfTI dataset (~150 subjects × multiple scans) is NOT
  included in this repository. Without real neuroimaging volumes, true
  neuroscientific validity cannot be achieved.

  This script trains on SYNTHETIC volumetric data whose CDR label
  distribution exactly matches the OASIS-2 training split. The volumes
  are constructed as Gaussian random fields (not real brain MRI).

  This gives a checkpoint that:
    ✓ Has the correct architecture
    ✓ Loads without error
    ✓ Supports full Grad-CAM (real forward/backward hooks)
    ✓ Produces valid class logits
    ✗ Does NOT have neuroimaging predictive validity

  Any Grad-CAM output from this model reflects learned synthetic
  distribution patterns, NOT true anatomical biomarkers.

  STATUS: EXP-MRI-CNN3D-001 = RESEARCH SCAFFOLD (synthetic training).
          Clinically meaningful Grad-CAM requires real T1 NIfTI volumes.

Usage:
    python scripts/train_cnn3d.py
    python scripts/train_cnn3d.py --epochs 20 --batch-size 8 --seed 42

Outputs:
    artifacts/EXP-MRI-CNN3D-001/cnn3d_cpu.pt        — state_dict
    artifacts/EXP-MRI-CNN3D-001/cnn3d_config.json   — architecture config
    artifacts/EXP-MRI-CNN3D-001/cnn3d_meta.json     — training metadata
    artifacts/EXP-MRI-CNN3D-001/metrics.json         — train/val metrics
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.models.deep.cnn3d import MRICerebroNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_cnn3d")

ARTIFACT_DIR = Path("artifacts/EXP-MRI-CNN3D-001")
VOLUME_SHAPE = (64, 64, 64)

# OASIS-2 CDR distribution (approximate, from EXP-LONGITUDINAL-001)
# CDR 0.0 → class 0 (~60%), CDR 0.5 → class 1 (~30%), CDR 1.0 → class 2 (~8%), CDR 2.0 → class 3 (~2%)
CDR_CLASS_WEIGHTS = [0.60, 0.30, 0.08, 0.02]


class SyntheticMRIDataset(Dataset):
    """
    Synthetic volumetric MRI dataset matching OASIS-2 CDR distribution.

    Each sample is a 64×64×64 Gaussian volume with class-specific
    mean intensity slightly shifted per CDR class (to give the CNN
    a learnable signal). The shift is subtle, mimicking atrophy trends.

    CDR class → mean intensity offset:
        0 (Normal)      → +0.2 (higher brain volume proxy)
        1 (Very Mild)   → +0.05
        2 (Mild)        → -0.1
        3 (Moderate)    → -0.3 (lower brain volume proxy)

    This is a purely synthetic training signal. It is NOT real MRI.
    """

    def __init__(
        self,
        n_samples: int = 200,
        seed: int = 42,
        augment: bool = False,
    ):
        rng = np.random.default_rng(seed)
        n_classes = len(CDR_CLASS_WEIGHTS)
        class_counts = np.round(np.array(CDR_CLASS_WEIGHTS) * n_samples).astype(int)
        # Fix rounding
        diff = n_samples - class_counts.sum()
        class_counts[0] += diff

        labels = []
        for cls, cnt in enumerate(class_counts):
            labels.extend([cls] * cnt)
        labels = np.array(labels, dtype=np.int64)
        rng.shuffle(labels)

        # Class-specific mean offsets (to give learnable signal)
        class_offsets = {0: 0.20, 1: 0.05, 2: -0.10, 3: -0.30}

        volumes = []
        for lbl in labels:
            # Gaussian random field + class offset + mild spatial pattern
            vol = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
            vol += class_offsets[lbl]
            # Add a slight ellipsoidal "brain-like" mask
            cx, cy, cz = 32, 32, 32
            x = np.arange(64) - cx
            y = np.arange(64) - cy
            z = np.arange(64) - cz
            xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
            mask = (xx**2 / 28**2 + yy**2 / 24**2 + zz**2 / 24**2) < 1.0
            vol[~mask] *= 0.05  # attenuate background
            volumes.append(vol)

        self.volumes = np.stack(volumes)[:, np.newaxis, ...]  # (N, 1, 64, 64, 64)
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        vol = torch.from_numpy(self.volumes[idx].copy())  # (1, 64, 64, 64)
        if self.augment:
            # Simple random flip augmentation
            if torch.rand(1).item() > 0.5:
                vol = torch.flip(vol, dims=[1])
            if torch.rand(1).item() > 0.5:
                vol = torch.flip(vol, dims=[2])
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return vol, lbl


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for vols, labels in loader:
        vols, labels = vols.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(vols)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for vols, labels in loader:
        vols, labels = vols.to(device), labels.to(device)
        logits = model(vols)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Train MRICerebroNet on synthetic data")
    parser.add_argument("--epochs",     type=int,   default=25,  help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=8,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n-train",    type=int,   default=160, help="Training samples")
    parser.add_argument("--n-val",      type=int,   default=40,  help="Validation samples")
    parser.add_argument("--seed",       type=int,   default=42,  help="Random seed")
    parser.add_argument("--embedding-dim", type=int, default=64, help="CNN embedding dim")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # CPU training for portability

    logger.info("=" * 60)
    logger.info("MRICerebroNet Training — SYNTHETIC DATA")
    logger.info("=" * 60)
    logger.warning(
        "IMPORTANT: Training on SYNTHETIC volumes (not real MRI NIfTI). "
        "This checkpoint supports Grad-CAM architecture but does NOT have "
        "neuroimaging predictive validity. See script docstring for details."
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = SyntheticMRIDataset(n_samples=args.n_train, seed=args.seed, augment=True)
    val_ds   = SyntheticMRIDataset(n_samples=args.n_val,   seed=args.seed + 1, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))
    logger.info("CDR distribution (train): %s",
                {c: int((train_ds.labels == c).sum()) for c in range(4)})

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MRICerebroNet(
        in_channels=1,
        embedding_dim=args.embedding_dim,
        num_classes=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: MRICerebroNet | Parameters: %d", n_params)

    # Class weights to handle imbalance
    class_weights = torch.tensor(
        [1.0 / w for w in CDR_CLASS_WEIGHTS], dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * 4  # normalize

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = -1.0
    best_state = None
    history = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f train_acc=%.3f | "
                "val_loss=%.4f val_acc=%.3f | best_val_acc=%.3f",
                epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, best_val_acc,
            )

    elapsed = time.time() - t0
    logger.info("Training complete in %.1f s | Best val_acc=%.4f", elapsed, best_val_acc)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    checkpoint_path = ARTIFACT_DIR / "cnn3d_cpu.pt"
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, str(checkpoint_path))
    ckpt_hash = sha256_of_file(checkpoint_path)
    logger.info("Checkpoint saved: %s (sha256=%s...)", checkpoint_path, ckpt_hash[:16])

    # ── Architecture config ───────────────────────────────────────────────────
    config = {
        "model_class": "cerebro_x.models.deep.cnn3d.MRICerebroNet",
        "in_channels": 1,
        "embedding_dim": args.embedding_dim,
        "num_classes": 4,
        "input_shape": [1, 1, 64, 64, 64],
        "gradcam_target": "cnn.features[-4] (last Conv3d before AdaptiveAvgPool3d)",
    }
    config_path = ARTIFACT_DIR / "cnn3d_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # ── Training metadata ─────────────────────────────────────────────────────
    meta = {
        "experiment_id": "EXP-MRI-CNN3D-001",
        "model_class": "MRICerebroNet",
        "status": "SYNTHETIC_SCAFFOLD",
        "warning": (
            "Trained on SYNTHETIC volumetric data, NOT real MRI NIfTI. "
            "Architecture is correct. Grad-CAM hooks fire correctly. "
            "Predictions do NOT have neuroimaging validity. "
            "Replace with real OASIS-2/ADNI NIfTI training for clinical relevance."
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_hash,
        "dataset": "Synthetic (Gaussian volumes, CDR distribution from OASIS-2)",
        "n_train": args.n_train,
        "n_val": args.n_val,
        "cdr_class_weights": CDR_CLASS_WEIGHTS,
        "epochs": args.epochs,
        "best_epoch": max(history, key=lambda x: x["val_acc"])["epoch"],
        "best_val_acc": round(best_val_acc, 4),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "seed": args.seed,
        "training_time_s": round(elapsed, 1),
        "device": str(device),
        "volume_shape": list(VOLUME_SHAPE),
        "n_parameters": n_params,
        "training_history": history,
        "clinically_validated": False,
    }
    meta_path = ARTIFACT_DIR / "cnn3d_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # ── metrics.json (for registry._load_experiment_metrics) ─────────────────
    metrics = {
        "experiment_id": "EXP-MRI-CNN3D-001",
        "model": "MRICerebroNet",
        "dataset": "Synthetic (NOT real MRI)",
        "best_val_accuracy": round(best_val_acc, 4),
        "warning": "Synthetic training data only. Not neuroimaging-valid.",
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("=" * 60)
    logger.info("Artifacts written to %s", ARTIFACT_DIR)
    logger.info("  cnn3d_cpu.pt    — model checkpoint")
    logger.info("  cnn3d_config.json — architecture")
    logger.info("  cnn3d_meta.json — training metadata")
    logger.info("  metrics.json    — experiment metrics")
    logger.info("=" * 60)
    logger.warning(
        "REMINDER: This is a SYNTHETIC checkpoint. "
        "Replace with real NIfTI training data for clinically meaningful Grad-CAM."
    )


if __name__ == "__main__":
    main()
