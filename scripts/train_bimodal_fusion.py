#!/usr/bin/env python
"""
scripts/train_bimodal_fusion.py
--------------------------------
Phase 5: Bimodal Fusion Ablation Study.

Runs three conditions and compares them:
    A) Clinical Only  — ClinicalGRUEncoder + head
    B) MRI Only       — MRIScalarClassifier
    C) Clinical + MRI — BimodalCerebroNet (Z_t = 96-dim brain state)

This is the core Phase 5 ablation required by the master instruction.
Saves to: artifacts/EXP-FUSION-BIMODAL-001/

Usage:
    python scripts/train_bimodal_fusion.py
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
    accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.oasis2.loader import load_oasis2_auto
from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs
from cerebro_x.data.pytorch.bimodal_dataset import BimodalSequenceDataset, bimodal_collate_fn
from cerebro_x.data.pytorch.sequence_dataset import SequenceDataset, sequence_collate_fn, CDR_MAP
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.models.deep.mri_scalar import MRIScalarClassifier, MRI_SCALAR_FEATURES
from cerebro_x.models.deep.bimodal_fusion import ClinicalOnlyClassifier, BimodalCerebroNet
from cerebro_x.utils.io import setup_logging, make_artifact_dir

logger = logging.getLogger("train_bimodal_fusion")

CDR_INV = {v: k for k, v in CDR_MAP.items()}
NUM_CLASSES = 4
SEED = 42
EPOCHS = 40
PATIENCE = 10
LR = 1e-3


def get_class_weights(dataset_or_array, num_classes=4):
    if hasattr(dataset_or_array, "samples"):
        targets = torch.tensor([s["target"] for s in dataset_or_array.samples], dtype=torch.long)
    else:
        targets = torch.tensor(dataset_or_array, dtype=torch.long)
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts[counts == 0] = 1.0
    return len(targets) / (num_classes * counts)


def eval_model_seq(model, loader, device, is_bimodal=False):
    """Evaluate a sequence model (clinical-only or bimodal)."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            if is_bimodal:
                clin_seq, mri_sc, targets, lengths = batch
                clin_seq = clin_seq.to(device)
                mri_sc = mri_sc.to(device)
                lengths = lengths.to(device)
                logits = model(clin_seq, mri_sc, lengths)
            else:
                features, targets, lengths = batch
                logits = model(features.to(device), lengths.to(device))

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(targets.numpy())

    y_true_cdr = np.array([CDR_INV[int(p)] for p in all_true])
    y_pred_cdr = np.array([CDR_INV[int(p)] for p in all_preds])

    return {
        "accuracy": float(accuracy_score(all_true, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_preds)),
        "f1_macro": float(f1_score(all_true, all_preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(all_true, all_preds, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_true_cdr, y_pred_cdr)),
    }


def train_seq_model(model, train_loader, val_loader, test_loader,
                    class_weights, device, artifact_dir, name, is_bimodal=False):
    """Generic training loop for sequence-based models."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model = model.to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    logger.info("=== Training: %s ===", name)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_n = 0.0, 0

        for batch in train_loader:
            if is_bimodal:
                clin_seq, mri_sc, targets, lengths = batch
                clin_seq = clin_seq.to(device)
                mri_sc = mri_sc.to(device)
                targets = targets.to(device)
                lengths = lengths.to(device)
                optimizer.zero_grad()
                logits = model(clin_seq, mri_sc, lengths)
            else:
                features, targets, lengths = batch
                features = features.to(device)
                targets = targets.to(device)
                lengths = lengths.to(device)
                optimizer.zero_grad()
                logits = model(features, lengths)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            total_n += len(targets)

        train_loss = total_loss / max(total_n, 1)

        # Validation loss
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if is_bimodal:
                    clin_seq, mri_sc, targets, lengths = batch
                    logits = model(clin_seq.to(device), mri_sc.to(device), lengths.to(device))
                    targets = targets.to(device)
                else:
                    features, targets, lengths = batch
                    logits = model(features.to(device), lengths.to(device))
                    targets = targets.to(device)
                val_loss += criterion(logits, targets).item() * len(targets)
                val_n += len(targets)
        val_loss /= max(val_n, 1)

        logger.info("[%s] Epoch %02d | Train: %.4f | Val: %.4f", name, epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("[%s] Early stopping at epoch %d.", name, epoch)
                break

    model.load_state_dict(best_state)
    metrics = eval_model_seq(model, test_loader, device, is_bimodal=is_bimodal)
    logger.info("[%s] Test: %s", name, json.dumps(metrics, indent=2))

    # Save checkpoint
    save_name = name.lower().replace(" ", "_").replace("+", "plus")
    torch.save(model.state_dict(), artifact_dir / f"{save_name}.pt")

    return metrics


def train_mri_only(train_df, val_df, test_df, artifact_dir, device):
    """Train standalone MRI scalar classifier (Condition B)."""
    from cerebro_x.models.deep.mri_scalar import MRIScalarClassifier, MRI_SCALAR_FEATURES

    def prep(df, imp=None, sc=None, fit=True):
        import pandas as pd
        X_raw = df[MRI_SCALAR_FEATURES].values.astype(float)
        y_raw = df["next_CDR"].map(CDR_MAP).values.astype(int)
        if fit:
            imp = SimpleImputer(strategy="median")
            X_imp = imp.fit_transform(X_raw)
            sc = StandardScaler()
            X_sc = sc.fit_transform(X_imp)
            return X_sc, y_raw, imp, sc
        return imp.transform(sc.inverse_transform(sc.transform(imp.transform(X_raw)))), y_raw

    X_train_raw = train_df[MRI_SCALAR_FEATURES].values.astype(float)
    y_train = train_df["next_CDR"].map(CDR_MAP).values.astype(int)
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train_raw)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train_imp)

    def make_mri(df):
        X = sc.transform(imp.transform(df[MRI_SCALAR_FEATURES].values.astype(float)))
        y = df["next_CDR"].map(CDR_MAP).values.astype(int)
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=16, shuffle=False)

    train_ds_raw = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds_raw, batch_size=16, shuffle=True)
    val_loader   = make_mri(val_df)
    test_loader  = make_mri(test_df)

    model = MRIScalarClassifier(input_dim=len(MRI_SCALAR_FEATURES), embed_dim=32, num_classes=NUM_CLASSES).to(device)
    cw = get_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    best_val, best_state, pat = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vl += criterion(model(Xb.to(device)), yb.to(device)).item() * len(yb)
        vl /= max(sum(len(b[0]) for b in val_loader), 1)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds = model(Xb.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    y_true_cdr = np.array([CDR_INV[p] for p in all_true])
    y_pred_cdr = np.array([CDR_INV[p] for p in all_preds])

    metrics = {
        "accuracy": float(accuracy_score(all_true, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_preds)),
        "f1_macro": float(f1_score(all_true, all_preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(all_true, all_preds, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_true_cdr, y_pred_cdr)),
    }
    torch.save(model.state_dict(), artifact_dir / "mri_only.pt")
    logger.info("[MRI Only] Test: %s", json.dumps(metrics, indent=2))
    return metrics


def main():
    setup_logging("INFO")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    artifact_dir = make_artifact_dir("artifacts", "EXP-FUSION-BIMODAL-001")
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

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION A: Clinical Only (GRU sequence model)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== CONDITION A: Clinical Only ===")
    train_ds_clin = SequenceDataset(train_df, is_train=True)
    val_ds_clin   = SequenceDataset(val_df, imputer=train_ds_clin.imputer, scaler=train_ds_clin.scaler, is_train=False)
    test_ds_clin  = SequenceDataset(test_df, imputer=train_ds_clin.imputer, scaler=train_ds_clin.scaler, is_train=False)

    train_loader_clin = DataLoader(train_ds_clin, batch_size=8, shuffle=True, collate_fn=sequence_collate_fn)
    val_loader_clin   = DataLoader(val_ds_clin, batch_size=8, shuffle=False, collate_fn=sequence_collate_fn)
    test_loader_clin  = DataLoader(test_ds_clin, batch_size=8, shuffle=False, collate_fn=sequence_collate_fn)

    clin_only_model = ClinicalOnlyClassifier(
        input_dim=train_ds_clin.num_features, hidden_dim=64, num_classes=NUM_CLASSES
    )
    cw_clin = get_class_weights(train_ds_clin)
    metrics_A = train_seq_model(
        clin_only_model, train_loader_clin, val_loader_clin, test_loader_clin,
        cw_clin, device, artifact_dir, "clinical_only", is_bimodal=False
    )

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION B: MRI Only
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== CONDITION B: MRI Only ===")
    metrics_B = train_mri_only(train_df, val_df, test_df, artifact_dir, device)

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION C: Clinical + MRI (BimodalCerebroNet)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== CONDITION C: Clinical + MRI (BimodalCerebroNet) ===")
    train_ds_bi = BimodalSequenceDataset(train_df, is_train=True)
    val_ds_bi   = BimodalSequenceDataset(
        val_df,
        clinical_imputer=train_ds_bi.clinical_imputer,
        clinical_scaler=train_ds_bi.clinical_scaler,
        mri_imputer=train_ds_bi.mri_imputer,
        mri_scaler=train_ds_bi.mri_scaler,
        is_train=False,
    )
    test_ds_bi  = BimodalSequenceDataset(
        test_df,
        clinical_imputer=train_ds_bi.clinical_imputer,
        clinical_scaler=train_ds_bi.clinical_scaler,
        mri_imputer=train_ds_bi.mri_imputer,
        mri_scaler=train_ds_bi.mri_scaler,
        is_train=False,
    )

    train_loader_bi = DataLoader(train_ds_bi, batch_size=8, shuffle=True, collate_fn=bimodal_collate_fn)
    val_loader_bi   = DataLoader(val_ds_bi, batch_size=8, shuffle=False, collate_fn=bimodal_collate_fn)
    test_loader_bi  = DataLoader(test_ds_bi, batch_size=8, shuffle=False, collate_fn=bimodal_collate_fn)

    bimodal_model = BimodalCerebroNet(
        clinical_input_dim=train_ds_bi.num_clinical_features,
        mri_input_dim=train_ds_bi.num_mri_features,
        clinical_hidden_dim=64,
        mri_embed_dim=32,
        num_classes=NUM_CLASSES,
    )
    cw_bi = get_class_weights(train_ds_bi)
    metrics_C = train_seq_model(
        bimodal_model, train_loader_bi, val_loader_bi, test_loader_bi,
        cw_bi, device, artifact_dir, "clinical_plus_mri", is_bimodal=True
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Save combined results
    # ─────────────────────────────────────────────────────────────────────────
    results = {
        "experiment_id": "EXP-FUSION-BIMODAL-001",
        "phase": 5,
        "description": "Bimodal Fusion Ablation — Clinical vs MRI vs Clinical+MRI",
        "dataset": "OASIS-2 Kaggle CSV",
        "seed": SEED,
        "train_subjects": len(train_df["Subject ID"].unique()),
        "val_subjects": len(val_df["Subject ID"].unique()),
        "test_subjects": len(test_df["Subject ID"].unique()),
        "mri_features": MRI_SCALAR_FEATURES,
        "note": (
            "MRI features are derived scalars (nWBV, eTIV, ASF) from OASIS-2. "
            "NOT raw 3D MRI volumes. EEG modality is absent — see DOCS/EEG_LIMITATION.md."
        ),
        "ablation": {
            "A_clinical_only": metrics_A,
            "B_mri_only": metrics_B,
            "C_clinical_plus_mri": metrics_C,
        },
    }

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PHASE 5 — BIMODAL FUSION ABLATION RESULTS")
    logger.info("=" * 70)
    logger.info("%-30s  Acc     B-Acc   F1-Mac  MAE", "Condition")
    logger.info("-" * 70)
    for label, m in [
        ("A) Clinical Only (GRU)", metrics_A),
        ("B) MRI Only (MLP)",      metrics_B),
        ("C) Clinical + MRI",      metrics_C),
    ]:
        logger.info(
            "%-30s  %.3f   %.3f   %.3f   %.3f",
            label, m["accuracy"], m["balanced_accuracy"], m["f1_macro"], m["mae"]
        )
    logger.info("=" * 70)
    logger.info("Artifacts saved to: %s", artifact_dir)


if __name__ == "__main__":
    main()
