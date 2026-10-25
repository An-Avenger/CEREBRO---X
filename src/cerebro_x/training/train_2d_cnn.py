"""
Training and Evaluation Pipeline for 2D ResNet-18 MRI Alzheimer's Classifier.

Features:
    - Subject-level split isolation (Train / Val / Test).
    - Class-weighted CrossEntropyLoss for class imbalance.
    - Early stopping & best checkpoint recovery.
    - Metric tracking: Accuracy, Balanced Accuracy, Macro F1, Per-class F1, Confusion Matrix.
    - Exporting classification report, learning curves, and predictions.
    - Held-out test evaluation + subject-level majority voting evaluation.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cerebro_x.imaging.mri_slice_dataset import MRISliceDataset
from cerebro_x.models.deep.cnn2d import MRI2DCNN

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for mitigating extreme class imbalance:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C), targets: (B,)
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    all_subjects = []

    for inputs, targets, subjs in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_subjects.extend(subjs)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Slice-level metrics
    acc = float(accuracy_score(all_targets, all_preds))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    avg_loss = float(total_loss / max(len(all_targets), 1))

    # Subject-level aggregation
    df_pred = pd.DataFrame({
        "subject_id": all_subjects,
        "target": all_targets,
        "prob_0": all_probs[:, 0],
        "prob_1": all_probs[:, 1],
        "prob_2": all_probs[:, 2],
    })
    subj_group = df_pred.groupby("subject_id").agg({
        "target": "first",
        "prob_0": "mean",
        "prob_1": "mean",
        "prob_2": "mean",
    }).reset_index()

    subj_probs = subj_group[["prob_0", "prob_1", "prob_2"]].values
    subj_preds = np.argmax(subj_probs, axis=1)
    subj_targets = subj_group["target"].values.astype(int)

    subj_acc = float(accuracy_score(subj_targets, subj_preds))
    subj_bal_acc = float(balanced_accuracy_score(subj_targets, subj_preds))
    subj_macro_f1 = float(f1_score(subj_targets, subj_preds, average="macro", zero_division=0))

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "slice_predictions": all_preds.tolist(),
        "slice_targets": all_targets.tolist(),
        "slice_probs": all_probs.tolist(),
        "subject_accuracy": subj_acc,
        "subject_balanced_accuracy": subj_bal_acc,
        "subject_macro_f1": subj_macro_f1,
        "subject_predictions": subj_preds.tolist(),
        "subject_targets": subj_targets.tolist(),
        "subject_probs": subj_probs.tolist(),
        "subject_ids": subj_group["subject_id"].tolist(),
    }


def plot_training_curves(history: list[dict], output_path: str | Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, label="Train Loss", color="royalblue")
    ax1.plot(epochs, val_loss, label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Learning Curves: Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1, label="Val Macro F1", color="forestgreen")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Validation Macro F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix (Subject-level)")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
            )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_mri_2d_pipeline(
    manifest_csv: str = "data/processed/mri_2d/slices_manifest.csv",
    split_csv: str = "artifacts/EXP-MRI-2D-CNN-001/subject_splits.csv",
    artifact_dir: str = "artifacts/EXP-MRI-2D-CNN-001",
    num_classes: int = 3,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    loss_type: str = "focal",
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    df_manifest = pd.read_csv(manifest_csv)
    df_splits = pd.read_csv(split_csv)[["subject_id", "split"]]

    # Merge split onto slice manifest
    merged_df = pd.merge(df_manifest, df_splits, on="subject_id", how="inner")
    logger.info(
        "Loaded %d slices across %d subjects. Splits: %s",
        len(merged_df),
        merged_df["subject_id"].nunique(),
        merged_df.groupby("split")["subject_id"].nunique().to_dict(),
    )

    df_train = merged_df[merged_df["split"] == "train"].copy()
    df_val = merged_df[merged_df["split"] == "val"].copy()
    df_test = merged_df[merged_df["split"] == "test"].copy()

    # Compute class weights for imbalanced loss
    train_class_counts = df_train["label_idx"].value_counts().sort_index().values
    total_train = len(df_train)
    class_weights = total_train / (num_classes * train_class_counts.astype(np.float32))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info("Class weights: %s", class_weights.tolist())

    if loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights_t, gamma=2.0)
        logger.info("Using FocalLoss (gamma=2.0, class-weighted alpha).")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        logger.info("Using weighted CrossEntropyLoss.")

    # Datasets and Loaders
    train_ds = MRISliceDataset(df_train)
    val_ds = MRISliceDataset(df_val)
    test_ds = MRISliceDataset(df_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = MRI2DCNN(num_classes=num_classes, pretrained=True, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_macro_f1 = -1.0
    best_checkpoint_path = artifact_path / "best_checkpoint.pt"
    history = []

    logger.info("Beginning training for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_res = evaluate_model(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_res["loss"],
            "val_accuracy": val_res["accuracy"],
            "val_balanced_acc": val_res["balanced_accuracy"],
            "val_macro_f1": val_res["macro_f1"],
            "val_subject_acc": val_res["subject_accuracy"],
            "val_subject_macro_f1": val_res["subject_macro_f1"],
        })

        scheduler.step(val_res["macro_f1"])

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f, Acc: %.3f | Val Loss: %.4f, Slice F1: %.3f, Subj Acc: %.3f, Subj F1: %.3f",
            epoch, epochs, train_loss, train_acc, val_res["loss"], val_res["macro_f1"],
            val_res["subject_accuracy"], val_res["subject_macro_f1"]
        )

        # Checkpoint if best validation macro F1
        if val_res["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_res["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_res["macro_f1"],
                "num_classes": num_classes,
                "architecture": "ResNet-18",
            }, str(best_checkpoint_path))
            logger.info("Saved best checkpoint (Val Macro F1: %.4f) to %s", best_val_macro_f1, best_checkpoint_path)

    # Save training history
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(artifact_path / "training_history.csv", index=False)
    plot_training_curves(history, artifact_path / "training_curves.png")

    # Evaluate held-out test set using best checkpoint
    logger.info("Loading best checkpoint for held-out test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_res = evaluate_model(model, test_loader, criterion, device)

    classes = ["CN", "MCI", "AD"]
    subj_y_true = test_res["subject_targets"]
    subj_y_pred = test_res["subject_predictions"]
    cm_subj = confusion_matrix(subj_y_true, subj_y_pred, labels=[0, 1, 2])
    plot_confusion_matrix(cm_subj, classes, artifact_path / "confusion_matrix.png")

    clf_report = classification_report(
        subj_y_true, subj_y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    clf_report_text = classification_report(
        subj_y_true, subj_y_pred, target_names=classes, zero_division=0
    )

    with open(artifact_path / "classification_report.txt", "w") as f:
        f.write(clf_report_text)

    # Save test predictions table
    df_test_preds = pd.DataFrame({
        "subject_id": test_res["subject_ids"],
        "target": [classes[i] for i in subj_y_true],
        "prediction": [classes[i] for i in subj_y_pred],
        "prob_CN": [p[0] for p in test_res["subject_probs"]],
        "prob_MCI": [p[1] for p in test_res["subject_probs"]],
        "prob_AD": [p[2] for p in test_res["subject_probs"]],
    })
    df_test_preds.to_csv(artifact_path / "test_predictions.csv", index=False)

    final_metrics = {
        "status": "completed",
        "model_architecture": "ResNet-18 (ImageNet Pretrained Transfer Learning)",
        "input_dimension": "(1, 224, 224)",
        "num_classes": num_classes,
        "classes": classes,
        "epochs_trained": epochs,
        "best_epoch": checkpoint["epoch"],
        "test_subjects_count": len(test_res["subject_ids"]),
        "test_slices_count": len(test_res["slice_targets"]),
        "subject_level": {
            "accuracy": test_res["subject_accuracy"],
            "balanced_accuracy": test_res["subject_balanced_accuracy"],
            "macro_f1": test_res["subject_macro_f1"],
            "precision_macro": float(precision_score(subj_y_true, subj_y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(subj_y_true, subj_y_pred, average="macro", zero_division=0)),
            "confusion_matrix": cm_subj.tolist(),
            "per_class": clf_report,
        },
        "slice_level": {
            "accuracy": test_res["accuracy"],
            "balanced_accuracy": test_res["balanced_accuracy"],
            "macro_f1": test_res["macro_f1"],
        },
        "checkpoints": {
            "best": str(best_checkpoint_path),
        },
    }

    with open(artifact_path / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Held-out Test Evaluation Complete!")
    logger.info("Subject Accuracy: %.4f | Subject Macro F1: %.4f | Balanced Acc: %.4f",
                test_res["subject_accuracy"], test_res["subject_macro_f1"], test_res["subject_balanced_accuracy"])
    logger.info("Classification Report:\n%s", clf_report_text)

    return final_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_mri_2d_pipeline()
