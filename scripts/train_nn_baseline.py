#!/usr/bin/env python
"""
scripts/train_nn_baseline.py
----------------------------
Train a PyTorch Multi-Layer Perceptron (MLP) baseline model.

Usage:
    python scripts/train_nn_baseline.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.schemas import PairColumns
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.data.pytorch.datasets import LongitudinalClinicalDataset
from cerebro_x.models.deep.mlp import ClinicalMLP
from cerebro_x.utils.io import setup_logging, make_artifact_dir

logger = logging.getLogger("train_nn_baseline")


def get_class_weights(y: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Compute balanced class weights for CrossEntropyLoss."""
    class_counts = torch.bincount(y, minlength=num_classes).float()
    # Avoid division by zero for classes with 0 instances
    class_counts[class_counts == 0] = 1.0 
    
    total = len(y)
    weights = total / (num_classes * class_counts)
    return weights


def main():
    setup_logging("INFO")
    
    # ── 1. Setup outputs ───────────────────────────────────────────────────
    artifact_dir = make_artifact_dir("artifacts", "EXP-NN-BASELINE-001")
    logger.info("Artifacts will be saved to: %s", artifact_dir)
    
    # Set device to CPU since this is just scaffolding/baseline
    device = torch.device("cpu")
    logger.info("Using device: %s", device)
    
    # ── 2. Load data ───────────────────────────────────────────────────────
    pairs_path = Path("data/processed/next_visit_pairs.csv")
    if not pairs_path.exists():
        logger.error("Pairs CSV not found. Run scripts/build_pairs.py first.")
        sys.exit(1)
        
    pairs_df = pd.read_csv(pairs_path)
    logger.info("Loaded %d pairs.", len(pairs_df))
    
    # ── 3. Split data ──────────────────────────────────────────────────────
    train_val_df, _, test_df, _ = subject_level_split(
        pairs_df, test_size=0.15, val_size=0.0, seed=42, target_col=PairColumns.NEXT_CDR
    )
    
    # ── 4. Build PyTorch Datasets & Loaders ─────────────────────────────────
    train_dataset = LongitudinalClinicalDataset(train_val_df, is_train=True)
    
    # Use the imputer and scaler fitted on train to transform test
    test_dataset = LongitudinalClinicalDataset(
        test_df, 
        imputer=train_dataset.imputer, 
        scaler=train_dataset.scaler, 
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = len(train_dataset.feature_names)
    logger.info("Input features: %d", input_dim)
    
    # Compute class weights for the loss function
    class_weights = get_class_weights(train_dataset.y_tensor).to(device)
    logger.info("Class weights: %s", class_weights)
    
    # ── 5. Setup Model, Loss, Optimizer ─────────────────────────────────────
    model = ClinicalMLP(input_dim=input_dim, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # ── 6. Short Training Loop (Scaffolding Proof of Concept) ───────────────
    epochs = 10
    logger.info("Starting training for %d epochs...", epochs)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            
        avg_train_loss = total_loss / len(train_dataset)
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                
        avg_val_loss = val_loss / len(test_dataset)
        val_acc = correct / len(test_dataset)
        
        logger.info(
            "Epoch [%2d/%d] | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f",
            epoch + 1, epochs, avg_train_loss, avg_val_loss, val_acc
        )
        
    # Save model
    model_path = artifact_dir / "clinical_mlp_cpu.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Training complete. Model saved to %s", model_path)


if __name__ == "__main__":
    main()
