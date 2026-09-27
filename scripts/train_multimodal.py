#!/usr/bin/env python
"""
scripts/train_multimodal.py
---------------------------
Train the PyTorch Multi-Modal Fusion network (Clinical + MRI).

Usage:
    python scripts/train_multimodal.py
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
from cerebro_x.data.pytorch.datasets import MultiModalDataset
from cerebro_x.models.deep.fusion import MultiModalCerebroNetwork
from cerebro_x.utils.io import setup_logging, make_artifact_dir

logger = logging.getLogger("train_multimodal")


def get_class_weights(y: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    class_counts = torch.bincount(y, minlength=num_classes).float()
    class_counts[class_counts == 0] = 1.0 
    return len(y) / (num_classes * class_counts)


def main():
    setup_logging("INFO")
    
    # ── 1. Setup outputs ───────────────────────────────────────────────────
    artifact_dir = make_artifact_dir("artifacts", "EXP-MULTIMODAL-001")
    logger.info("Artifacts will be saved to: %s", artifact_dir)
    
    device = torch.device("cpu")
    logger.info("Using device: %s", device)
    
    # ── 2. Load data ───────────────────────────────────────────────────────
    pairs_path = Path("data/processed/next_visit_pairs.csv")
    if not pairs_path.exists():
        logger.error("Pairs CSV not found.")
        sys.exit(1)
        
    pairs_df = pd.read_csv(pairs_path)
    # Use a smaller subset for quick CPU testing
    pairs_df = pairs_df.head(10).copy()
    logger.info("Loaded %d pairs (mock subset).", len(pairs_df))
    
    # ── 3. Split data ──────────────────────────────────────────────────────
    train_val_df, _, test_df, _ = subject_level_split(
        pairs_df, test_size=0.2, val_size=0.0, seed=42, target_col=PairColumns.NEXT_CDR
    )
    
    # ── 4. Build PyTorch Datasets & Loaders ─────────────────────────────────
    train_dataset = MultiModalDataset(train_val_df, is_train=True)
    test_dataset = MultiModalDataset(
        test_df, 
        imputer=train_dataset.imputer, 
        scaler=train_dataset.scaler, 
        is_train=False
    )
    
    # Small batch size due to 3D convolutions
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    input_dim = len(train_dataset.feature_names)
    logger.info("Clinical input features: %d", input_dim)
    
    class_weights = get_class_weights(train_dataset.y_tensor).to(device)
    
    # ── 5. Setup Model, Loss, Optimizer ─────────────────────────────────────
    model = MultiModalCerebroNetwork(clinical_input_dim=input_dim, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # ── 6. Short Training Loop (Proof of Concept) ───────────────────────────
    epochs = 2
    logger.info("Starting training for %d epochs...", epochs)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for X_clin, X_mri, y_batch in train_loader:
            X_clin = X_clin.to(device)
            X_mri = X_mri.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_clin, X_mri)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_clin.size(0)
            
        avg_train_loss = total_loss / len(train_dataset)
        logger.info("Epoch [%d/%d] | Train Loss: %.4f", epoch + 1, epochs, avg_train_loss)
        
    model_path = artifact_dir / "multimodal_cpu.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Training complete. Model saved to %s", model_path)


if __name__ == "__main__":
    main()
