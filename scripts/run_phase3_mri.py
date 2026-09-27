"""Phase 3 Evaluation: Training 3D CNN on MRI volumes for next-visit CDR.

Designed to be run on GPU (e.g., Kaggle).
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cerebro_x.data.oasis2.loader import load_oasis2_auto
from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs
from cerebro_x.data.mri.mri_loader import align_pairs_with_mri
from cerebro_x.data.mri.transforms import MRITransformPipeline
from cerebro_x.data.pytorch.mri_dataset import MRIDataset
from cerebro_x.models.deep.cnn3d import MRICerebroNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Phase 3 MRI Training on {device}...")
    
    mri_dir = Path(args.mri_dir)
    if not mri_dir.exists():
        logger.warning(f"MRI directory {mri_dir} does not exist. Please mount dataset!")
        
    # 1. Load clinical data
    raw_df = load_oasis2_auto()
    pairs_df, _ = build_next_visit_pairs(raw_df)
    
    # 2. Align with MRI
    aligned_df = align_pairs_with_mri(pairs_df, raw_df, mri_dir)
    
    if len(aligned_df) == 0:
        logger.error("No MRI pairs could be aligned. Exiting.")
        return
        
    # In a real scenario, we'd use splits.subject_level_split here on aligned_df
    # For scaffolding, we just use a simple train/val split
    split_idx = int(0.8 * len(aligned_df))
    train_df = aligned_df.iloc[:split_idx]
    val_df = aligned_df.iloc[split_idx:]
    
    logger.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}")
    
    # 3. Create Datasets
    transform = MRITransformPipeline(target_shape=(64, 64, 64))
    
    train_ds = MRIDataset(train_df, transform=transform)
    val_ds = MRIDataset(val_df, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 4. Model
    model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 5. Training Loop
    logger.info("Beginning Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mri_dir", type=str, default="data/raw/mri", help="Path to raw MRI nifti files")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)
