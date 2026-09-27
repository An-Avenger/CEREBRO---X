"""Phase 4 Evaluation: Training the TriModalCerebroNet.

STATUS: Tri-modal architecture scaffold
WARNING: This is a scaffold. Real patient-aligned MRI + EEG + clinical data does not exist.
Do NOT report synthetic training metrics as real medical-model performance.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from cerebro_x.data.oasis2.loader import load_oasis2_auto
from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs
from cerebro_x.features.clinical import build_feature_matrix
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.data.pytorch.multimodal_dataset import TriModalDataset
from cerebro_x.data.eeg.transforms import get_default_eeg_transforms
from cerebro_x.models.deep.fusion import TriModalCerebroNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def align_all_modalities(pairs_df: pd.DataFrame, mri_dir: str, eeg_dir: str) -> list[dict]:
    """Combines alignment logic for all three modalities."""
    aligned = []
    for _, row in pairs_df.iterrows():
        item = {
            'subject_id': row['Subject ID'],
            'visit': int(row['current_visit']),
            'clinical_row': row.to_dict(),
            'next_cdr': row['next_CDR'],
            'mri_path': f"{mri_dir}/OAS2_{row['Subject ID']}_MR{int(row['current_visit'])}.nii.gz",
            'eeg_path': f"{eeg_dir}/{row['Subject ID']}_EEG{int(row['current_visit'])}.edf",
            'is_synthetic_mri': True, # Hardcoded fallback for scaffold
            'is_synthetic_eeg': True  # Hardcoded fallback for scaffold
        }
        aligned.append(item)
    return aligned


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Tri-modal architecture scaffold on {device}...")
    logger.warning("WARNING: This is a scaffold training loop with synthetic MRI and EEG data.")
    logger.warning("Do NOT claim this as real patient-level Clinical + MRI + EEG fusion.")

    # 1. Load Clinical Data
    raw_df = load_oasis2_auto()
    
    # 2. Build Longitudinal Pairs
    pairs_df, stats = build_next_visit_pairs(raw_df)
    logger.info(f"Constructed {len(pairs_df)} clinical pairs.")

    # 3. Extract feature list and encode categorical variables
    X_df, feat_cols = build_feature_matrix(
        pairs_df, 
        active_groups=['static_demographic', 'clinical_state', 'volumetric', 'timing', 'temporal_engineered']
    )
    # Update pairs_df with encoded values (sex, hand)
    for col in X_df.columns:
        pairs_df[col] = X_df[col]

    # 4. Align Modalities
    aligned_data = align_all_modalities(pairs_df, args.mri_dir, args.eeg_dir)
    
    # 5. Split Dataset
    train_df, val_df, _, _ = subject_level_split(pairs_df)
    train_ids = set(train_df['Subject ID'])
    val_ids = set(val_df['Subject ID'])
    
    train_aligned = [item for item in aligned_data if item['subject_id'] in train_ids]
    val_aligned = [item for item in aligned_data if item['subject_id'] in val_ids]
    
    # 6. PyTorch Datasets
    eeg_tfs = get_default_eeg_transforms()
    
    train_ds = TriModalDataset(train_aligned, feat_cols.feature_columns, eeg_transforms=eeg_tfs)
    val_ds = TriModalDataset(val_aligned, feat_cols.feature_columns, eeg_transforms=eeg_tfs)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 7. Initialize TriModal Model
    model = TriModalCerebroNet(
        clinical_features=len(feat_cols.feature_columns),
        clinical_embed_dim=64,
        mri_embed_dim=128,
        eeg_embed_dim=128,
        num_classes=4,
        dropout_rate=0.4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info("Beginning TriModal Training Loop...")
    
    # 8. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for c_seq, m_vol, e_sig, targets in train_loader:
            c_seq = c_seq.to(device)
            m_vol = m_vol.to(device)
            e_sig = e_sig.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(c_seq, m_vol, e_sig)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_loss:.4f}")

    # 9. Validation
    model.eval()
    all_preds, all_targs = [], []
    with torch.no_grad():
        for c_seq, m_vol, e_sig, targets in val_loader:
            c_seq = c_seq.to(device)
            m_vol = m_vol.to(device)
            e_sig = e_sig.to(device)
            
            logits = model(c_seq, m_vol, e_sig)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targs.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targs, all_preds)
    bacc = balanced_accuracy_score(all_targs, all_preds)
    f1 = f1_score(all_targs, all_preds, average='macro')
    
    logger.info(f"Validation Results - Acc: {acc:.4f} | B-Acc: {bacc:.4f} | F1: {f1:.4f}")
    
    out_dir = Path("artifacts/experiments/PHASE4_MULTIMODAL")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), out_dir / "trimodal_cerebro_net_scaffold.pt")
    
    import json
    metadata = {
        "status": "Tri-modal architecture scaffold",
        "modality_status": {
            "clinical": "real",
            "mri": "synthetic fallback (no checkpoint/data)",
            "eeg": "standalone (synthetic fallback for tri-modal)",
            "aligned_tri_modal": False
        },
        "warning": "Do NOT report synthetic training metrics as real medical-model performance."
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Saved scaffold model and metadata to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: TriModal Training")
    parser.add_argument("--mri_dir", type=str, default="data/raw/mri", help="Directory for MRI data")
    parser.add_argument("--eeg_dir", type=str, default="data/raw/eeg", help="Directory for EEG data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    main(args)
