"""Phase 5: Explainability & Ablation Studies.

Performs Inference-Time Ablation across modalities and generates 
Grad-CAM heatmaps and SHAP feature importance for the Multimodal Digital Brain Twin.
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs
from cerebro_x.data.eeg.transforms import get_default_eeg_transforms
from cerebro_x.data.oasis2.loader import load_oasis2_auto
from cerebro_x.data.pytorch.multimodal_dataset import TriModalDataset
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.explainability.gradcam import GradCAM3D
from cerebro_x.explainability.multimodal_shap import compute_clinical_shap_multimodal
from cerebro_x.features.clinical import build_feature_matrix
from cerebro_x.models.deep.fusion import TriModalCerebroNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def align_all_modalities(pairs_df: pd.DataFrame, mri_dir: str, eeg_dir: str) -> list[dict]:
    aligned = []
    for _, row in pairs_df.iterrows():
        item = {
            'subject_id': row['Subject ID'],
            'visit': int(row['current_visit']),
            'clinical_row': row.to_dict(),
            'next_cdr': row['next_CDR'],
            'mri_path': f"{mri_dir}/OAS2_{row['Subject ID']}_MR{int(row['current_visit'])}.nii.gz",
            'eeg_path': f"{eeg_dir}/{row['Subject ID']}_EEG{int(row['current_visit'])}.edf",
            'is_synthetic_mri': True,
            'is_synthetic_eeg': True
        }
        aligned.append(item)
    return aligned


def evaluate_model(model, loader, device, mask_clinical=False, mask_mri=False, mask_eeg=False):
    model.eval()
    all_preds, all_targs = [], []
    
    with torch.no_grad():
        for c_seq, m_vol, e_sig, targets in loader:
            c_seq = c_seq.to(device)
            m_vol = m_vol.to(device)
            e_sig = e_sig.to(device)
            
            if mask_clinical:
                c_seq = torch.zeros_like(c_seq)
            if mask_mri:
                m_vol = torch.zeros_like(m_vol)
            if mask_eeg:
                e_sig = torch.zeros_like(e_sig)
                
            logits = model(c_seq, m_vol, e_sig)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targs.extend(targets.cpu().numpy())

    return f1_score(all_targs, all_preds, average='macro'), accuracy_score(all_targs, all_preds)


def main(args):
    device = torch.device("cpu") # Grad-CAM/SHAP easier to debug on CPU for scaffolding
    logger.info("Starting Phase 5: Explainability & Ablation...")
    out_dir = Path("artifacts/experiments/PHASE5_EXPLAINABILITY")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data Prep
    raw_df = load_oasis2_auto()
    pairs_df, _ = build_next_visit_pairs(raw_df)
    X_df, feat_cols = build_feature_matrix(
        pairs_df, 
        active_groups=['static_demographic', 'clinical_state', 'volumetric', 'timing', 'temporal_engineered']
    )
    for col in X_df.columns:
        pairs_df[col] = X_df[col]
        
    aligned_data = align_all_modalities(pairs_df, args.mri_dir, args.eeg_dir)
    train_df, val_df, _, _ = subject_level_split(pairs_df)
    
    train_aligned = [item for item in aligned_data if item['subject_id'] in set(train_df['Subject ID'])]
    val_aligned = [item for item in aligned_data if item['subject_id'] in set(val_df['Subject ID'])]
    
    eeg_tfs = get_default_eeg_transforms()
    train_ds = TriModalDataset(train_aligned, feat_cols.feature_columns, eeg_transforms=eeg_tfs)
    val_ds = TriModalDataset(val_aligned, feat_cols.feature_columns, eeg_transforms=eeg_tfs)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Load Model
    model = TriModalCerebroNet(
        clinical_features=len(feat_cols.feature_columns),
        clinical_embed_dim=64, mri_embed_dim=128, eeg_embed_dim=128, num_classes=4
    ).to(device)
    
    model_path = Path("artifacts/experiments/PHASE4_MULTIMODAL/trimodal_cerebro_net.pt")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded trained weights from {model_path}")
    else:
        logger.warning("No trained weights found. Using random initialization for demonstration.")

    # 3. Inference-Time Ablation Study
    logger.info("Running Inference-Time Ablation Study on Validation Set...")
    base_f1, base_acc = evaluate_model(model, val_loader, device)
    logger.info(f"Baseline F1: {base_f1:.4f} | Acc: {base_acc:.4f}")
    
    no_clin_f1, _ = evaluate_model(model, val_loader, device, mask_clinical=True)
    logger.info(f"No-Clinical F1: {no_clin_f1:.4f} (Drop: {base_f1 - no_clin_f1:.4f})")
    
    no_mri_f1, _ = evaluate_model(model, val_loader, device, mask_mri=True)
    logger.info(f"No-MRI F1: {no_mri_f1:.4f} (Drop: {base_f1 - no_mri_f1:.4f})")
    
    no_eeg_f1, _ = evaluate_model(model, val_loader, device, mask_eeg=True)
    logger.info(f"No-EEG F1: {no_eeg_f1:.4f} (Drop: {base_f1 - no_eeg_f1:.4f})")

    # 4. Grad-CAM on MRI
    logger.info("Generating Grad-CAM heatmap for sample patient...")
    # Grab one batch, pick first sample
    c_seq, m_vol, e_sig, targ = next(iter(val_loader))
    c_seq_1 = c_seq[0:1].to(device)
    m_vol_1 = m_vol[0:1].to(device)
    e_sig_1 = e_sig[0:1].to(device)
    
    # Target the last Conv3D layer in the MRI encoder
    target_layer = model.mri_encoder.cnn.features[12] # The last Conv3d before pooling in Lightweight3DCNN
    cam_generator = GradCAM3D(model, target_layer)
    heatmap = cam_generator.generate(c_seq_1, m_vol_1, e_sig_1)
    
    # Extract middle slice (axial)
    mid_idx = heatmap.shape[0] // 2
    slice_heatmap = heatmap[mid_idx, :, :]
    slice_mri = m_vol_1[0, 0, mid_idx, :, :].cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original MRI Slice (Synthetic)")
    plt.imshow(slice_mri, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(slice_mri, cmap='gray', alpha=0.5)
    plt.imshow(slice_heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    heatmap_path = out_dir / "gradcam_mri.png"
    plt.savefig(heatmap_path, bbox_inches='tight')
    logger.info(f"Saved Grad-CAM visualization to {heatmap_path}")

    # 5. Multimodal SHAP for Clinical Features
    logger.info("Computing SHAP values for clinical features...")
    # Use training data as background
    c_bg, m_bg, e_bg, _ = next(iter(train_loader))
    bg_clinical = c_bg.to(device)
    
    shap_vals = compute_clinical_shap_multimodal(
        model, 
        background_clinical=bg_clinical,
        fixed_background_mri=m_vol_1,
        fixed_background_eeg=e_sig_1,
        test_clinical=c_seq_1
    )
    
    # shap_vals is usually a list of length num_classes
    # We take the SHAP values for the predicted class
    pred_class = int(model(c_seq_1, m_vol_1, e_sig_1).argmax().item())
    
    # Average impact over all dimensions except the last one (features)
    # shap_vals could be a list of arrays or a single array of varying ndim
    pat_shap = np.array(shap_vals)
    pat_shap = np.abs(pat_shap).mean(axis=tuple(range(pat_shap.ndim - 1))) # Shape: (19,)
        
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    features = feat_cols.feature_columns
    
    # Sort by absolute impact
    idx = np.argsort(np.abs(pat_shap))
    features_list = [features[i] for i in idx[-15:]]
    plt.barh(features_list, pat_shap[idx][-15:], color='steelblue')
    plt.title(f"SHAP Clinical Feature Impact (Class {pred_class})")
    plt.xlabel("SHAP Value (Impact on model output)")
    
    shap_path = out_dir / "shap_clinical.png"
    plt.tight_layout()
    plt.savefig(shap_path)
    logger.info(f"Saved SHAP visualization to {shap_path}")
    logger.info("Phase 5 Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mri_dir", type=str, default="data/raw/mri")
    parser.add_argument("--eeg_dir", type=str, default="data/raw/eeg")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
