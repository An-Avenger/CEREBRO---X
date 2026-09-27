#!/usr/bin/env python
"""
scripts/run_explainability.py
-------------------------------
Generates SHAP visual explanations for the Temporal GRU model.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.pytorch.sequence_dataset import SequenceDataset, sequence_collate_fn
from cerebro_x.data.schemas import PairColumns
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.explainability.shap_temporal import compute_temporal_shap
from cerebro_x.explainability.visualizations import plot_temporal_shap_summary, plot_temporal_shap_waterfall
from cerebro_x.models.deep.temporal import TemporalCerebroNet
from cerebro_x.utils.io import setup_logging, make_artifact_dir

logger = logging.getLogger("run_explainability")


def main():
    setup_logging("INFO")
    
    # ── 1. Setup outputs ──────────────────────────────────────────────────────
    artifact_dir = make_artifact_dir("artifacts", "EXP-EXPLAIN-001")
    logger.info("Artifacts directory: %s", artifact_dir)
    device = torch.device("cpu")
    
    # ── 2. Load data ──────────────────────────────────────────────────────────
    pairs_path = Path("data/processed/next_visit_pairs.csv")
    if not pairs_path.exists():
        logger.error("Pairs CSV not found.")
        sys.exit(1)
        
    pairs_df = pd.read_csv(pairs_path)
    train_df, _, test_df, _ = subject_level_split(
        pairs_df, test_size=0.15, val_size=0.15, seed=42, target_col=PairColumns.NEXT_CDR
    )
    
    train_ds = SequenceDataset(train_df, is_train=True)
    test_ds = SequenceDataset(test_df, imputer=train_ds.imputer, scaler=train_ds.scaler, is_train=False)
    
    # Need to load actual model architecture config used in train_longitudinal
    model = TemporalCerebroNet(
        input_dim=train_ds.num_features,
        gru_hidden_dim=64,
        gru_num_layers=1,
        head_hidden_dim=32,
        dropout=0.3,
    )
    
    model_path = Path("artifacts/EXP-LONGITUDINAL-001/temporal_gru_cpu.pt")
    if not model_path.exists():
        logger.error("Model weights not found at %s", model_path)
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Loaded trained model from %s", model_path)
    
    # ── 3. Prepare data for SHAP ──────────────────────────────────────────────
    # SHAP needs a background dataset (to integrate over) and a test dataset (to explain)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=sequence_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=sequence_collate_fn)
    
    bg_features, _, bg_lengths = next(iter(train_loader))
    test_features, _, test_lengths = next(iter(test_loader))
    
    # SHAP requires background and test data to have the exact same shape for interpolation.
    # Because collate_fn pads to max length of the batch, they might differ.
    max_seq_len = max(bg_features.size(1), test_features.size(1))
    
    import torch.nn.functional as F
    if bg_features.size(1) < max_seq_len:
        bg_features = F.pad(bg_features, (0, 0, 0, max_seq_len - bg_features.size(1)))
    if test_features.size(1) < max_seq_len:
        test_features = F.pad(test_features, (0, 0, 0, max_seq_len - test_features.size(1)))
        
    # ── 4. Compute SHAP Values ────────────────────────────────────────────────
    logger.info("Computing SHAP values (this may take a minute)...")
    shap_values_raw = compute_temporal_shap(
        model=model,
        background_data=bg_features,
        background_lengths=bg_lengths,
        test_data=test_features,
        test_lengths=test_lengths,
    )
    
    # GradientExplainer returns a list of shape (num_classes, num_test, seq_len, num_features)
    # We are usually most interested in the highest class or the positive class.
    # Let's explain class 3 (CDR=2.0) or class 1/2 if 3 is empty
    class_to_explain = 2  # Explain why model thinks CDR=1.0 (Mild Dementia)
    
    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[class_to_explain]
    else:
        # If it returns a single array of shape (num_classes, batch, seq_len, features)
        if hasattr(shap_values_raw, 'shape') and len(shap_values_raw.shape) == 4:
            if shap_values_raw.shape[0] == 4: # Assuming 4 classes
                shap_vals = shap_values_raw[class_to_explain]
            else:
                shap_vals = shap_values_raw[:, class_to_explain, :, :]
        else:
            shap_vals = shap_values_raw
        
    if isinstance(shap_vals, torch.Tensor):
        shap_vals = shap_vals.detach().numpy()


    
    test_features_np = test_features.detach().numpy()
    
    # ── 5. Generate Visualizations ────────────────────────────────────────────
    summary_path = artifact_dir / "temporal_shap_summary.png"
    plot_temporal_shap_summary(
        shap_values=shap_vals,
        features=test_features_np,
        feature_names=train_ds.feature_names,
        output_path=summary_path,
        max_display=15,
    )
    
    waterfall_path = artifact_dir / "temporal_shap_patient_0.png"
    # Get expected value from explainer base value (rough estimate for GradientExplainer)
    # Gradient explainer base value is the mean model output on the background set
    with torch.no_grad():
        bg_preds = model(bg_features, bg_lengths)
        expected_value = float(bg_preds[:, class_to_explain].mean().item())
        
    try:
        plot_temporal_shap_waterfall(
            shap_values=shap_vals,
            features=test_features_np,
            feature_names=train_ds.feature_names,
            expected_value=expected_value,
            sample_idx=0,
            output_path=waterfall_path,
        )
    except Exception as e:
        logger.warning("Could not generate waterfall plot: %s", e)
        
    logger.info("Explainability pipeline complete!")


if __name__ == "__main__":
    main()
