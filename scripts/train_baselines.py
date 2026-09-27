#!/usr/bin/env python
"""
scripts/train_baselines.py
--------------------------
Train baseline models for next-visit CDR prediction.

Usage:
    python scripts/train_baselines.py

Outputs:
    artifacts/EXP-BASELINE-001/
        metrics.json
        model_weights.joblib
        plots/
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.schemas import PairColumns
from cerebro_x.features.clinical import build_feature_matrix
from cerebro_x.evaluation.splits import subject_level_split, get_cv_folds
from cerebro_x.models.baselines import get_all_baselines
from cerebro_x.evaluation.metrics import evaluate_predictions, plot_confusion_matrix_custom
from cerebro_x.utils.io import setup_logging, make_artifact_dir, save_json

logger = logging.getLogger("train_baselines")

def main():
    setup_logging("INFO")
    
    # ── 1. Setup outputs ───────────────────────────────────────────────────
    artifact_dir = make_artifact_dir("artifacts", "EXP-BASELINE-001")
    logger.info("Artifacts will be saved to: %s", artifact_dir)
    
    # ── 2. Load data ───────────────────────────────────────────────────────
    pairs_path = Path("data/processed/next_visit_pairs.csv")
    if not pairs_path.exists():
        logger.error("Pairs CSV not found. Run scripts/build_pairs.py first.")
        sys.exit(1)
        
    pairs_df = pd.read_csv(pairs_path)
    logger.info("Loaded %d pairs.", len(pairs_df))
    
    # Target values mapping if we wanted to enforce it, but they are floats 0.0, 0.5, 1.0, 2.0
    valid_classes = [0.0, 0.5, 1.0, 2.0]
    
    # ── 3. Split data ──────────────────────────────────────────────────────
    # We hold out 15% for testing
    train_val_df, _, test_df, split_report = subject_level_split(
        pairs_df, test_size=0.15, val_size=0.0, seed=42, target_col=PairColumns.NEXT_CDR
    )
    
    # ── 4. Build features ──────────────────────────────────────────────────
    X_train, schema_train = build_feature_matrix(train_val_df)
    y_train = train_val_df[PairColumns.NEXT_CDR].astype(str).values
    
    X_test, _ = build_feature_matrix(test_df)
    y_test = test_df[PairColumns.NEXT_CDR].astype(str).values
    
    logger.info("Feature matrix built. X_train shape: %s", X_train.shape)
    
    # ── 5. Train & Evaluate ────────────────────────────────────────────────
    models = get_all_baselines(seed=42)
    results = {}
    
    for name, pipeline in models.items():
        logger.info("Training %s...", name)
        
        # Fit on full train
        pipeline.fit(X_train, y_train)
        
        # Evaluate on train (to check for overfitting)
        y_train_pred = pipeline.predict(X_train)
        train_metrics = evaluate_predictions(y_train, y_train_pred)
        
        # Evaluate on test
        y_test_pred = pipeline.predict(X_test)
        test_metrics = evaluate_predictions(y_test, y_test_pred)
        
        results[name] = {
            "train": train_metrics,
            "test": test_metrics,
        }
        
        # Plot confusion matrix for test set
        plot_path = artifact_dir / "plots" / f"cm_{name.lower().replace(' ', '_')}.png"
        plot_confusion_matrix_custom(y_test, y_test_pred, valid_classes, f"Confusion Matrix: {name} (Test)", plot_path)
        
        # Save model weights
        model_path = artifact_dir / "model" / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipeline, model_path)
        
        # If Random Forest, save feature importance
        if name == "Random Forest":
            rf = pipeline.named_steps["model"]
            importances = rf.feature_importances_
            feat_names = X_train.columns
            # Sort importances
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(importances)), importances[indices], align="center")
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels(feat_names[indices], rotation=90)
            ax.set_title("Random Forest Feature Importances")
            plt.tight_layout()
            plt.savefig(artifact_dir / "plots" / "rf_feature_importance.png", dpi=150)
            plt.close(fig)

    # ── 6. Save results ────────────────────────────────────────────────────
    save_json(results, artifact_dir / "metrics.json")
    
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Test Balanced Accuracy: {res['test']['balanced_accuracy']:.4f}")
        print(f"  Test Macro F1:          {res['test']['f1_macro']:.4f}")
        print(f"  Test MAE:               {res['test']['mae']:.4f}")
        print(f"  Train Balanced Acc:     {res['train']['balanced_accuracy']:.4f}")
    
    print(f"\nArtifacts saved to: {artifact_dir}")

if __name__ == "__main__":
    main()
