#!/usr/bin/env python
"""
scripts/build_shap_background.py
----------------------------------
Build a deterministic SHAP background artifact from the OASIS-2 training split.

This script MUST be run once before the /explain/clinical SHAP endpoint is live.
It uses the same trained ClinicalPreprocessor and the same subject-level train split
as the model training — NO test-set contamination.

Output:
    artifacts/EXP-LONGITUDINAL-001/shap_background.pt  — (N, seq_len, 19) tensor
    artifacts/EXP-LONGITUDINAL-001/shap_background_meta.json — provenance

Usage:
    python scripts/build_shap_background.py

Options:
    --n-samples  Number of background samples to select (default 50)
    --seed       Random seed (default 42)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.features.preprocessor import ClinicalPreprocessor, FEATURE_ORDER
from cerebro_x.evaluation.splits import subject_level_split


ARTIFACT_DIR = Path("artifacts/EXP-LONGITUDINAL-001")
PAIRS_CSV    = Path("data/processed/next_visit_pairs.csv")

CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}


def build_sequences_from_df(df: pd.DataFrame, prep: ClinicalPreprocessor, max_seq_len: int = 5):
    """
    Build (seq_tensor, length) for each subject from the OASIS-2 pairs DataFrame.

    Groups rows by Subject ID (training split only), applies the fitted preprocessor,
    and pads to max_seq_len.

    Returns:
        tensors: (N, max_seq_len, 19) float32 numpy array
        lengths: (N,) int array of true sequence lengths
    """
    # Map the pairs CSV columns → FEATURE_ORDER names
    col_map = {
        "curr_age": "curr_age", "sex": "sex", "hand": "hand",
        "educ": "educ", "ses": "ses", "curr_mmse": "curr_mmse",
        "curr_cdr": "curr_cdr", "curr_etiv": "curr_etiv",
        "curr_nwbv": "curr_nwbv", "curr_asf": "curr_asf",
        "current_mr_delay": "current_mr_delay",
        "days_between_visits": "days_between_visits",
        "n_prior_visits": "n_prior_visits",
        "prev_mmse": "prev_mmse", "prev_cdr": "prev_cdr", "prev_nwbv": "prev_nwbv",
        "mmse_delta": "mmse_delta", "cdr_delta": "cdr_delta", "nwbv_delta": "nwbv_delta",
    }

    # Encode sex/hand same way as ClinicalPreprocessor.visits_to_dataframe
    df = df.copy()
    sex_map = {"M": 1.0, "F": 0.0}
    hand_map = {"R": 1.0, "L": 0.0}
    df["sex"]  = df["sex"].map(sex_map)
    df["hand"] = df["hand"].map(hand_map).fillna(0.5)

    all_tensors, all_lengths = [], []

    for subject_id, grp in df.groupby("Subject ID"):
        grp_sorted = grp.sort_values("current_visit").reset_index(drop=True)
        rows = grp_sorted[list(col_map.keys())].rename(columns=col_map)
        # Ensure columns in correct order
        rows = rows[FEATURE_ORDER]
        X = prep.transform(rows)   # (T, 19)

        T = min(len(X), max_seq_len)
        X = X[:T]  # truncate to max_seq_len

        # Pad to max_seq_len
        pad_len = max_seq_len - T
        if pad_len > 0:
            pad = np.zeros((pad_len, 19), dtype=np.float32)
            X = np.vstack([X, pad])

        all_tensors.append(X)
        all_lengths.append(T)

    tensors = np.stack(all_tensors, axis=0).astype(np.float32)  # (N, max_seq_len, 19)
    lengths = np.array(all_lengths, dtype=np.int64)
    return tensors, lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of background samples (default 50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading pairs from {PAIRS_CSV}...")
    if not PAIRS_CSV.exists():
        print(f"ERROR: {PAIRS_CSV} not found. Run: python scripts/build_pairs.py")
        sys.exit(1)

    df = pd.read_csv(PAIRS_CSV)
    print(f"Total pairs: {len(df)} from {df['Subject ID'].nunique()} subjects")

    # Use the same subject-level split as training — NO test-set contamination
    train_df, val_df, _, _ = subject_level_split(df)
    train_subjects = set(train_df["Subject ID"].unique())
    train_df_all = df[df["Subject ID"].isin(train_subjects)].copy()

    print(f"Training subjects: {len(train_subjects)}, pairs: {len(train_df_all)}")

    # Load the fitted preprocessor (from training)
    print(f"Loading ClinicalPreprocessor from {ARTIFACT_DIR}...")
    prep = ClinicalPreprocessor.load(ARTIFACT_DIR)

    # Build sequences
    print("Building per-subject sequence tensors...")
    tensors, lengths = build_sequences_from_df(train_df_all, prep, max_seq_len=5)
    print(f"  Raw tensor shape: {tensors.shape}, lengths: {lengths.shape}")

    # Sample up to n_samples subjects for background (deterministic)
    N = len(tensors)
    n_bg = min(args.n_samples, N)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=n_bg, replace=False)
    idx.sort()

    bg_tensors = tensors[idx]       # (n_bg, max_seq_len, 19)
    bg_lengths = lengths[idx]       # (n_bg,)

    bg_tensor  = torch.tensor(bg_tensors, dtype=torch.float32)
    bg_lengths_t = torch.tensor(bg_lengths, dtype=torch.long)

    # Save
    out_tensor_path = ARTIFACT_DIR / "shap_background.pt"
    out_meta_path   = ARTIFACT_DIR / "shap_background_meta.json"

    torch.save({"background": bg_tensor, "lengths": bg_lengths_t}, str(out_tensor_path))

    meta = {
        "source": "OASIS-2 training split (subject-level split, seed=42)",
        "n_samples": n_bg,
        "max_seq_len": 5,
        "n_features": 19,
        "feature_order": FEATURE_ORDER,
        "split": "train (70%)",
        "contamination": "none — test set excluded",
        "build_seed": args.seed,
        "preprocessor": str(ARTIFACT_DIR / "clinical_preprocessor.pkl"),
        "background_tensor": str(out_tensor_path),
        "tensor_shape": list(bg_tensor.shape),
    }
    out_meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved background tensor: {out_tensor_path}")
    print(f"  Shape: {bg_tensor.shape}  (n_samples, seq_len, features)")
    print(f"Saved provenance:       {out_meta_path}")
    print("\nDone. You can now use /explain/clinical?method=shap")


if __name__ == "__main__":
    main()
