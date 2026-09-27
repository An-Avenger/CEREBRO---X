#!/usr/bin/env python
"""
scripts/extract_brain_twin.py
------------------------------
Phase 6: Extract Digital Brain Twin Z_t trajectories.

Loads the trained TemporalCerebroNet GRU from EXP-LONGITUDINAL-001 and
extracts the latent brain state Z_t at each visit for every patient.

Z_t definition:
    The GRU hidden state h_n after processing the patient's clinical visit
    history up to time t. This is the formal brain state representation.

    For patient P with visits [V1, V2, V3]:
        Z_1 = GRU([V1])        — brain state after visit 1
        Z_2 = GRU([V1, V2])   — brain state after visit 2
        Z_3 = GRU([V1, V2, V3]) — brain state after visit 3

Generates:
    artifacts/EXP-BRAIN-TWIN-001/
        per_patient/            — one .npy file per patient
        brain_twin_Z_trajectories_PCA.png
        brain_twin_Z_heatmap.png
        trajectory_metadata.json

Usage:
    python scripts/extract_brain_twin.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.oasis2.loader import load_oasis2_auto
from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs
from cerebro_x.data.pytorch.sequence_dataset import SequenceDataset, CDR_MAP
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.models.deep.temporal import TemporalCerebroNet
from cerebro_x.brain_twin.extractor import ClinicalBrainTwinExtractor
from cerebro_x.utils.io import setup_logging, make_artifact_dir

logger = logging.getLogger("extract_brain_twin")

SEED = 42
GRU_CHECKPOINT = Path("artifacts/EXP-LONGITUDINAL-001/temporal_gru_cpu.pt")


def main():
    setup_logging("INFO")

    artifact_dir = make_artifact_dir("artifacts", "EXP-BRAIN-TWIN-001")
    per_patient_dir = artifact_dir / "per_patient"
    per_patient_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = artifact_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Artifact dir: %s", artifact_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    pairs_path = Path("data/processed/next_visit_pairs.csv")
    if not pairs_path.exists():
        logger.info("Building pairs...")
        raw_df = load_oasis2_auto()
        pairs_df, _ = build_next_visit_pairs(raw_df)
    else:
        pairs_df = pd.read_csv(pairs_path)
    logger.info("Loaded %d pairs from %d patients.", len(pairs_df), pairs_df["Subject ID"].nunique())

    # ── Split (same seed as training) ─────────────────────────────────────────
    train_df, val_df, test_df, _ = subject_level_split(
        pairs_df, test_size=0.15, val_size=0.15, seed=SEED
    )

    # ── Build datasets for preprocessing objects ──────────────────────────────
    train_ds = SequenceDataset(train_df, is_train=True)
    num_features = train_ds.num_features

    # ── Load trained GRU model ────────────────────────────────────────────────
    if not GRU_CHECKPOINT.exists():
        logger.error(
            "GRU checkpoint not found at %s. "
            "Run scripts/train_longitudinal.py first.", GRU_CHECKPOINT
        )
        sys.exit(1)

    model = TemporalCerebroNet(
        input_dim=num_features,
        gru_hidden_dim=64,
        gru_num_layers=1,
        head_hidden_dim=32,
        dropout=0.3,
    )
    model.load_state_dict(torch.load(GRU_CHECKPOINT, map_location="cpu"))
    model.eval()
    logger.info("Loaded TemporalCerebroNet from %s (hidden_dim=64)", GRU_CHECKPOINT)

    # ── Extract Z_t trajectories ──────────────────────────────────────────────
    extractor = ClinicalBrainTwinExtractor(model, device=torch.device("cpu"))

    trajectories = extractor.extract_all_patients(
        dataset=train_ds,
        pairs_df=pairs_df,
        imputer=train_ds.imputer,
        scaler=train_ds.scaler,
    )
    logger.info("Extracted trajectories for %d patients.", len(trajectories))

    # ── Save per-patient .npy files ───────────────────────────────────────────
    extractor.save_trajectories(trajectories, per_patient_dir)

    # ── Generate visualizations ───────────────────────────────────────────────
    logger.info("Generating PCA trajectory plot...")
    extractor.visualize_trajectories_pca(
        trajectories=trajectories,
        pairs_df=pairs_df,
        output_dir=plots_dir,
        max_patients=40,
    )

    logger.info("Generating heatmap...")
    extractor.visualize_progression_heatmap(
        trajectories=trajectories,
        output_dir=plots_dir,
        max_patients=25,
        max_visits=5,
    )

    # ── Statistics ────────────────────────────────────────────────────────────
    all_traj_lengths = [len(t) for t in trajectories.values()]
    traj_metadata = {
        "experiment_id": "EXP-BRAIN-TWIN-001",
        "phase": 6,
        "description": "Digital Brain Twin — Z_t trajectory extraction from clinical GRU",
        "model_checkpoint": str(GRU_CHECKPOINT),
        "z_t_definition": (
            "Z_t is the GRU final hidden state h_n[-1] of shape (64,) "
            "after processing clinical visit history up to time t. "
            "See DOCS/DIGITAL_BRAIN_TWIN_SPECIFICATION.md for formal definition."
        ),
        "hidden_dim": 64,
        "num_patients": len(trajectories),
        "trajectory_lengths": {
            "min": int(min(all_traj_lengths)),
            "max": int(max(all_traj_lengths)),
            "mean": float(np.mean(all_traj_lengths)),
        },
        "split_breakdown": {
            "train_patients": train_df["Subject ID"].nunique(),
            "val_patients": val_df["Subject ID"].nunique(),
            "test_patients": test_df["Subject ID"].nunique(),
        },
        "outputs": {
            "per_patient_npy": str(per_patient_dir),
            "pca_plot": str(plots_dir / "brain_twin_Z_trajectories_PCA.png"),
            "heatmap": str(plots_dir / "brain_twin_Z_heatmap.png"),
        },
    }

    with open(artifact_dir / "trajectory_metadata.json", "w") as f:
        json.dump(traj_metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("PHASE 6 — DIGITAL BRAIN TWIN EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Patients processed:   %d", len(trajectories))
    logger.info("Z_t dimension:        64 (GRU hidden dim)")
    logger.info("Min/Max visits:       %d / %d", min(all_traj_lengths), max(all_traj_lengths))
    logger.info("Per-patient .npy:     %s", per_patient_dir)
    logger.info("PCA plot:             %s", plots_dir / "brain_twin_Z_trajectories_PCA.png")
    logger.info("Heatmap:              %s", plots_dir / "brain_twin_Z_heatmap.png")
    logger.info("Metadata:             %s", artifact_dir / "trajectory_metadata.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
