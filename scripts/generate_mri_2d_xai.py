"""
Generate XAI explanations using MRI2DExplainer for test cases.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cerebro_x.explainability.mri_2d_shap import MRI2DExplainer
from cerebro_x.models.deep.cnn2d import MRI2DCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_test_explanations():
    artifact_dir = Path("artifacts/EXP-MRI-2D-CNN-001")
    checkpoint_path = artifact_dir / "best_checkpoint.pt"

    model = MRI2DCNN(num_classes=3, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df_manifest = pd.read_csv("data/processed/mri_2d/slices_manifest.csv")
    df_splits = pd.read_csv(artifact_dir / "subject_splits.csv")[["subject_id", "split"]]
    merged = pd.merge(df_manifest, df_splits, on="subject_id")

    train_slices = merged[merged["split"] == "train"]["cache_path"].tolist()
    bg_tensors = []
    for p in train_slices[:10]:
        arr = np.load(p).astype(np.float32)
        bg_tensors.append(torch.from_numpy(arr).unsqueeze(0))
    bg_tensors = torch.stack(bg_tensors)

    explainer = MRI2DExplainer(model, background_tensors=bg_tensors)

    test_df = merged[merged["split"] == "test"]
    explanation_metadata = []

    for cls_name in ["CN", "MCI", "AD"]:
        sub_df = test_df[test_df["label_name"] == cls_name]
        if len(sub_df) == 0:
            continue
        sample_row = sub_df.iloc[0]
        sample_path = sample_row["cache_path"]
        subj_id = str(sample_row["subject_id"])
        arr = np.load(sample_path).astype(np.float32)
        img_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        explanation = explainer.explain(img_t)
        out_fig = artifact_dir / f"shap_{cls_name}_{subj_id}.png"
        explainer.save_explanation_figure(
            arr,
            explanation,
            out_fig,
            title=f"XAI Explanation: True={cls_name} (Subject {subj_id})",
        )
        logger.info(
            "Generated XAI explanation for %s (%s): method=%s -> %s",
            cls_name,
            subj_id,
            explanation["method"],
            out_fig,
        )

        explanation_metadata.append({
            "subject_id": subj_id,
            "true_class": cls_name,
            "predicted_class": explanation["predicted_label"],
            "probabilities": explanation["probabilities"],
            "method": explanation["method"],
            "artifact_path": str(out_fig),
        })

    with open(artifact_dir / "xai_explanations.json", "w") as f:
        json.dump(explanation_metadata, f, indent=2)
    logger.info("Saved XAI explanation metadata to %s", artifact_dir / "xai_explanations.json")


if __name__ == "__main__":
    generate_test_explanations()
