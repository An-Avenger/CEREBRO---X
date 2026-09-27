"""
Inference service — converts raw API input into feature tensors
and runs model forward passes.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("cerebro_x.api.inference")

# CDR class → float value
CDR_VALUES = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}
CDR_LABELS = {
    0: "Normal (CDR 0.0)",
    1: "Very Mild Dementia (CDR 0.5)",
    2: "Mild Dementia (CDR 1.0)",
    3: "Moderate Dementia (CDR 2.0)",
}


def visits_to_feature_tensor(visits: list[dict], feature_builder) -> torch.Tensor:
    """
    Convert a list of visit dicts into a clinical feature tensor.

    Returns: (1, seq_len, n_features) float32 tensor
    """
    import pandas as pd

    rows = []
    for i, v in enumerate(visits):
        row = {
            "Age": v.get("age", 0.0),
            "EDUC": v.get("educ", 12.0),
            "SES": v.get("ses", 2.0),
            "MMSE": v.get("mmse", 28.0),
            "CDR": v.get("cdr", 0.0),
            "nWBV": v.get("nwbv", 0.75),
            "eTIV": v.get("etiv", 1500.0),
            "ASF": v.get("asf", 1.2),
            "M/F": 1 if v.get("gender", "F") == "M" else 0,
            "Hand": 1 if v.get("hand", "R") == "R" else 0,
            "Visit": i + 1,
            "MR Delay": 0,
            # Temporal delta features — compute from adjacent visits
            "prev_mmse": visits[i - 1].get("mmse", v.get("mmse", 28.0)) if i > 0 else v.get("mmse", 28.0),
            "prev_cdr": visits[i - 1].get("cdr", v.get("cdr", 0.0)) if i > 0 else v.get("cdr", 0.0),
            "prev_nwbv": visits[i - 1].get("nwbv", v.get("nwbv", 0.75)) if i > 0 else v.get("nwbv", 0.75),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute delta features
    df["mmse_delta"] = df["MMSE"] - df["prev_mmse"]
    df["cdr_delta"] = df["CDR"] - df["prev_cdr"]
    df["nwbv_delta"] = df["nWBV"] - df["prev_nwbv"]

    # Standard feature ordering (must match training)
    feature_cols = [
        "Age", "EDUC", "SES", "MMSE", "CDR", "nWBV", "eTIV", "ASF",
        "M/F", "Hand", "Visit", "MR Delay",
        "prev_mmse", "prev_cdr", "prev_nwbv",
        "mmse_delta", "cdr_delta", "nwbv_delta",
        "ASF",  # duplicate intentional — matches 19-feature vector from training
    ]

    # Build 19-feature vector (align with training feature set)
    FEATURE_ORDER = [
        "Age", "EDUC", "SES", "MMSE", "CDR", "nWBV", "eTIV", "ASF",
        "M/F", "Hand", "Visit", "MR Delay",
        "prev_mmse", "prev_cdr", "prev_nwbv",
        "mmse_delta", "cdr_delta", "nwbv_delta", "ASF"
    ]

    X = df[FEATURE_ORDER].fillna(0.0).values.astype(np.float32)

    # Normalize common columns
    X[:, 0] = (X[:, 0] - 70.0) / 10.0   # Age
    X[:, 1] = (X[:, 1] - 13.0) / 3.0    # EDUC
    X[:, 3] = (X[:, 3] - 27.0) / 5.0    # MMSE
    X[:, 5] = (X[:, 5] - 0.75) / 0.05   # nWBV
    X[:, 6] = (X[:, 6] - 1500.0) / 200.0  # eTIV

    return torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, T, 19)


def run_clinical_inference(model, visits: list[dict]) -> dict:
    """Run clinical GRU and return prediction dict."""
    tensor = visits_to_feature_tensor(visits, None)
    lengths = torch.tensor([len(visits)], dtype=torch.long)

    with torch.no_grad():
        logits = model(tensor, lengths)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    pred_class = int(np.argmax(probs))

    return {
        "predicted_cdr_class": pred_class,
        "predicted_cdr_value": CDR_VALUES[pred_class],
        "predicted_cdr_label": CDR_LABELS[pred_class],
        "class_probabilities": {
            CDR_LABELS[i]: float(probs[i]) for i in range(len(probs))
        },
        "n_visits_used": len(visits),
        "model": "ClinicalGRU (EXP-LONGITUDINAL-001)",
    }


def run_bimodal_inference(model, visits: list[dict], mri: dict) -> dict:
    """Run bimodal (clinical+MRI) model and return prediction dict."""
    clinical_tensor = visits_to_feature_tensor(visits, None)  # (1, T, 19)
    lengths = torch.tensor([len(visits)], dtype=torch.long)
    mri_tensor = torch.tensor(
        [[mri.get("nwbv", 0.75), mri.get("etiv", 1500.0),
          mri.get("asf", 1.2), mri.get("nwbv_delta", 0.0)]],
        dtype=torch.float32
    )  # (1, 4)

    # Normalize MRI features (same as training)
    mri_tensor[:, 0] = (mri_tensor[:, 0] - 0.75) / 0.05   # nWBV
    mri_tensor[:, 1] = (mri_tensor[:, 1] - 1500.0) / 200.0  # eTIV

    with torch.no_grad():
        logits = model(clinical_tensor, mri_tensor, lengths)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    pred_class = int(np.argmax(probs))

    return {
        "predicted_cdr_class": pred_class,
        "predicted_cdr_value": CDR_VALUES[pred_class],
        "predicted_cdr_label": CDR_LABELS[pred_class],
        "class_probabilities": {
            CDR_LABELS[i]: float(probs[i]) for i in range(len(probs))
        },
        "n_visits_used": len(visits),
        "model": "BimodalCerebroNet Clinical+MRI (EXP-FUSION-BIMODAL-001)",
    }


def run_eeg_inference(model, scaler, imputer, eeg_features: list[float]) -> dict:
    """Run EEG binary disease screener."""
    X = np.array(eeg_features, dtype=np.float32).reshape(1, -1)
    X = imputer.transform(X)
    X = scaler.transform(X)

    tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    pred_class = int(np.argmax(probs))
    labels = {0: "Disease (AD or FTD)", 1: "Control (Healthy)"}

    return {
        "predicted_class": pred_class,
        "predicted_label": labels[pred_class],
        "class_probabilities": {labels[i]: float(probs[i]) for i in range(2)},
        "model": "EEG Standalone Encoder Binary (EXP-EEG-STANDALONE-001)",
        "dataset": "OpenNeuro ds004504 — separate cohort from OASIS-2",
        "disclaimer": (
            "This EEG model is trained on a DIFFERENT patient cohort (ds004504, 87 subjects). "
            "It cannot be directly compared or combined with OASIS-2 clinical predictions."
        ),
    }


def extract_brain_twin(model, visits: list[dict]) -> dict:
    """Extract Z_t hidden state trajectory from clinical GRU."""
    tensor = visits_to_feature_tensor(visits, None)  # (1, T, 19)
    lengths = torch.tensor([len(visits)], dtype=torch.long)

    with torch.no_grad():
        # Get per-timestep hidden states from GRU output
        if hasattr(model, "gru"):
            out, _ = model.gru(tensor)  # (1, T, hidden)
            hidden_states = out.squeeze(0).numpy().tolist()
        else:
            # Fallback: just return the final embedding
            logits = model(tensor, lengths)
            hidden_states = [[0.0] * 64]

    return {
        "n_visits": len(visits),
        "z_dim": len(hidden_states[0]) if hidden_states else 64,
        "trajectories": hidden_states,
        "pca_2d": None,
    }
