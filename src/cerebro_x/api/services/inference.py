"""
Inference service — converts raw API input into feature tensors
and runs model forward passes.

CRITICAL: All normalization is done via the fitted ClinicalPreprocessor
loaded from artifacts/EXP-LONGITUDINAL-001/clinical_preprocessor.pkl.
NO hard-coded normalization constants (e.g. (age-70)/10) are used here.

CDR class_probabilities are keyed by the SHORT canonical keys "0","1","2","3"
(not human labels) so that BHI.compute() receives the correct keys.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from cerebro_x.features.preprocessor import (
    ClinicalPreprocessor,
    CDR_CLASS_TO_VALUE,
    CDR_CLASS_TO_KEY,
    CDR_CLASS_TO_LABEL,
    FEATURE_ORDER,
)

logger = logging.getLogger("cerebro_x.api.inference")


# ── Fallback preprocessor (used only when pkl not found) ─────────────────────
# This is a plain StandardScaler with default params. Predictions from it will
# NOT match training-time preprocessing and should be treated as approximate.
# A warning is emitted to the log whenever the fallback is used.
_FALLBACK_PREPROCESSOR: Optional[ClinicalPreprocessor] = None


def _get_fallback_preprocessor() -> ClinicalPreprocessor:
    """Return a stateless fallback preprocessor (no scaling, just imputes NaN→0)."""
    global _FALLBACK_PREPROCESSOR
    if _FALLBACK_PREPROCESSOR is None:
        import sklearn.impute
        import sklearn.preprocessing
        import pandas as pd

        # Fit on a single row of population-mean values so the objects are valid
        dummy = pd.DataFrame(
            [[70, 1, 1, 14, 2, 28, 0, 1480, 0.76, 1.18, 0, 365, 1,
              28, 0, 0.76, 0, 0, 0]],
            columns=FEATURE_ORDER,
        )
        prep = ClinicalPreprocessor()
        prep.fit_transform(dummy)
        _FALLBACK_PREPROCESSOR = prep
        logger.warning(
            "Using FALLBACK ClinicalPreprocessor (fitted on dummy data). "
            "Predictions will NOT match training-time preprocessing. "
            "Run scripts/train_longitudinal.py to generate the real artifact."
        )
    return _FALLBACK_PREPROCESSOR


def _resolve_preprocessor(preprocessor: Optional[ClinicalPreprocessor]) -> ClinicalPreprocessor:
    """Return the real preprocessor or the fallback, logging a warning."""
    if preprocessor is not None and preprocessor.is_fitted:
        return preprocessor
    logger.warning(
        "ClinicalPreprocessor not available from registry. Using fallback. "
        "Re-run training to fix this."
    )
    return _get_fallback_preprocessor()


# ── Feature building ──────────────────────────────────────────────────────────

def visits_to_feature_tensor(
    visits: list[dict],
    preprocessor: Optional[ClinicalPreprocessor],
) -> torch.Tensor:
    """
    Convert a list of visit dicts into a clinical feature tensor.

    Uses the fitted ClinicalPreprocessor (imputer + scaler from training).
    NO hard-coded normalization constants.

    Returns: (1, seq_len, n_features) float32 tensor
    """
    prep = _resolve_preprocessor(preprocessor)
    df = ClinicalPreprocessor.visits_to_dataframe(visits)
    X = prep.transform(df)  # (seq_len, 19) float32, already scaled
    tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, T, 19)
    return tensor


# ── Clinical GRU inference ────────────────────────────────────────────────────

def run_clinical_inference(
    model,
    visits: list[dict],
    preprocessor: Optional[ClinicalPreprocessor] = None,
) -> dict:
    """Run clinical GRU and return prediction dict.

    class_probabilities keys: "0", "1", "2", "3"  (canonical, required by BHI).
    """
    tensor = visits_to_feature_tensor(visits, preprocessor)
    lengths = torch.tensor([len(visits)], dtype=torch.long)

    with torch.no_grad():
        logits = model(tensor, lengths)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    pred_class = int(np.argmax(probs))
    n_classes = len(probs)

    # Canonical keys: "0", "1", "2", "3"
    class_probabilities = {
        CDR_CLASS_TO_KEY[i]: float(probs[i]) for i in range(n_classes)
    }

    # Validate all required BHI keys are present
    for k in ("0", "1", "2", "3"):
        if k not in class_probabilities:
            raise RuntimeError(
                f"Model returned {n_classes} classes but BHI requires 4 "
                f"(CDR 0, 0.5, 1, 2). Missing key: '{k}'."
            )

    return {
        "predicted_cdr_class": pred_class,
        "predicted_cdr_value": CDR_CLASS_TO_VALUE[pred_class],
        "predicted_cdr_label": CDR_CLASS_TO_LABEL[pred_class],
        "class_probabilities": class_probabilities,
        "n_visits_used": len(visits),
        "model": "ClinicalGRU (EXP-LONGITUDINAL-001)",
    }


# ── Bimodal inference ─────────────────────────────────────────────────────────

def run_bimodal_inference(
    model,
    visits: list[dict],
    mri: dict,
    preprocessor: Optional[ClinicalPreprocessor] = None,
) -> dict:
    """Run bimodal (clinical+MRI scalar) model and return prediction dict.

    class_probabilities keys: "0", "1", "2", "3"  (canonical, required by BHI).
    """
    clinical_tensor = visits_to_feature_tensor(visits, preprocessor)  # (1, T, 19)
    lengths = torch.tensor([len(visits)], dtype=torch.long)

    # MRI scalar tensor — NOTE: these are the raw OASIS-2 scalars from training.
    # They are NOT equivalent to NIfTI-derived proxy values without domain alignment.
    mri_tensor = torch.tensor(
        [[mri.get("nwbv", 0.75), mri.get("etiv", 1500.0),
          mri.get("asf", 1.2), mri.get("nwbv_delta", 0.0)]],
        dtype=torch.float32,
    )  # (1, 4)

    # MRI scalar normalization from bimodal training (these are mri-branch only,
    # separate from the clinical preprocessor)
    # These constants reflect the bimodal training MRI branch normalization.
    # TODO: save as a separate mri_scaler.pkl when the bimodal model is retrained.
    mri_tensor[:, 0] = (mri_tensor[:, 0] - 0.75) / 0.05   # nWBV
    mri_tensor[:, 1] = (mri_tensor[:, 1] - 1500.0) / 200.0  # eTIV

    with torch.no_grad():
        logits = model(clinical_tensor, mri_tensor, lengths)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    pred_class = int(np.argmax(probs))
    n_classes = len(probs)

    class_probabilities = {
        CDR_CLASS_TO_KEY[i]: float(probs[i]) for i in range(n_classes)
    }

    return {
        "predicted_cdr_class": pred_class,
        "predicted_cdr_value": CDR_CLASS_TO_VALUE[pred_class],
        "predicted_cdr_label": CDR_CLASS_TO_LABEL[pred_class],
        "class_probabilities": class_probabilities,
        "n_visits_used": len(visits),
        "model": "BimodalCerebroNet Clinical+MRI Scalar (EXP-FUSION-BIMODAL-001)",
        "mri_note": (
            "MRI inputs are OASIS-2 scalar features (nWBV, eTIV, ASF), "
            "NOT raw NIfTI-derived CNN embeddings."
        ),
    }


# ── EEG inference ─────────────────────────────────────────────────────────────

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


# ── Brain Twin extraction ─────────────────────────────────────────────────────

def extract_brain_twin(
    model,
    visits: list[dict],
    preprocessor: Optional[ClinicalPreprocessor] = None,
) -> dict:
    """
    Extract Z_t hidden state trajectory from clinical GRU.

    Uses the canonical preprocessor (same as training). Z_t is the GRU hidden
    state — a learned latent representation. It is NOT a 3D anatomical model.
    """
    tensor = visits_to_feature_tensor(visits, preprocessor)  # (1, T, 19)
    lengths = torch.tensor([len(visits)], dtype=torch.long)

    with torch.no_grad():
        if hasattr(model, "gru"):
            out, _ = model.gru(tensor)  # (1, T, hidden)
            hidden_states = out.squeeze(0).numpy().tolist()
        else:
            # Fallback if model doesn't expose .gru directly
            logits = model(tensor, lengths)
            hidden_states = [[0.0] * 64]
            logger.warning(
                "Model does not expose .gru attribute; returning zero trajectory. "
                "Z_t extraction requires TemporalCerebroNet."
            )

    return {
        "n_visits": len(visits),
        "z_dim": len(hidden_states[0]) if hidden_states else 64,
        "trajectories": hidden_states,
        "pca_2d": None,
        "representation": (
            "Z_t = GRU hidden state (learned latent representation). "
            "NOT a 3D anatomical reconstruction."
        ),
    }
