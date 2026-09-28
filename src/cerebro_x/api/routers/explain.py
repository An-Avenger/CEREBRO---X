"""
Router: Explainability Endpoints.

/explain/clinical — Real SHAP (GradientExplainer on TemporalCerebroNet)
                    Falls back to heuristic only when background artifact
                    is unavailable (clearly labelled "method": "heuristic_fallback").

/explain/mri      — Real 3D Grad-CAM (hook-based) when CNN checkpoint loaded;
                    returns honest 503 when checkpoint unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Form

from cerebro_x.api.schemas.patient import PredictionRequest
from cerebro_x.api.services.model_loader import get_registry
from cerebro_x.features.preprocessor import FEATURE_ORDER

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explainability"])

# Human-readable feature names (same order as FEATURE_ORDER)
_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "curr_age":            "Age (current)",
    "sex":                 "Sex",
    "hand":                "Handedness",
    "educ":                "Education (years)",
    "ses":                 "Socioeconomic Status",
    "curr_mmse":           "MMSE (current)",
    "curr_cdr":            "CDR (current)",
    "curr_etiv":           "eTIV (current)",
    "curr_nwbv":           "nWBV (current)",
    "curr_asf":            "ASF (current)",
    "current_mr_delay":    "MR Delay",
    "days_between_visits": "Days Between Visits",
    "n_prior_visits":      "Prior Visit Count",
    "prev_mmse":           "MMSE (previous)",
    "prev_cdr":            "CDR (previous)",
    "prev_nwbv":           "nWBV (previous)",
    "mmse_delta":          "MMSE Δ (change)",
    "cdr_delta":           "CDR Δ (change)",
    "nwbv_delta":          "nWBV Δ (change)",
}

CDR_LABELS = {
    0: "Normal (CDR 0.0)",
    1: "Very Mild Dementia (CDR 0.5)",
    2: "Mild Dementia (CDR 1.0)",
    3: "Moderate Dementia (CDR 2.0)",
}


# ─── Clinical SHAP (Real GradientExplainer) ───────────────────────────────────

@router.post("/clinical", summary="Real SHAP feature attributions (GradientExplainer)")
def explain_clinical(
    request: PredictionRequest,
    registry=Depends(get_registry),
):
    """
    Feature attributions for the clinical GRU prediction via real SHAP.

    Uses shap.GradientExplainer on the trained TemporalCerebroNet with
    a pre-built background artifact derived from the OASIS-2 training split.

    Returns:
        - Aggregated feature importance (mean |SHAP| across time steps)
        - Raw temporal SHAP attributions per visit
        - Prediction with class and probability
        - Method clearly labelled as "SHAP_GradientExplainer" or "heuristic_fallback"

    If the SHAP background artifact is unavailable, returns a heuristic
    approximation clearly labelled as "heuristic_fallback" (never as SHAP).
    """
    # ── 1. Check clinical model ───────────────────────────────────────────────
    if not registry.status.get("clinical_gru"):
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error_code": "CLINICAL_MODEL_UNAVAILABLE",
                "message": (
                    "Clinical GRU model is not loaded. "
                    "Ensure artifacts/EXP-LONGITUDINAL-001/temporal_gru_cpu.pt exists."
                ),
            },
        )

    visits = request.visits
    if not visits:
        raise HTTPException(status_code=400, detail={"error": "No visits provided."})

    # ── 2. Check SHAP background ──────────────────────────────────────────────
    shap_available = registry.status.get("shap_background", False)

    if not shap_available:
        # Clearly-labeled heuristic fallback — never presented as SHAP
        return _heuristic_fallback(request, registry)

    # ── 3. Preprocess visits using the canonical ClinicalPreprocessor ─────────
    try:
        from cerebro_x.features.preprocessor import ClinicalPreprocessor
        from cerebro_x.api.services.inference import visits_to_feature_tensor

        preprocessor: Optional[ClinicalPreprocessor] = registry.scalers.get(
            "clinical_preprocessor"
        )
        test_tensor = visits_to_feature_tensor(
            [v.model_dump() for v in visits],
            preprocessor,
        )  # (1, seq_len, 19)
        test_lengths = torch.tensor([len(visits)], dtype=torch.long)

    except Exception as e:
        logger.error("Preprocessing failed for SHAP: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "PREPROCESSING_FAILED",
                "message": f"Clinical preprocessing failed: {e}",
            },
        )

    # ── 4. Get clinical prediction ────────────────────────────────────────────
    model = registry.models["clinical_gru"]
    try:
        with torch.no_grad():
            logits = model(test_tensor, test_lengths)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
        pred_class = int(np.argmax(probs))
        pred_prob = float(probs[pred_class])
    except Exception as e:
        logger.error("Model inference failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INFERENCE_FAILED",
                "message": f"Model inference failed: {e}",
            },
        )

    # ── 5. Run real SHAP GradientExplainer ───────────────────────────────────
    try:
        from cerebro_x.explainability.shap_temporal import (
            compute_temporal_shap,
            TemporalModelWrapper,
        )

        bg_tensor: torch.Tensor = registry.models["shap_background"]    # (N, seq_len, 19)
        bg_lengths: torch.Tensor = registry.models["shap_bg_lengths"]   # (N,)

        # Pad test tensor to match background sequence length if needed
        bg_seq_len = bg_tensor.size(1)
        test_seq_len = test_tensor.size(1)

        if test_seq_len < bg_seq_len:
            pad = torch.zeros(1, bg_seq_len - test_seq_len, 19)
            test_tensor_padded = torch.cat([test_tensor, pad], dim=1)
        elif test_seq_len > bg_seq_len:
            # Truncate bg to match test — prefer longer test sequences
            bg_tensor = bg_tensor[:, :test_seq_len, :]
            test_tensor_padded = test_tensor
        else:
            test_tensor_padded = test_tensor

        # compute_temporal_shap uses model.eval() internally
        shap_values_raw = compute_temporal_shap(
            model=model,
            background_data=bg_tensor,
            background_lengths=bg_lengths,
            test_data=test_tensor_padded,
            test_lengths=test_lengths,
        )

    except Exception as e:
        logger.error("SHAP computation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "SHAP_COMPUTATION_FAILED",
                "message": f"SHAP GradientExplainer failed: {e}",
            },
        )

    # ── 6. Parse SHAP output ──────────────────────────────────────────────────
    # GradientExplainer returns list of length num_classes, each (batch, seq, feat)
    # OR a single ndarray of shape (num_classes, batch, seq, feat) — version-dependent
    try:
        shap_arr = _extract_shap_array(shap_values_raw, pred_class)
        # shap_arr: (seq_len, 19) for sample 0
        seq_len_used = min(len(visits), shap_arr.shape[0])
        shap_for_patient = shap_arr[:seq_len_used]   # (actual_visits, 19)

    except Exception as e:
        logger.error("SHAP output parsing failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "SHAP_PARSE_FAILED",
                "message": f"Could not parse SHAP output: {e}",
            },
        )

    # ── 7. Build response ─────────────────────────────────────────────────────
    # Aggregate: mean |SHAP| across time steps → feature importance
    mean_abs_shap = np.abs(shap_for_patient).mean(axis=0)  # (19,)
    total = float(mean_abs_shap.sum()) + 1e-9

    # Sort by importance (descending)
    feature_importances = []
    for i, feat_name in enumerate(FEATURE_ORDER):
        shap_col = shap_for_patient[:, i]        # (visits,)
        mean_shap = float(shap_col.mean())
        mean_abs = float(mean_abs_shap[i])
        feature_importances.append({
            "name": _FEATURE_DISPLAY_NAMES.get(feat_name, feat_name),
            "feature_key": feat_name,
            "importance": round(mean_abs, 6),
            "mean_abs_shap": round(mean_abs, 6),
            "mean_shap": round(mean_shap, 6),
            "direction": "positive" if mean_shap >= 0 else "negative",
            "relative_importance_pct": round(100.0 * mean_abs / total, 2),
        })
    feature_importances.sort(key=lambda x: x["importance"], reverse=True)

    # Raw temporal attributions per visit
    temporal_attributions = []
    for t in range(seq_len_used):
        visit_shap = {
            feat_name: round(float(shap_for_patient[t, i]), 6)
            for i, feat_name in enumerate(FEATURE_ORDER)
        }
        temporal_attributions.append({
            "visit": t,
            "shap_values": visit_shap,
        })

    # Background metadata
    bg_meta = registry.metrics.get("shap_background_meta", {})

    return {
        "success": True,
        "method": "SHAP_GradientExplainer",
        "model": "TemporalCerebroNet (EXP-LONGITUDINAL-001)",
        "subject_id": request.subject_id,
        "prediction": {
            "class": pred_class,
            "label": CDR_LABELS.get(pred_class, "Unknown"),
            "probability": round(pred_prob, 4),
            "all_probabilities": {
                str(i): round(float(probs[i]), 4) for i in range(len(probs))
            },
        },
        "features": feature_importances,
        "temporal_attributions": temporal_attributions,
        "background": {
            "source": bg_meta.get("source", "OASIS-2 training split"),
            "num_samples": bg_meta.get("n_samples", int(bg_tensor.size(0))),
            "contamination": bg_meta.get("contamination", "none — test set excluded"),
            "build_seed": bg_meta.get("build_seed", 42),
        },
        "disclaimer": (
            "SHAP GradientExplainer attributions reflect the contribution of each "
            "clinical feature to the model's prediction. Research use only. "
            "Not validated for clinical decision-making."
        ),
    }


def _extract_shap_array(shap_values_raw: Any, pred_class: int) -> np.ndarray:
    """
    Extract a (batch, seq, features) numpy array from the raw SHAP output.

    shap.GradientExplainer returns one of:
      - list of length num_classes, each ndarray (batch, seq, feat)
      - ndarray of shape (num_classes, batch, seq, feat)   [some versions]
      - ndarray of shape (batch, seq, feat)                [single-class output]

    We always return (seq, features) for the first (and only) patient.
    """
    if isinstance(shap_values_raw, list):
        # Most common: list of arrays per class
        vals = shap_values_raw[pred_class]  # (batch, seq, feat)
    elif isinstance(shap_values_raw, np.ndarray):
        if shap_values_raw.ndim == 4:
            # (batch, seq, feat, num_classes) OR (num_classes, batch, seq, feat)
            if shap_values_raw.shape[-1] == 4:
                vals = shap_values_raw[..., pred_class]  # (batch, seq, feat)
            else:
                vals = shap_values_raw[pred_class]  # (batch, seq, feat)
        elif shap_values_raw.ndim == 3:
            vals = shap_values_raw  # (batch, seq, feat)
        else:
            raise ValueError(f"Unexpected SHAP array shape: {shap_values_raw.shape}")
    else:
        raise TypeError(f"Unexpected SHAP output type: {type(shap_values_raw)}")

    if isinstance(vals, torch.Tensor):
        vals = vals.detach().numpy()
    elif not isinstance(vals, np.ndarray):
        vals = np.array(vals)

    # vals: (batch, seq, feat) — take first sample
    return vals[0]  # (seq, feat)


def _heuristic_fallback(request: PredictionRequest, registry) -> dict:
    """
    Heuristic attribution fallback — used ONLY when SHAP background is unavailable.

    Explicitly labelled 'heuristic_fallback'. Never presented as SHAP.
    """
    visits = request.visits
    last_visit = visits[-1]
    base_value = 0.45

    attributions = [
        {
            "name": "Age (current)",
            "feature_key": "curr_age",
            "contribution": round((last_visit.age - 75) * 0.02, 3),
            "note": "heuristic_linear",
        },
        {
            "name": "MMSE (current)",
            "feature_key": "curr_mmse",
            "contribution": round((28 - (last_visit.mmse or 28)) * 0.05, 3),
            "note": "heuristic_linear",
        },
        {
            "name": "nWBV (current)",
            "feature_key": "curr_nwbv",
            "contribution": round((0.80 - (last_visit.nwbv or 0.75)) * 2.0, 3),
            "note": "heuristic_linear",
        },
        {
            "name": "CDR (current)",
            "feature_key": "curr_cdr",
            "contribution": round(last_visit.cdr * 0.4, 3),
            "note": "heuristic_linear",
        },
    ]

    return {
        "success": True,
        "method": "heuristic_fallback",
        "model": "TemporalCerebroNet (EXP-LONGITUDINAL-001)",
        "subject_id": request.subject_id,
        "warning": (
            "SHAP background artifact is unavailable. Returning heuristic "
            "linear approximation instead. Run: "
            "python scripts/build_shap_background.py to enable real SHAP."
        ),
        "base_value": base_value,
        "features": sorted(attributions, key=lambda x: abs(x["contribution"]), reverse=True),
        "disclaimer": (
            "These attributions are heuristic linear approximations, "
            "NOT SHAP values. They do NOT reflect the trained model's "
            "internal feature weights."
        ),
    }


# ─── Real MRI Grad-CAM ───────────────────────────────────────────────────────

@router.post(
    "/mri",
    summary="Real 3D Grad-CAM for MRI (requires CNN checkpoint + NIfTI file)",
)
async def explain_mri(
    file: UploadFile = File(..., description="NIfTI MRI file (.nii or .nii.gz)"),
    target_class: Optional[int] = Form(None),
    registry=Depends(get_registry),
):
    """
    Generate a real 3D Grad-CAM heatmap from a T1-weighted MRI NIfTI file.

    Pipeline:
      1. Validate and load the NIfTI file (nibabel)
      2. Preprocess: normalize, resize to 64×64×64
      3. Run Lightweight3DCNN forward pass
      4. Run Grad-CAM backward pass on target_class
      5. Upsample heatmap to MRI dimensions
      6. Generate axial / sagittal / coronal slice PNGs

    Requires:
      - CNN3D checkpoint loaded (registry.status["cnn3d"] == True)
      - nibabel installed

    Returns:
      - target_class and probability from the CNN3D
      - base64-encoded PNG images (axial, sagittal, coronal overlays)
      - 64-dim MRI embedding
      - heatmap statistics

    If CNN3D checkpoint is not available, returns an honest 503 with
    explanation — does NOT return hardcoded scores.
    """
    # ── 1. Check CNN3D availability ───────────────────────────────────────────
    if not registry.status.get("cnn3d"):
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error_code": "CNN3D_CHECKPOINT_NOT_AVAILABLE",
                "error": (
                    "The 3D CNN model for MRI processing has not been trained yet. "
                    "No checkpoint exists in artifacts/EXP-MRI-CNN3D-001/. "
                    "To train: run python scripts/train_cnn3d.py "
                    "(requires MRI NIfTI dataset). "
                    "Grad-CAM is not available without a trained model."
                ),
                "cnn3d_architecture": "Lightweight3DCNN (in models/deep/cnn3d.py)",
                "input_shape": "(1, 1, 64, 64, 64)",
                "embedding_dim": 64,
                "heatmap_available": False,
            },
        )

    # ── 2. Validate file extension ────────────────────────────────────────────
    filename = file.filename or "upload.nii"
    fname_lower = filename.lower()
    if not (fname_lower.endswith(".nii") or fname_lower.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "INVALID_EXTENSION",
                "error": f"Expected .nii or .nii.gz, got: {filename}",
            },
        )

    # ── 3. Read and process the MRI ───────────────────────────────────────────
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(500, detail={"error": f"File read error: {e}"})

    try:
        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes
    except ImportError as e:
        raise HTTPException(503, detail={"error": f"Pipeline unavailable: {e}"})

    proc = process_mri_bytes(file_bytes, filename, subject_id="EXPLAINABILITY")

    if not proc.success:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_code": proc.error_code,
                "error": proc.error_message,
            },
        )

    # ── 4. Run real Grad-CAM ──────────────────────────────────────────────────
    try:
        from cerebro_x.explainability.gradcam_3d import run_gradcam_3d_cnn

        cnn_model = registry.models["cnn3d"]
        preprocessed = proc.preprocessed   # (1, D, H, W)

        # Add batch dim: (1, 1, D, H, W)
        tensor_5d = preprocessed.unsqueeze(0)

        cam_result = run_gradcam_3d_cnn(
            model=cnn_model,
            preprocessed=tensor_5d,
            target_class=target_class,
        )
    except Exception as e:
        logger.error("Grad-CAM failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "GRADCAM_RUNTIME_ERROR",
                "error": str(e),
            },
        )

    if not cam_result.success:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": cam_result.error_code,
                "error": cam_result.error_message,
            },
        )

    # ── 5. Build response ──────────────────────────────────────────────────────
    heatmap_stats = {}
    if cam_result.heatmap_3d is not None:
        h = cam_result.heatmap_3d
        heatmap_stats = {
            "min": float(h.min()),
            "max": float(h.max()),
            "mean": float(h.mean()),
            "shape": list(h.shape),
        }

    return {
        "success": True,
        "filename": filename,
        "heatmap_available": cam_result.success,
        "explanation_method": cam_result.method,
        "target_layer": cam_result.layer_name,
        "prediction": {
            "class": cam_result.target_class,
            "label": CDR_LABELS.get(cam_result.target_class or 0, "Unknown"),
            "probability": cam_result.target_class_probability,
        },
        "mri_embedding": {
            "values": cam_result.embedding,
            "dimension": len(cam_result.embedding) if cam_result.embedding else 0,
            "model": "Lightweight3DCNN",
        },
        "heatmap_stats": heatmap_stats,
        "visualizations": cam_result.slices_base64,
        "mri_processing": {
            "original_shape": list(proc.scalars.volume_shape) if proc.scalars else None,
            "preprocessed_to": [64, 64, 64],
            "nwbv_proxy": proc.scalars.nwbv_proxy if proc.scalars else None,
        },
        "model": "Lightweight3DCNN (EXP-MRI-CNN3D-001)",
        "disclaimer": (
            "Grad-CAM heatmap shows regions most influential for the CNN prediction. "
            "This is NOT equivalent to a clinical radiology report. "
            "Research use only."
        ),
    }
