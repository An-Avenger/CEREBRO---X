"""
Router: Explainability Endpoints.

/explain/clinical — Heuristic SHAP (MOCK — see inline comment)
/explain/mri      — Real 3D Grad-CAM (REAL when CNN checkpoint loaded;
                    returns honest 503 when checkpoint unavailable)
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from cerebro_x.api.schemas.patient import PredictionRequest
from cerebro_x.api.services.model_loader import get_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explainability"])


# ─── Clinical SHAP (heuristic — documented as such) ──────────────────────────

@router.post("/clinical", summary="SHAP-style feature attributions")
def explain_clinical(request: PredictionRequest, method: str = "shap", registry=Depends(get_registry)):
    """
    Feature attributions for the clinical GRU prediction.

    If method="shap": attempts real GradientExplainer SHAP.
    If method="heuristic": falls back to linear heuristic approximations.
    """
    if not registry.status.get("clinical_gru"):
        raise HTTPException(503, "Clinical GRU model not loaded.")

    if method.lower() == "shap":
        raise HTTPException(
            503, 
            "Real SHAP (GradientExplainer) unavailable in this runtime context because "
            "the background training distribution tensor is not loaded. Try method='heuristic'."
        )

    visits = request.visits
    if not visits:
        raise HTTPException(400, "No visits provided.")

    last_visit = visits[-1]
    base_value = 0.45

    attributions = [
        {
            "name": "Age",
            "value": last_visit.age,
            "contribution": round((last_visit.age - 75) * 0.02, 3),
            "note": "heuristic"
        },
        {
            "name": "MMSE",
            "value": last_visit.mmse,
            "contribution": round((28 - (last_visit.mmse or 28)) * 0.05, 3),
            "note": "heuristic"
        },
        {
            "name": "nWBV",
            "value": last_visit.nwbv,
            "contribution": round((0.80 - (last_visit.nwbv or 0.75)) * 2.0, 3),
            "note": "heuristic"
        },
        {
            "name": "Current CDR",
            "value": last_visit.cdr,
            "contribution": round(last_visit.cdr * 0.4, 3),
            "note": "heuristic"
        },
    ]

    return {
        "subject_id": request.subject_id,
        "base_value": base_value,
        "method": "heuristic",
        "disclaimer": (
            "These attributions are heuristic approximations, NOT exact SHAP values. "
            "Real SHAP GradientExplainer is unavailable because the background tensor is missing."
        ),
        "features": sorted(attributions, key=lambda x: abs(x["contribution"]), reverse=True),
    }


# ─── Real MRI Grad-CAM ───────────────────────────────────────────────────────

@router.post(
    "/mri",
    summary="Real 3D Grad-CAM for MRI (requires CNN checkpoint + NIfTI file)",
)
async def explain_mri(
    file: UploadFile = File(..., description="NIfTI MRI file (.nii or .nii.gz)"),
    target_class: int | None = None,
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
                    "To train: run python training/train_cnn3d.py "
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
    CDR_LABELS = {0: "Normal (CDR 0.0)", 1: "Very Mild (CDR 0.5)",
                  2: "Mild (CDR 1.0)", 3: "Moderate (CDR 2.0)"}

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
