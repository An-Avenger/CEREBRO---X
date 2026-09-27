"""
Router: Digital Brain Twin endpoints.
Exposes Z_t trajectory extraction from the clinical GRU.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from cerebro_x.api.schemas.patient import PredictionRequest, BrainTwinResponse
from cerebro_x.api.services.model_loader import get_registry
from cerebro_x.api.services.inference import extract_brain_twin

router = APIRouter(prefix="/brain-twin", tags=["Digital Brain Twin"])


@router.post(
    "/extract",
    response_model=BrainTwinResponse,
    summary="Extract Z_t Brain Twin Trajectory",
    description=(
        "Extracts the per-visit latent brain state vectors (Z_t) from the clinical GRU's "
        "hidden states. Returns a trajectory of 64-dimensional vectors, one per visit. "
        "Z_t represents a compressed, latent snapshot of cognitive state estimated from "
        "clinical history. It is NOT a neuroimaging measurement."
    ),
)
def extract_twin(request: PredictionRequest, registry=Depends(get_registry)):
    if not registry.status.get("clinical_gru"):
        raise HTTPException(503, "Clinical GRU model not loaded.")

    visits = [v.model_dump() for v in request.visits]
    try:
        result = extract_brain_twin(registry.models["clinical_gru"], visits)
    except Exception as e:
        raise HTTPException(500, f"Brain twin extraction error: {e}")

    return BrainTwinResponse(
        subject_id=request.subject_id,
        **result,
    )
