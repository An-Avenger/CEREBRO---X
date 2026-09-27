"""
Router: CDR prediction endpoints.
Exposes clinical GRU and bimodal fusion models.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from cerebro_x.api.database import get_db
from cerebro_x.api.models import Patient, PredictionRecord
from cerebro_x.api.schemas.patient import (
    PredictionRequest,
    BimodalPredictionRequest,
    PredictionResponse,
)
from cerebro_x.api.services.model_loader import get_registry
from cerebro_x.api.services.inference import run_clinical_inference, run_bimodal_inference
from cerebro_x.features.preprocessor import CDR_CLASS_TO_VALUE, CDR_CLASS_TO_LABEL

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "/clinical",
    response_model=PredictionResponse,
    summary="Predict next-visit CDR (Clinical GRU)",
    description=(
        "Predicts the next-visit CDR class using the trained longitudinal GRU model "
        "(EXP-LONGITUDINAL-001). Accepts one or more chronological visit records. "
        "Test accuracy: 75.0% | Balanced accuracy: 55.7% | Dataset: OASIS-2 (150 subjects)."
    ),
)
def predict_clinical(
    request: PredictionRequest,
    registry=Depends(get_registry),
    db: Session = Depends(get_db),
):
    if not registry.status.get("clinical_gru"):
        raise HTTPException(503, "Clinical GRU model not loaded. Check artifacts/EXP-LONGITUDINAL-001/")

    visits = [v.model_dump() for v in request.visits]
    preprocessor = registry.scalers.get("clinical_preprocessor")

    try:
        result = run_clinical_inference(
            registry.models["clinical_gru"], visits, preprocessor=preprocessor
        )
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    # Save to database
    patient = db.query(Patient).filter(Patient.id == request.subject_id).first()
    if not patient:
        patient = Patient(id=request.subject_id)
        db.add(patient)

    last_visit = request.visits[-1]
    record = PredictionRecord(
        patient_id=request.subject_id,
        age=last_visit.age,
        educ=last_visit.educ,
        ses=last_visit.ses,
        mmse=last_visit.mmse,
        cdr=last_visit.cdr,
        nwbv=last_visit.nwbv,
        etiv=last_visit.etiv,
        asf=last_visit.asf,
        predicted_cdr_class=result["predicted_cdr_class"],
        predicted_cdr_value=result["predicted_cdr_value"],
        predicted_cdr_label=result["predicted_cdr_label"],
        class_probabilities=result["class_probabilities"],
        apoe4=last_visit.apoe4,
        p_tau=last_visit.p_tau,
    )
    db.add(record)
    db.commit()

    return PredictionResponse(subject_id=request.subject_id, **result)


@router.post(
    "/bimodal",
    response_model=PredictionResponse,
    summary="Predict next-visit CDR (Clinical + MRI Scalar Bimodal Fusion)",
    description=(
        "Predicts the next-visit CDR class using the bimodal fusion model that combines "
        "clinical visit history (GRU) with MRI-derived scalars (nWBV, eTIV, ASF). "
        "MRI inputs are OASIS-2 scalar features — NOT raw NIfTI CNN embeddings. "
        "(EXP-FUSION-BIMODAL-001). Test accuracy: 71.4% | Balanced accuracy: 52.9%"
    ),
)
def predict_bimodal(
    request: BimodalPredictionRequest,
    registry=Depends(get_registry),
):
    if not registry.status.get("bimodal_fusion"):
        raise HTTPException(503, "Bimodal fusion model not loaded. Check artifacts/EXP-FUSION-BIMODAL-001/")

    visits = [v.model_dump() for v in request.visits]
    mri = request.mri.model_dump()
    preprocessor = registry.scalers.get("clinical_preprocessor")

    try:
        result = run_bimodal_inference(
            registry.models["bimodal_fusion"], visits, mri, preprocessor=preprocessor
        )

        # Apply NextGen Biomarker Adjustment if provided (APOE4 / p-tau)
        last_visit = request.visits[-1]
        if last_visit.apoe4 is not None or last_visit.p_tau is not None:
            probs = dict(result["class_probabilities"])
            risk_multiplier = 1.0
            if last_visit.apoe4:
                risk_multiplier += 0.15
            if last_visit.p_tau is not None and last_visit.p_tau > 21.7:
                risk_multiplier += 0.20

            if risk_multiplier > 1.0:
                shift = min(probs["0"] * (risk_multiplier - 1.0), probs["0"])
                probs["0"] -= shift
                probs["1"] += shift * 0.5
                probs["2"] += shift * 0.3
                probs["3"] += shift * 0.2
                total = sum(probs.values())
                probs = {k: v / total for k, v in probs.items()}

                pred_class_str = max(probs, key=probs.get)
                pred_class = int(pred_class_str)
                result["class_probabilities"] = probs
                result["predicted_cdr_class"] = pred_class
                result["predicted_cdr_value"] = CDR_CLASS_TO_VALUE[pred_class]
                result["predicted_cdr_label"] = CDR_CLASS_TO_LABEL[pred_class]

    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    return PredictionResponse(subject_id=request.subject_id, **result)
