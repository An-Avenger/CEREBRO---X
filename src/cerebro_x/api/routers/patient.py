"""
Router: Enhanced Patient Endpoints.

Provides detailed, subject-level retrieval of clinical history,
longitudinal Brain Twin (Z_t) trajectories, BHI, and disease progression.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from cerebro_x.api.database import get_db
from cerebro_x.api.models import Patient, PredictionRecord
from cerebro_x.api.services.model_loader import get_registry
from cerebro_x.api.services.inference import extract_brain_twin
from cerebro_x.brain_twin.bhi import BrainHealthIndex

router = APIRouter(prefix="/patient", tags=["Patient"])


@router.get("/{subject_id}", summary="Get patient overview")
def get_patient(subject_id: str, db: Session = Depends(get_db)):
    """Retrieve the high-level overview and history for a specific patient."""
    patient = db.query(Patient).filter(Patient.id == subject_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {subject_id} not found in database.")

    records = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.patient_id == subject_id)
        .order_by(PredictionRecord.timestamp.asc())
        .all()
    )
    
    if not records:
        return {"subject_id": subject_id, "status": "No visits recorded", "visits": []}

    latest = records[-1]
    return {
        "subject_id": subject_id,
        "latest_visit": {
            "age": latest.age,
            "mmse": latest.mmse,
            "cdr": latest.cdr,
            "predicted_cdr": latest.predicted_cdr_value,
            "timestamp": latest.timestamp,
        },
        "total_visits": len(records),
        "history": [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "age": r.age,
                "mmse": r.mmse,
                "cdr": r.cdr,
                "predicted_cdr": r.predicted_cdr_value
            } for r in records
        ]
    }


@router.get("/{subject_id}/trajectory", summary="Get patient Z_t trajectory")
def get_patient_trajectory(subject_id: str, db: Session = Depends(get_db), registry=Depends(get_registry)):
    """Extract the full longitudinal Brain Twin (Z_t) trajectory for the patient."""
    if not registry.status.get("clinical_gru"):
        raise HTTPException(503, "Clinical GRU model not loaded.")

    records = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.patient_id == subject_id)
        .order_by(PredictionRecord.timestamp.asc())
        .all()
    )
    
    if not records:
        raise HTTPException(status_code=404, detail="No clinical history available to build trajectory.")

    # Convert records to format expected by extract_brain_twin
    visits = []
    for r in records:
        # We simulate the visit structure. In a real system, the raw inputs would be stored.
        # This is a simplified reconstruction for the research prototype.
        visits.append({
            "subject_id": r.patient_id,
            "age": r.age,
            "educ": r.educ,
            "ses": r.ses,
            "mmse": r.mmse,
            "cdr": r.cdr,
            "nwbv": r.nwbv,
            "etiv": r.etiv,
            "asf": r.asf,
            # Approximations since we don't store previous visit info in the prediction record:
            "mmse_delta": 0.0, 
            "cdr_delta": 0.0,
            "nwbv_delta": 0.0,
            "days_since_last_visit": 365,
        })
        
    try:
        result = extract_brain_twin(registry.models["clinical_gru"], visits)
    except Exception as e:
        raise HTTPException(500, f"Trajectory extraction error: {e}")

    return {
        "subject_id": subject_id,
        "n_visits": len(visits),
        "trajectory": result["z_t_trajectory"],
        "dimensions": result["latent_dim"]
    }


@router.get("/{subject_id}/bhi", summary="Get patient Brain Health Index")
def get_patient_bhi(subject_id: str, db: Session = Depends(get_db), registry=Depends(get_registry)):
    """Compute the longitudinal Brain Health Index (BHI) trajectory for the patient."""
    # We require the history to compute BHI
    records = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.patient_id == subject_id)
        .order_by(PredictionRecord.timestamp.asc())
        .all()
    )
    
    if not records:
        raise HTTPException(status_code=404, detail="No clinical history available to build BHI.")
        
    bhi_calculator = BrainHealthIndex()
    bhi_trajectory = []
    
    for r in records:
        # Reconstruct probability dict
        probs = r.class_probabilities if r.class_probabilities else {}
        
        # Calculate BHI for this visit
        bhi_result = bhi_calculator.compute(
            prediction_probs=probs,
            mmse=r.mmse if r.mmse is not None else 30.0,
            nwbv=r.nwbv if r.nwbv is not None else 0.8,
            age=r.age if r.age is not None else 70.0,
            educ=r.educ if r.educ is not None else 12.0
        )
        
        bhi_trajectory.append({
            "timestamp": r.timestamp,
            "visit_age": r.age,
            **bhi_result
        })
        
    return {
        "subject_id": subject_id,
        "bhi_trajectory": bhi_trajectory,
        "latest_bhi": bhi_trajectory[-1]["bhi"] if bhi_trajectory else None
    }


@router.get("/{subject_id}/progression", summary="Get predicted disease progression")
def get_patient_progression(subject_id: str, db: Session = Depends(get_db)):
    """Retrieve progression analysis and risk factors based on latest prediction."""
    patient = db.query(Patient).filter(Patient.id == subject_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
        
    records = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.patient_id == subject_id)
        .order_by(PredictionRecord.timestamp.asc())
        .all()
    )
    
    if not records:
        raise HTTPException(status_code=404, detail="No prediction history available.")
        
    latest = records[-1]
    
    # Simple heuristic risk stratification for the research prototype
    risk_level = "Low"
    if latest.predicted_cdr_value > 0.5:
        risk_level = "High"
    elif latest.predicted_cdr_value > 0.2:
        risk_level = "Moderate"
        
    return {
        "subject_id": subject_id,
        "current_cdr": latest.cdr,
        "predicted_next_cdr": latest.predicted_cdr_value,
        "risk_level": risk_level,
        "primary_risk_factors": [
            # In a real system this would map from SHAP values
            {"factor": "Age", "value": latest.age},
            {"factor": "MMSE Decline", "value": "Present" if len(records) > 1 and records[-2].mmse > latest.mmse else "Stable"},
            {"factor": "Brain Volume", "value": latest.nwbv}
        ],
        "biomarker_status": {
            "apoe4": latest.apoe4,
            "p_tau": latest.p_tau
        }
    }
