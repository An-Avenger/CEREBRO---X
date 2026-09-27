from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from cerebro_x.api.database import get_db
from cerebro_x.api.models import Patient, PredictionRecord

router = APIRouter(tags=["History"])


@router.get("/history/", summary="Get all prediction records")
def get_all_history(limit: int = 200, db: Session = Depends(get_db)):
    """Return all prediction records across all patients, newest first."""
    records = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "patient_id": r.patient_id,
            "timestamp": r.timestamp,
            "age": r.age,
            "mmse": r.mmse,
            "cdr": r.cdr,
            "predicted_cdr_class": r.predicted_cdr_class,
            "predicted_cdr_value": r.predicted_cdr_value,
            "predicted_cdr_label": r.predicted_cdr_label,
        }
        for r in records
    ]


@router.delete("/history/", summary="Clear all prediction records")
def clear_all_history(db: Session = Depends(get_db)):
    """Delete all prediction records from the database."""
    deleted = db.query(PredictionRecord).delete()
    db.commit()
    return {"deleted": deleted}


@router.get("/patients/{subject_id}/history", summary="Get patient prediction history")
def get_patient_history(subject_id: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == subject_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    records = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.patient_id == subject_id)
        .order_by(PredictionRecord.timestamp.desc())
        .all()
    )

    return {
        "subject_id": subject_id,
        "history": [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "age": r.age,
                "mmse": r.mmse,
                "predicted_cdr_label": r.predicted_cdr_label,
                "predicted_cdr_value": r.predicted_cdr_value,
            }
            for r in records
        ],
    }
