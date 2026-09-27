"""
Pydantic schemas for patient input and API responses.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ─── Patient Visit Input ──────────────────────────────────────────────────────

class VisitInput(BaseModel):
    """One clinical visit record for a patient."""
    age: float = Field(..., ge=18, le=120, description="Patient age in years")
    educ: Optional[float] = Field(None, description="Years of education")
    ses: Optional[float] = Field(None, ge=1, le=5, description="Socioeconomic status (1-5)")
    mmse: Optional[float] = Field(None, ge=0, le=30, description="MMSE score (0-30)")
    cdr: float = Field(..., ge=0, le=3, description="Current CDR score (0/0.5/1/2/3)")
    nwbv: Optional[float] = Field(None, description="Normalized whole brain volume")
    etiv: Optional[float] = Field(None, description="Estimated total intracranial volume (mm³)")
    asf: Optional[float] = Field(None, description="Atlas scaling factor")
    gender: Optional[str] = Field(None, description="'M' or 'F'")
    hand: Optional[str] = Field(None, description="Handedness ('R' or 'L')")
    apoe4: Optional[bool] = Field(None, description="APOE4 gene carrier status")
    p_tau: Optional[float] = Field(None, ge=0, description="p-tau181 biomarker level (pg/mL)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 74.0,
                "educ": 16.0,
                "ses": 2.0,
                "mmse": 27.0,
                "cdr": 0.5,
                "nwbv": 0.743,
                "etiv": 1502.0,
                "asf": 1.21,
                "gender": "F",
                "hand": "R",
                "apoe4": True,
                "p_tau": 24.5
            }
        }


class PredictionRequest(BaseModel):
    """Request body for CDR prediction — one or more visits in chronological order."""
    subject_id: Optional[str] = Field(None, description="Optional patient identifier")
    visits: list[VisitInput] = Field(..., min_length=1, description="Chronological list of visits")

    class Config:
        json_schema_extra = {
            "example": {
                "subject_id": "OAS2_0001",
                "visits": [
                    {"age": 72.0, "educ": 16.0, "ses": 2.0, "mmse": 29.0, "cdr": 0.0,
                     "nwbv": 0.763, "etiv": 1480.0, "asf": 1.23},
                    {"age": 74.0, "educ": 16.0, "ses": 2.0, "mmse": 27.0, "cdr": 0.5,
                     "nwbv": 0.743, "etiv": 1480.0, "asf": 1.23},
                ]
            }
        }


class MRIInput(BaseModel):
    """MRI-derived scalar inputs for a single visit."""
    nwbv: float = Field(..., description="Normalized whole brain volume")
    etiv: float = Field(..., description="Estimated total intracranial volume (mm³)")
    asf: float = Field(..., description="Atlas scaling factor")
    nwbv_delta: Optional[float] = Field(0.0, description="Change in nWBV since last visit")


class BimodalPredictionRequest(BaseModel):
    """Request body for bimodal (Clinical + MRI) prediction."""
    subject_id: Optional[str] = Field(None)
    visits: list[VisitInput] = Field(..., min_length=1)
    mri: MRIInput = Field(..., description="MRI scalars for the current visit")

    class Config:
        json_schema_extra = {
            "example": {
                "subject_id": "OAS2_0001",
                "visits": [
                    {"age": 72.0, "educ": 16.0, "ses": 2.0, "mmse": 29.0, "cdr": 0.0,
                     "nwbv": 0.763, "etiv": 1480.0, "asf": 1.23},
                    {"age": 74.0, "educ": 16.0, "ses": 2.0, "mmse": 27.0, "cdr": 0.5,
                     "nwbv": 0.743, "etiv": 1480.0, "asf": 1.23},
                ],
                "mri": {"nwbv": 0.743, "etiv": 1480.0, "asf": 1.23, "nwbv_delta": -0.02}
            }
        }


# ─── Prediction Responses ─────────────────────────────────────────────────────

CDR_LABELS = {0: "Normal (CDR 0)", 1: "Very Mild (CDR 0.5)", 2: "Mild (CDR 1)", 3: "Moderate (CDR 2)"}
CDR_VALUES = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}


class PredictionResponse(BaseModel):
    subject_id: Optional[str]
    model: str
    predicted_cdr_class: int
    predicted_cdr_value: float
    predicted_cdr_label: str
    class_probabilities: dict[str, float]
    n_visits_used: int
    disclaimer: str = (
        "Research use only. Not validated for clinical decision-making. "
        "Based on OASIS-2 longitudinal data (150 subjects)."
    )


class BrainTwinResponse(BaseModel):
    subject_id: Optional[str]
    n_visits: int
    z_dim: int
    trajectories: list[list[float]]
    pca_2d: Optional[list[list[float]]]
    disclaimer: str = (
        "Z_t is the GRU hidden state vector, representing a latent brain state "
        "estimated from clinical history. It is NOT a neuroimaging measurement."
    )


class ExperimentSummary(BaseModel):
    experiment_id: str
    phase: int
    description: str
    metrics: dict
    artifact_path: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    version: str = "1.0.0"
    dataset: str = "OASIS-2 Longitudinal (150 subjects)"
