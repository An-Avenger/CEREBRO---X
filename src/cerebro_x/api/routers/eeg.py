"""
Router: EEG endpoints.
Exposes the standalone EEG binary disease screener (ds004504 cohort).
"""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cerebro_x.api.services.model_loader import get_registry
from cerebro_x.api.services.inference import run_eeg_inference

router = APIRouter(prefix="/eeg", tags=["EEG (Standalone)"])

EEG_BAND_NAMES = [
    "global_delta", "global_theta", "global_alpha", "global_beta", "global_gamma",
    "rel_global_delta", "rel_global_theta", "rel_global_alpha", "rel_global_beta", "rel_global_gamma",
    "theta_alpha_ratio", "spectral_edge_freq_95",
]


class EEGScreenRequest(BaseModel):
    """EEG screening request using pre-extracted global band power features."""
    global_delta: float = Field(..., description="Global delta band power (1-4 Hz)")
    global_theta: float = Field(..., description="Global theta band power (4-8 Hz)")
    global_alpha: float = Field(..., description="Global alpha band power (8-13 Hz)")
    global_beta:  float = Field(..., description="Global beta band power (13-30 Hz)")
    global_gamma: float = Field(..., description="Global gamma band power (30-45 Hz)")
    rel_global_delta: Optional[float] = Field(None, description="Relative delta power")
    rel_global_theta: Optional[float] = Field(None, description="Relative theta power")
    rel_global_alpha: Optional[float] = Field(None, description="Relative alpha power")
    rel_global_beta:  Optional[float] = Field(None, description="Relative beta power")
    rel_global_gamma: Optional[float] = Field(None, description="Relative gamma power")
    theta_alpha_ratio: Optional[float] = Field(None, description="Theta/Alpha ratio")
    spectral_edge_freq_95: Optional[float] = Field(None, description="SEF95 in Hz")

    class Config:
        json_schema_extra = {
            "example": {
                "global_delta": 2.45e-10,
                "global_theta": 1.23e-10,
                "global_alpha": 3.12e-10,
                "global_beta": 0.87e-10,
                "global_gamma": 0.34e-10,
                "rel_global_delta": 0.30,
                "rel_global_theta": 0.15,
                "rel_global_alpha": 0.38,
                "rel_global_beta": 0.11,
                "rel_global_gamma": 0.06,
                "theta_alpha_ratio": 0.40,
                "spectral_edge_freq_95": 22.5
            }
        }


@router.post(
    "/screen",
    summary="EEG Binary Disease Screener",
    description=(
        "Screens for cognitive impairment (AD or FTD) vs Control using pre-extracted EEG "
        "spectral band power features. This model was trained on the OpenNeuro ds004504 "
        "dataset (87 subjects, separate cohort from OASIS-2). "
        "Binary accuracy: 77.8% | Balanced accuracy: 66.7%"
    ),
)
def eeg_screen(request: EEGScreenRequest, registry=Depends(get_registry)):
    if not registry.status.get("eeg_encoder"):
        raise HTTPException(503, "EEG encoder not loaded. Check artifacts/EXP-EEG-STANDALONE-001/")

    # Build feature vector in the same order as training
    # The model expects the full 107-feature vector — we use global features only
    # and pad the rest with NaN (imputer handles it)
    n_features = registry.scalers["eeg_scaler"].n_features_in_
    features = [float("nan")] * n_features

    # Fill in the global features at known positions (0-11)
    feature_vals = [
        request.global_delta, request.global_theta, request.global_alpha,
        request.global_beta, request.global_gamma,
        request.rel_global_delta or float("nan"),
        request.rel_global_theta or float("nan"),
        request.rel_global_alpha or float("nan"),
        request.rel_global_beta or float("nan"),
        request.rel_global_gamma or float("nan"),
        request.theta_alpha_ratio or float("nan"),
        request.spectral_edge_freq_95 or float("nan"),
    ]
    # Place at beginning of feature vector
    for i, val in enumerate(feature_vals[:n_features]):
        features[i] = val

    try:
        result = run_eeg_inference(
            registry.models["eeg_encoder"],
            registry.scalers["eeg_scaler"],
            registry.scalers["eeg_imputer"],
            features,
        )
    except Exception as e:
        raise HTTPException(500, f"EEG inference error: {e}")

    return result
