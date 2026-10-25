"""
Router: EEG endpoints.
Exposes the standalone EEG binary disease screener (ds004504 cohort).
"""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

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

    model_config = ConfigDict(json_schema_extra={
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
    })


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

    # Build feature vector in the same order as training (107 features total)
    # Feature order: 19 channels x 5 bands (0-94), followed by global features (95-106)
    CHANNELS = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"]
    BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

    n_features = registry.scalers["eeg_scaler"].n_features_in_
    features = [float("nan")] * n_features

    # 1. Fill 19 channels with global band approximations if individual electrode values not supplied
    band_map = {
        "delta": request.global_delta,
        "theta": request.global_theta,
        "alpha": request.global_alpha,
        "beta": request.global_beta,
        "gamma": request.global_gamma,
    }
    idx = 0
    for ch in CHANNELS:
        for b in BANDS:
            if idx < 95:
                features[idx] = band_map[b]
                idx += 1

    # 2. Fill global features at indices 95-106 (exact dataset alignment)
    global_vals = [
        request.global_delta,
        request.global_theta,
        request.global_alpha,
        request.global_beta,
        request.global_gamma,
        request.rel_global_delta if request.rel_global_delta is not None else float("nan"),
        request.rel_global_theta if request.rel_global_theta is not None else float("nan"),
        request.rel_global_alpha if request.rel_global_alpha is not None else float("nan"),
        request.rel_global_beta if request.rel_global_beta is not None else float("nan"),
        request.rel_global_gamma if request.rel_global_gamma is not None else float("nan"),
        request.theta_alpha_ratio if request.theta_alpha_ratio is not None else float("nan"),
        request.spectral_edge_freq_95 if request.spectral_edge_freq_95 is not None else float("nan"),
    ]
    for i, val in enumerate(global_vals):
        if 95 + i < n_features:
            features[95 + i] = val

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
