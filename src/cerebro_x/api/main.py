"""
Cerebro X — FastAPI Research Backend
======================================
An Explainable Multimodal AI-Based Digital Brain Twin
for Longitudinal Prediction of Alzheimer's Disease Progression.

Endpoints:
    GET  /                        — Project info
    GET  /health                  — Model load status
    POST /predict/clinical        — CDR prediction (Clinical GRU)
    POST /predict/bimodal         — CDR prediction (Clinical + MRI)
    POST /brain-twin/extract      — Z_t trajectory extraction
    GET  /experiments/            — List all experiment results
    GET  /experiments/{id}        — Specific experiment details
    POST /eeg/screen              — EEG binary disease screener

Interactive docs: http://localhost:8000/docs
ReDoc:            http://localhost:8000/redoc
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cerebro_x.api.services.model_loader import get_registry
from cerebro_x.api.routers import prediction, brain_twin, experiments, eeg, history, mri_upload, patient, explain
from cerebro_x.api.schemas.patient import HealthResponse
import cerebro_x.api.database as _database  # use module ref so test patches apply

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cerebro_x.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    logger.info("Initializing Database...")
    _database.Base.metadata.create_all(bind=_database.engine)
    logger.info("Database initialized.")
    
    logger.info("Loading Cerebro X models...")
    registry = get_registry()
    registry.load_all()
    logger.info("All models loaded. Status: %s", registry.status)
    yield
    logger.info("Cerebro X API shutting down.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cerebro X Research API",
    description=__doc__,
    version="1.0.0",
    contact={
        "name": "Cerebro X Research Team",
        "url": "https://github.com/cerebro-x",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# Allow all origins for local research use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(prediction.router)
app.include_router(brain_twin.router)
app.include_router(experiments.router)
app.include_router(eeg.router)
app.include_router(history.router)
app.include_router(mri_upload.router)
app.include_router(patient.router)
app.include_router(explain.router)


# ─── Root & Health ───────────────────────────────────────────────────────────

@app.get("/", tags=["Info"], summary="Project info")
def root():
    return {
        "project": "Cerebro X",
        "title": (
            "An Explainable Multimodal AI-Based Digital Brain Twin "
            "for Longitudinal Prediction of Alzheimer's Disease Progression"
        ),
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "dataset": "OASIS-2 Longitudinal (150 subjects) + OpenNeuro ds004504 (87 EEG subjects)",
        "disclaimer": (
            "This is a research prototype. All models are trained on publicly available "
            "datasets. Results are NOT validated for clinical use."
        ),
        "endpoints": {
            "predict_clinical":  "POST /predict/clinical",
            "predict_bimodal":   "POST /predict/bimodal",
            "brain_twin":        "POST /brain-twin/extract",
            "experiments":       "GET  /experiments/",
            "eeg_screen":        "POST /eeg/screen",
            "health":            "GET  /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"], summary="Health check")
def health():
    registry = get_registry()
    all_ok = all(registry.status.values()) if registry.status else False
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        models_loaded=registry.status,
    )
