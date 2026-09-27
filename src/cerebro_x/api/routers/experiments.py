"""
Router: Experiment results endpoints.
Lists all completed experiments and their metrics.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from cerebro_x.api.schemas.patient import ExperimentSummary
from cerebro_x.api.services.model_loader import get_registry

router = APIRouter(prefix="/experiments", tags=["Experiments"])

PHASE_DESCRIPTIONS = {
    "EXP-BASELINE-001":      (1, "Traditional ML baselines (Logistic Regression, Random Forest) on OASIS-2 clinical data."),
    "EXP-LONGITUDINAL-001":  (2, "Longitudinal GRU (TemporalCerebroNet) — clinical visit history sequence model."),
    "EXP-MRI-SCALAR-001":    (3, "MRI scalar branch — standalone encoder for nWBV, eTIV, ASF features."),
    "EXP-FUSION-BIMODAL-001":(5, "Bimodal fusion ablation — Clinical Only vs MRI Only vs Clinical+MRI."),
    "EXP-BRAIN-TWIN-001":    (6, "Digital Brain Twin Z_t extraction — per-patient latent trajectory."),
    "EXP-PROGRESSION-001":   (7, "Longitudinal progression analysis — per-class metrics and CDR trajectory plots."),
    "EXP-EXPLAIN-001":       (8, "SHAP temporal explainability for clinical GRU."),
    "EXP-EEG-STANDALONE-001":(4, "EEG standalone encoder on OpenNeuro ds004504 (87 subjects). Separate cohort from OASIS-2."),
}


@router.get(
    "/",
    response_model=list[ExperimentSummary],
    summary="List all experiments",
    description="Returns a list of all completed experiments with their metrics and artifact paths.",
)
def list_experiments(registry=Depends(get_registry)):
    results = []
    for exp_id, metrics in registry.metrics.items():
        phase, description = PHASE_DESCRIPTIONS.get(exp_id, (0, "Unknown experiment."))
        results.append(ExperimentSummary(
            experiment_id=exp_id,
            phase=phase,
            description=description,
            metrics=metrics,
            artifact_path=str(Path("artifacts") / exp_id),
        ))
    return sorted(results, key=lambda x: x.phase)


@router.get(
    "/{experiment_id}",
    response_model=ExperimentSummary,
    summary="Get experiment details",
    description="Returns detailed metrics for a specific experiment.",
)
def get_experiment(experiment_id: str, registry=Depends(get_registry)):
    if experiment_id not in registry.metrics:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found. "
                            f"Available: {list(registry.metrics.keys())}")
    phase, description = PHASE_DESCRIPTIONS.get(experiment_id, (0, "Unknown experiment."))
    return ExperimentSummary(
        experiment_id=experiment_id,
        phase=phase,
        description=description,
        metrics=registry.metrics[experiment_id],
        artifact_path=str(Path("artifacts") / experiment_id),
    )
