"""
Model loader service — loads all trained Cerebro X models at startup.
Implements lazy singleton loading to avoid re-loading on every request.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("cerebro_x.api.model_loader")

ARTIFACTS_DIR = Path("artifacts")


class ModelRegistry:
    """Singleton registry that holds all loaded model artifacts."""

    _instance: "ModelRegistry | None" = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance.status = {}
            cls._instance.models = {}
            cls._instance.scalers = {}
            cls._instance.metrics = {}
        return cls._instance

    def load_all(self) -> None:
        """Load all available trained models from artifacts/."""
        if self._loaded:
            return

        self.models: dict[str, Any] = {}
        self.scalers: dict[str, Any] = {}
        self.metrics: dict[str, dict] = {}
        self.status: dict[str, bool] = {}

        self._load_clinical_gru()
        self._load_shap_background()
        self._load_bimodal_fusion()
        self._load_eeg_encoder()
        self._load_cnn3d()
        self._load_experiment_metrics()

        self._loaded = True
        logger.info("Model registry loaded. Status: %s", self.status)

    def _load_clinical_gru(self) -> None:
        """Load TemporalCerebroNet (Phase 2 / Phase 7 GRU) + fitted ClinicalPreprocessor."""
        try:
            import sys
            sys.path.insert(0, str(Path("src")))
            from cerebro_x.models.deep.temporal import TemporalCerebroNet
            from cerebro_x.features.preprocessor import ClinicalPreprocessor

            checkpoint_path = ARTIFACTS_DIR / "EXP-LONGITUDINAL-001" / "temporal_gru_cpu.pt"
            artifact_dir = ARTIFACTS_DIR / "EXP-LONGITUDINAL-001"

            if not checkpoint_path.exists():
                logger.warning("Clinical GRU checkpoint not found: %s", checkpoint_path)
                self.status["clinical_gru"] = False
                return

            model = TemporalCerebroNet(
                input_dim=19, gru_hidden_dim=64, gru_num_layers=1,
                num_classes=4, dropout=0.3
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            model.eval()
            self.models["clinical_gru"] = model

            # Load the fitted preprocessor (imputer + scaler from training)
            try:
                preprocessor = ClinicalPreprocessor.load(artifact_dir)
                self.scalers["clinical_preprocessor"] = preprocessor
                logger.info("ClinicalPreprocessor loaded from %s", artifact_dir)
            except FileNotFoundError:
                logger.warning(
                    "clinical_preprocessor.pkl not found in %s. "
                    "Inference will use fallback preprocessor. "
                    "Re-run scripts/train_longitudinal.py to generate it.",
                    artifact_dir,
                )
                self.scalers["clinical_preprocessor"] = None

            self.status["clinical_gru"] = True
            logger.info("Clinical GRU loaded from %s", checkpoint_path)

        except Exception as e:
            logger.error("Failed to load Clinical GRU: %s", e)
            self.status["clinical_gru"] = False


    def _load_shap_background(self) -> None:
        """
        Load the SHAP background tensor and lengths from the pre-built artifact.

        Artifact: artifacts/EXP-LONGITUDINAL-001/shap_background.pt
        Build:    python scripts/build_shap_background.py

        The background size can be capped via env var SHAP_BACKGROUND_SIZE (default 50).
        """
        try:
            bg_path = ARTIFACTS_DIR / "EXP-LONGITUDINAL-001" / "shap_background.pt"
            meta_path = ARTIFACTS_DIR / "EXP-LONGITUDINAL-001" / "shap_background_meta.json"

            if not bg_path.exists():
                logger.warning(
                    "SHAP background artifact not found at %s. "
                    "Run: python scripts/build_shap_background.py",
                    bg_path,
                )
                self.status["shap_background"] = False
                return

            state = torch.load(bg_path, map_location="cpu", weights_only=True)
            bg_tensor: torch.Tensor = state["background"]   # (N, seq_len, 19)
            bg_lengths: torch.Tensor = state["lengths"]     # (N,)

            # Cap to configurable background size
            max_bg = int(os.getenv("SHAP_BACKGROUND_SIZE", "50"))
            n = min(max_bg, bg_tensor.size(0))
            self.models["shap_background"] = bg_tensor[:n]    # (n, seq_len, 19)
            self.models["shap_bg_lengths"] = bg_lengths[:n]   # (n,)

            # Load provenance metadata
            if meta_path.exists():
                with open(meta_path) as f:
                    self.metrics["shap_background_meta"] = json.load(f)

            self.status["shap_background"] = True
            logger.info(
                "SHAP background loaded: %d samples (capped from %d), shape %s",
                n, bg_tensor.size(0), tuple(self.models["shap_background"].shape),
            )

        except Exception as e:
            logger.error("Failed to load SHAP background: %s", e)
            self.status["shap_background"] = False

    def _load_bimodal_fusion(self) -> None:
        """Load BimodalCerebroNet (Phase 5)."""
        try:
            import sys
            sys.path.insert(0, str(Path("src")))
            from cerebro_x.models.deep.bimodal_fusion import BimodalCerebroNet

            checkpoint_path = ARTIFACTS_DIR / "EXP-FUSION-BIMODAL-001" / "clinical_plus_mri.pt"
            if not checkpoint_path.exists():
                logger.warning("Bimodal fusion checkpoint not found: %s", checkpoint_path)
                self.status["bimodal_fusion"] = False
                return

            model = BimodalCerebroNet(
                clinical_input_dim=19, mri_input_dim=4,
                clinical_hidden_dim=64, mri_embed_dim=32,
                num_classes=4, dropout=0.3
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            model.eval()
            self.models["bimodal_fusion"] = model
            self.status["bimodal_fusion"] = True
            logger.info("Bimodal fusion loaded from %s", checkpoint_path)

        except Exception as e:
            logger.error("Failed to load Bimodal Fusion: %s", e)
            self.status["bimodal_fusion"] = False

    def _load_eeg_encoder(self) -> None:
        """Load EEG standalone encoder (Phase 4)."""
        try:
            import sys, joblib
            sys.path.insert(0, str(Path("src")))

            artifact_dir = ARTIFACTS_DIR / "EXP-EEG-STANDALONE-001"
            binary_path = artifact_dir / "binary_encoder.pt"
            scaler_path = artifact_dir / "binary_scaler.pkl"
            imputer_path = artifact_dir / "binary_imputer.pkl"

            if not binary_path.exists():
                logger.warning("EEG encoder checkpoint not found: %s", binary_path)
                self.status["eeg_encoder"] = False
                return

            # We need to know input dim from the scaler
            scaler = joblib.load(scaler_path)
            imputer = joblib.load(imputer_path)
            n_features = scaler.n_features_in_

            # Recreate model architecture
            import torch.nn as nn
            class EEGSpectralEncoder(nn.Module):
                def __init__(self, input_dim, embed_dim=64, num_classes=2):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
                        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
                        nn.Linear(64, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(),
                    )
                    self.head = nn.Linear(embed_dim, num_classes)
                def forward(self, x): return self.head(self.encoder(x))

            model = EEGSpectralEncoder(n_features, embed_dim=64, num_classes=2)
            model.load_state_dict(torch.load(binary_path, map_location="cpu", weights_only=True))
            model.eval()

            self.models["eeg_encoder"] = model
            self.scalers["eeg_scaler"] = scaler
            self.scalers["eeg_imputer"] = imputer
            self.status["eeg_encoder"] = True
            logger.info("EEG encoder loaded. Input dim: %d", n_features)

        except Exception as e:
            logger.error("Failed to load EEG encoder: %s", e)
            self.status["eeg_encoder"] = False

    def _load_cnn3d(self) -> None:
        """
        Load MRICerebroNet for real MRI processing and Grad-CAM.

        MRICerebroNet = Lightweight3DCNN backbone + 4-class CDR classifier head.
        This is the correct model for Grad-CAM (outputs logits, not embeddings).

        Expects checkpoint at: artifacts/EXP-MRI-CNN3D-001/cnn3d_cpu.pt
        Config at:             artifacts/EXP-MRI-CNN3D-001/cnn3d_config.json

        If not found, status["cnn3d"] = False — no fake model is created.
        Callers must check registry.status["cnn3d"] before using the model.
        """
        try:
            import sys
            sys.path.insert(0, str(Path("src")))
            from cerebro_x.models.deep.cnn3d import MRICerebroNet

            checkpoint_path = ARTIFACTS_DIR / "EXP-MRI-CNN3D-001" / "cnn3d_cpu.pt"
            config_path     = ARTIFACTS_DIR / "EXP-MRI-CNN3D-001" / "cnn3d_config.json"

            if not checkpoint_path.exists():
                logger.warning(
                    "CNN3D checkpoint not found at %s. "
                    "MRI 3D-CNN and Grad-CAM will be unavailable. "
                    "To train: run python scripts/train_cnn3d.py",
                    checkpoint_path,
                )
                self.status["cnn3d"] = False
                return

            # Read architecture config if available
            embedding_dim = 64
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                embedding_dim = cfg.get("embedding_dim", 64)

            model = MRICerebroNet(
                in_channels=1,
                embedding_dim=embedding_dim,
                num_classes=4,
            )
            model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            )
            model.eval()
            self.models["cnn3d"] = model
            self.status["cnn3d"] = True
            logger.info("CNN3D (MRICerebroNet) loaded from %s", checkpoint_path)

        except Exception as e:
            logger.error("Failed to load CNN3D: %s", e)
            self.status["cnn3d"] = False

    def _load_experiment_metrics(self) -> None:
        """Load metrics.json from all experiment directories."""
        for exp_dir in sorted(ARTIFACTS_DIR.glob("EXP-*")):
            metrics_file = exp_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        self.metrics[exp_dir.name] = json.load(f)
                except Exception as e:
                    logger.warning("Could not load metrics from %s: %s", exp_dir.name, e)

        logger.info("Loaded metrics for experiments: %s", list(self.metrics.keys()))


# Global singleton
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """FastAPI dependency — returns the loaded model registry."""
    return _registry
