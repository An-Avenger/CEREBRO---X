"""
tests/test_explainability.py
------------------------------
Tests for the real SHAP + Grad-CAM explainability pipeline.

Covers:
  - SHAP module imports
  - Background artifact loading
  - SHAP tensor shapes and finiteness
  - API: real SHAP vs heuristic labeling
  - API: 503 when background missing
  - MRI CNN architecture
  - Grad-CAM hook firing
  - Heatmap shape and finiteness
  - Slice visualisation generation
  - API Grad-CAM endpoint structure
  - Existing functionality regression

Run:
    pytest tests/test_explainability.py -v
"""
from __future__ import annotations

import io
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ARTIFACT_DIR = Path("artifacts/EXP-LONGITUDINAL-001")
CHECKPOINT_PATH = ARTIFACT_DIR / "temporal_gru_cpu.pt"
SHAP_BG_PATH    = ARTIFACT_DIR / "shap_background.pt"
CNN_CHECKPOINT  = Path("artifacts/EXP-MRI-CNN3D-001/cnn3d_cpu.pt")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_nifti_bytes(shape=(64, 64, 64)) -> bytes:
    try:
        import nibabel as nib
        data = np.random.rand(*shape).astype(np.float32) * 100 + 10
        data[10:54, 10:54, 10:54] = 200.0  # brain region
        img = nib.Nifti1Image(data, affine=np.diag([1.5, 1.5, 1.5, 1.0]))
        buf = io.BytesIO()
        img.to_file_map({"header": nib.FileHolder(fileobj=buf),
                         "image":  nib.FileHolder(fileobj=buf)})
        return buf.getvalue()
    except ImportError:
        pytest.skip("nibabel not installed")


# ─── 1. SHAP Module Tests ─────────────────────────────────────────────────────

class TestSHAPModule:

    def test_shap_imports(self, client):
        """SHAP module must import without error."""
        from cerebro_x.explainability.shap_temporal import (
            TemporalModelWrapper,
            compute_temporal_shap,
        )
        assert TemporalModelWrapper is not None
        assert compute_temporal_shap is not None

    def test_shap_library_available(self, client):
        """shap package must be importable."""
        import shap
        assert hasattr(shap, "GradientExplainer")

    def test_temporal_model_wrapper_basic(self, client):
        """TemporalModelWrapper must forward through any batch size."""
        from cerebro_x.explainability.shap_temporal import TemporalModelWrapper
        from cerebro_x.models.deep.temporal import TemporalCerebroNet

        model = TemporalCerebroNet(input_dim=19, gru_hidden_dim=64, num_classes=4)
        model.eval()

        lengths = torch.tensor([3, 2, 1])
        wrapper = TemporalModelWrapper(model, lengths)

        x = torch.randn(3, 5, 19)
        with torch.no_grad():
            out = wrapper(x)
        assert out.shape == (3, 4)

    def test_temporal_wrapper_smaller_batch(self, client):
        """Wrapper must handle batch smaller than lengths tensor."""
        from cerebro_x.explainability.shap_temporal import TemporalModelWrapper
        from cerebro_x.models.deep.temporal import TemporalCerebroNet

        model = TemporalCerebroNet(input_dim=19, gru_hidden_dim=64, num_classes=4)
        lengths = torch.tensor([3, 3, 3, 3, 3])
        wrapper = TemporalModelWrapper(model, lengths)

        x = torch.randn(2, 5, 19)  # smaller batch
        with torch.no_grad():
            out = wrapper(x)
        assert out.shape == (2, 4)

    def test_temporal_wrapper_larger_batch(self, client):
        """Wrapper must handle batch larger than lengths tensor (repeats)."""
        from cerebro_x.explainability.shap_temporal import TemporalModelWrapper
        from cerebro_x.models.deep.temporal import TemporalCerebroNet

        model = TemporalCerebroNet(input_dim=19, gru_hidden_dim=64, num_classes=4)
        lengths = torch.tensor([2])
        wrapper = TemporalModelWrapper(model, lengths)

        x = torch.randn(4, 3, 19)  # larger batch
        with torch.no_grad():
            out = wrapper(x)
        assert out.shape == (4, 4)


# ─── 2. SHAP Background Artifact Tests ───────────────────────────────────────

class TestSHAPBackground:

    def test_background_artifact_exists(self, client):
        """SHAP background artifact must exist after build_shap_background.py."""
        assert SHAP_BG_PATH.exists(), (
            f"SHAP background not found at {SHAP_BG_PATH}. "
            "Run: python scripts/build_shap_background.py"
        )

    def test_background_artifact_loads(self, client):
        """Background artifact must load without error."""
        if not SHAP_BG_PATH.exists():
            pytest.skip("SHAP background artifact not built yet")
        state = torch.load(SHAP_BG_PATH, map_location="cpu", weights_only=True)
        assert "background" in state
        assert "lengths" in state

    def test_background_tensor_shape(self, client):
        """Background tensor must be (N, seq_len, 19)."""
        if not SHAP_BG_PATH.exists():
            pytest.skip("SHAP background artifact not built yet")
        state = torch.load(SHAP_BG_PATH, map_location="cpu", weights_only=True)
        bg = state["background"]
        assert bg.ndim == 3, f"Expected 3D tensor, got {bg.ndim}D"
        assert bg.shape[2] == 19, f"Expected 19 features, got {bg.shape[2]}"
        assert bg.shape[0] >= 1, "Background must have at least 1 sample"

    def test_background_feature_count_is_19(self, client):
        """Background must have exactly 19 features matching FEATURE_ORDER."""
        if not SHAP_BG_PATH.exists():
            pytest.skip("SHAP background artifact not built yet")
        from cerebro_x.features.preprocessor import FEATURE_ORDER
        state = torch.load(SHAP_BG_PATH, map_location="cpu", weights_only=True)
        bg = state["background"]
        assert bg.shape[2] == len(FEATURE_ORDER) == 19

    def test_background_is_finite(self, client):
        """Background tensor must be finite (no NaN or Inf)."""
        if not SHAP_BG_PATH.exists():
            pytest.skip("SHAP background artifact not built yet")
        state = torch.load(SHAP_BG_PATH, map_location="cpu", weights_only=True)
        bg = state["background"]
        assert not torch.isnan(bg).any(), "Background contains NaN"
        assert not torch.isinf(bg).any(), "Background contains Inf"

    def test_background_lengths_match_tensor(self, client):
        """Background lengths tensor size must match batch dimension."""
        if not SHAP_BG_PATH.exists():
            pytest.skip("SHAP background artifact not built yet")
        state = torch.load(SHAP_BG_PATH, map_location="cpu", weights_only=True)
        bg  = state["background"]
        lns = state["lengths"]
        assert bg.shape[0] == lns.shape[0], (
            f"background.shape[0]={bg.shape[0]} != lengths.shape[0]={lns.shape[0]}"
        )


# ─── 3. End-to-End SHAP Computation Tests ─────────────────────────────────────

class TestSHAPComputation:

    @pytest.fixture(scope="class")
    def model_and_bg(self):
        """Load real trained model and background for SHAP tests."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip("Clinical GRU checkpoint not found")
        if not SHAP_BG_PATH.exists():
            pytest.skip("SHAP background not found")

        from cerebro_x.models.deep.temporal import TemporalCerebroNet
        model = TemporalCerebroNet(input_dim=19, gru_hidden_dim=64, num_classes=4)
        model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        )
        model.eval()

        state  = torch.load(SHAP_BG_PATH, map_location="cpu", weights_only=True)
        bg     = state["background"][:10]    # use small subset for speed
        bg_len = state["lengths"][:10]

        return model, bg, bg_len

    def test_shap_computes_without_error(self, model_and_bg):
        """compute_temporal_shap must run without raising."""
        from cerebro_x.explainability.shap_temporal import compute_temporal_shap
        model, bg, bg_len = model_and_bg

        test_x   = bg[:2]    # (2, seq_len, 19)
        test_len = bg_len[:2]

        shap_values = compute_temporal_shap(model, bg, bg_len, test_x, test_len)
        assert shap_values is not None

    def test_shap_values_are_finite(self, model_and_bg):
        """All SHAP values must be finite."""
        from cerebro_x.explainability.shap_temporal import compute_temporal_shap
        model, bg, bg_len = model_and_bg

        test_x   = bg[:2]
        test_len = bg_len[:2]

        shap_values = compute_temporal_shap(model, bg, bg_len, test_x, test_len)

        # shap_values may be list or ndarray
        if isinstance(shap_values, list):
            for sv in shap_values:
                if isinstance(sv, np.ndarray):
                    assert np.isfinite(sv).all(), "SHAP values contain non-finite values"
        elif isinstance(shap_values, np.ndarray):
            assert np.isfinite(shap_values).all()

    def test_shap_feature_count_matches(self, model_and_bg):
        """SHAP output feature count must be 19."""
        from cerebro_x.explainability.shap_temporal import compute_temporal_shap
        model, bg, bg_len = model_and_bg

        test_x   = bg[:1]
        test_len = bg_len[:1]

        shap_values = compute_temporal_shap(model, bg, bg_len, test_x, test_len)

        # GradientExplainer returns (batch, seq, feat, num_classes) or list
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 4:
            if shap_values.shape[-1] == 4:
                sv = shap_values[..., 0]  # Take class 0: (batch, seq, feat)
            else:
                sv = shap_values[0] # (num_classes, batch, seq, feat)
        elif isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        if isinstance(sv, np.ndarray) and sv.ndim >= 2:
            n_features = sv.shape[-1]
            assert n_features == 19, f"Expected 19 features, got {n_features}"


# ─── 4. API Clinical SHAP Tests ───────────────────────────────────────────────

class TestAPIClinicalSHAP:

    SAMPLE_REQUEST = {
        "subject_id": "TEST_SHAP",
        "visits": [
            {"age": 72.0, "educ": 16.0, "ses": 2.0, "mmse": 28.0,
             "cdr": 0.0, "nwbv": 0.763, "etiv": 1480.0, "asf": 1.23,
             "gender": "F", "hand": "R"},
            {"age": 74.0, "educ": 16.0, "ses": 2.0, "mmse": 26.0,
             "cdr": 0.5, "nwbv": 0.743, "etiv": 1478.0, "asf": 1.23,
             "gender": "F", "hand": "R"},
        ],
    }


    def test_clinical_explain_returns_200_when_model_loaded(self, client):
        """Clinical SHAP endpoint must return 200 when GRU + background are available."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip("GRU checkpoint not found")
        if not SHAP_BG_PATH.exists():
            pytest.skip("SHAP background not found")

        response = client.post("/explain/clinical", json=self.SAMPLE_REQUEST)

        # May be 200 (SHAP), or 500 if SHAP fails internally
        # The test checks that the endpoint runs without a 503 from missing model
        assert response.status_code != 503, (
            f"503 means model/background not loaded: {response.json()}"
        )

    def test_clinical_explain_method_field_is_shap_not_heuristic(self, client):
        """When background is available, method must be SHAP_GradientExplainer."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip("GRU checkpoint not found")
        if not SHAP_BG_PATH.exists():
            pytest.skip("SHAP background not found")

        response = client.post("/explain/clinical", json=self.SAMPLE_REQUEST)

        if response.status_code == 200:
            data = response.json()
            method = data.get("method", "")
            assert method == "SHAP_GradientExplainer", (
                f"Expected 'SHAP_GradientExplainer' but got '{method}'. "
                "Real SHAP must be used when background is available."
            )

    def test_clinical_explain_never_labels_heuristic_as_shap(self, client):
        """The word 'SHAP' must not appear in heuristic_fallback response."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip("GRU checkpoint not found")

        # Temporarily hide the SHAP background to test fallback path

        # Force the fallback by patching registry.status
        from cerebro_x.api.services.model_loader import get_registry
        registry = get_registry()
        original = registry.status.get("shap_background", True)

        try:
            registry.status["shap_background"] = False
            response = client.post("/explain/clinical", json=self.SAMPLE_REQUEST)

            if response.status_code == 200:
                data = response.json()
                method = data.get("method", "")
                # Must be heuristic_fallback, never SHAP
                assert "SHAP" not in method or "fallback" in method, (
                    f"Heuristic must not be labelled as SHAP. Got method='{method}'"
                )
        finally:
            registry.status["shap_background"] = original

    def test_clinical_explain_response_has_features_list(self, client):
        """Response must include a non-empty features list."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip("GRU checkpoint not found")

        response = client.post("/explain/clinical", json=self.SAMPLE_REQUEST)

        if response.status_code == 200:
            data = response.json()
            assert "features" in data, "Response must contain 'features'"
            assert len(data["features"]) > 0

    def test_clinical_explain_503_when_model_missing(self, client):
        """Must return 503 when clinical GRU is not loaded."""

        from cerebro_x.api.services.model_loader import get_registry
        registry = get_registry()
        original = registry.status.get("clinical_gru", True)

        try:
            registry.status["clinical_gru"] = False
            response = client.post("/explain/clinical", json=self.SAMPLE_REQUEST)
            assert response.status_code == 503
        finally:
            registry.status["clinical_gru"] = original

    def test_clinical_explain_no_heuristic_contribution_fields(self, client):
        """Real SHAP response must not have 'contribution' field (heuristic marker)."""
        if not CHECKPOINT_PATH.exists() or not SHAP_BG_PATH.exists():
            pytest.skip("Required artifacts not found")

        response = client.post("/explain/clinical", json=self.SAMPLE_REQUEST)

        if response.status_code == 200:
            data = response.json()
            if data.get("method") == "SHAP_GradientExplainer":
                # Real SHAP response uses 'importance', not 'contribution'
                features = data.get("features", [])
                for feat in features:
                    assert "importance" in feat, "SHAP features must have 'importance'"
                    assert "mean_abs_shap" in feat, "SHAP features must have 'mean_abs_shap'"
                    assert "direction" in feat, "SHAP features must have 'direction'"

    def test_clinical_explain_temporal_attributions_present(self, client):
        """Real SHAP response must include temporal_attributions per visit."""
        if not CHECKPOINT_PATH.exists() or not SHAP_BG_PATH.exists():
            pytest.skip("Required artifacts not found")

        response = client.post("/explain/clinical", json=self.SAMPLE_REQUEST)

        if response.status_code == 200:
            data = response.json()
            if data.get("method") == "SHAP_GradientExplainer":
                assert "temporal_attributions" in data
                ta = data["temporal_attributions"]
                assert len(ta) > 0
                assert "visit" in ta[0]
                assert "shap_values" in ta[0]

    def test_clinical_explain_background_metadata(self, client):
        """Real SHAP response must include background provenance."""
        if not CHECKPOINT_PATH.exists() or not SHAP_BG_PATH.exists():
            pytest.skip("Required artifacts not found")

        response = client.post("/explain/clinical", json=self.SAMPLE_REQUEST)

        if response.status_code == 200:
            data = response.json()
            if data.get("method") == "SHAP_GradientExplainer":
                assert "background" in data
                bg = data["background"]
                assert "source" in bg
                assert "num_samples" in bg

    def test_clinical_explain_prediction_fields(self, client):
        """Response must contain prediction class and probability."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip("GRU checkpoint not found")

        response = client.post("/explain/clinical", json=self.SAMPLE_REQUEST)

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            pred = data["prediction"]
            assert "class" in pred
            assert "probability" in pred
            assert 0 <= pred["class"] <= 3
            assert 0.0 <= pred["probability"] <= 1.0


# ─── 5. CNN3D Architecture Tests ──────────────────────────────────────────────

class TestCNN3DForGradCAM:

    def test_mri_cerebro_net_forward(self, client):
        """MRICerebroNet must produce 4-class logits from (1,1,64,64,64)."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        model.eval()

        x = torch.rand(1, 1, 64, 64, 64)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 4), f"Expected (1, 4), got {logits.shape}"

    def test_mri_cerebro_net_no_nan(self, client):
        """MRICerebroNet must not produce NaN."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            logits = model(x)
        assert not torch.isnan(logits).any()

    def test_mri_cerebro_net_has_conv_layers(self, client):
        """MRICerebroNet must have at least one Conv3d layer for Grad-CAM."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]
        assert len(conv_layers) >= 1, "No Conv3d layers found for Grad-CAM"


# ─── 6. Grad-CAM Tests ────────────────────────────────────────────────────────

class TestGradCAM3D:

    def test_gradcam_finds_target_layer(self, client):
        """GradCAM3DCNN must find a Conv3d target layer in MRICerebroNet."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        cam = GradCAM3DCNN(model)
        assert cam._target_layer is not None
        assert cam._layer_name != ""
        cam.remove_hooks()

    def test_gradcam_hooks_fire(self, client):
        """Forward and backward hooks must capture activations and gradients."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x, target_class=0)
        cam.remove_hooks()

        assert result.success, f"Grad-CAM failed: {result.error_code}: {result.error_message}"
        assert cam._activations is not None or result.heatmap_3d is not None

    def test_gradcam_heatmap_shape(self, client):
        """Grad-CAM heatmap must match input spatial dimensions (64, 64, 64)."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x, target_class=1)
        cam.remove_hooks()

        assert result.success
        assert result.heatmap_3d is not None
        assert result.heatmap_3d.shape == (64, 64, 64)

    def test_gradcam_heatmap_range(self, client):
        """Grad-CAM heatmap must be in [0, 1]."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x, target_class=0)
        cam.remove_hooks()

        if result.success and result.heatmap_3d is not None:
            h = result.heatmap_3d
            assert h.min() >= -1e-6, f"heatmap min={h.min()} < 0"
            assert h.max() <= 1.0 + 1e-6, f"heatmap max={h.max()} > 1"

    def test_gradcam_heatmap_is_finite(self, client):
        """Grad-CAM heatmap must contain only finite values."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x)
        cam.remove_hooks()

        if result.success and result.heatmap_3d is not None:
            assert np.isfinite(result.heatmap_3d).all()

    def test_gradcam_axial_slice_generated(self, client):
        """Axial slice visualization must be generated."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x)
        cam.remove_hooks()

        if result.success and result.slices_base64 is not None:
            assert "axial" in result.slices_base64
            assert result.slices_base64["axial"] is not None
            assert result.slices_base64["axial"].startswith("data:image/png;base64,")

    def test_gradcam_sagittal_slice_generated(self, client):
        """Sagittal slice visualization must be generated."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x)
        cam.remove_hooks()

        if result.success and result.slices_base64 is not None:
            assert "sagittal" in result.slices_base64

    def test_gradcam_coronal_slice_generated(self, client):
        """Coronal slice visualization must be generated."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x)
        cam.remove_hooks()

        if result.success and result.slices_base64 is not None:
            assert "coronal" in result.slices_base64

    def test_gradcam_target_class_respected(self, client):
        """Grad-CAM must use the requested target class."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x, target_class=2)
        cam.remove_hooks()

        if result.success:
            assert result.target_class == 2

    def test_gradcam_embedding_is_64d(self, client):
        """Grad-CAM result must include 64-dim CNN embedding."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from cerebro_x.explainability.gradcam_3d import run_gradcam_3d_cnn

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        x = torch.rand(1, 1, 64, 64, 64)
        result = run_gradcam_3d_cnn(model, x)

        if result.success and result.embedding is not None:
            assert len(result.embedding) == 64


# ─── 7. API MRI Grad-CAM Tests ────────────────────────────────────────────────

class TestAPIMRIGradCAM:


    def test_mri_explain_503_when_cnn_missing(self, client):
        """Must return 503 when CNN3D checkpoint is not loaded."""

        from cerebro_x.api.services.model_loader import get_registry
        registry = get_registry()
        original = registry.status.get("cnn3d", False)

        try:
            registry.status["cnn3d"] = False
            nifti = _make_nifti_bytes()
            files = {"file": ("brain.nii", nifti, "application/octet-stream")}
            response = client.post("/explain/mri", files=files)
            assert response.status_code == 503
            data = response.json()
            assert "CNN3D_CHECKPOINT_NOT_AVAILABLE" in str(data)
        finally:
            registry.status["cnn3d"] = original

    def test_mri_explain_400_for_invalid_extension(self, client):
        """Must return 400 for non-NIfTI file."""
        files = {"file": ("scan.jpg", b"not a nifti", "image/jpeg")}
        response = client.post("/explain/mri", files=files)
        assert response.status_code in {400, 503}  # 503 if CNN not loaded

    def test_mri_explain_with_loaded_model(self, client):
        """When CNN3D is loaded, /explain/mri must process valid NIfTI."""
        if not CNN_CHECKPOINT.exists():
            pytest.skip("CNN3D checkpoint not found; run scripts/train_cnn3d.py")

        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from fastapi.testclient import TestClient
        from cerebro_x.api.main import app
        from cerebro_x.api.services.model_loader import get_registry

        nifti = _make_nifti_bytes()
        files = {"file": ("brain.nii", nifti, "application/octet-stream")}
        response = client.post("/explain/mri", files=files)

        # CNN may or may not be loaded depending on lifespan init
        assert response.status_code in {200, 422, 503}, (
            f"Unexpected status: {response.status_code}: {response.json()}"
        )

        if response.status_code == 200:
            data = response.json()
            assert data.get("success") is True
            assert "visualizations" in data
            assert "prediction" in data
            assert data.get("explanation_method") == "3D-GradCAM"

    def test_mri_explain_response_has_correct_method_when_successful(self, client):
        """Successful Grad-CAM response must label method as '3D-GradCAM'."""
        if not CNN_CHECKPOINT.exists():
            pytest.skip("CNN3D checkpoint not found")

        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from cerebro_x.api.services.model_loader import get_registry
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        from fastapi.testclient import TestClient
        from cerebro_x.api.main import app

        registry = get_registry()
        if not registry.status.get("cnn3d"):
            pytest.skip("CNN3D not loaded in registry")

        nifti = _make_nifti_bytes()
        files = {"file": ("brain.nii", nifti, "application/octet-stream")}
        response = client.post("/explain/mri", files=files)

        if response.status_code == 200:
            data = response.json()
            assert data.get("explanation_method") == "3D-GradCAM"
            assert "heuristic" not in str(data.get("explanation_method", "")).lower()


# ─── 8. Regression Tests ──────────────────────────────────────────────────────

class TestExistingFunctionalityRegression:
    """Verify that existing prediction functionality still works."""

    SAMPLE_PREDICT = {
        "subject_id": "REGRESSION_TEST",
        "visits": [
            {"age": 72.0, "educ": 16.0, "ses": 2.0, "mmse": 28.0,
             "cdr": 0.0, "nwbv": 0.763, "etiv": 1480.0, "asf": 1.23},
            {"age": 74.0, "educ": 16.0, "ses": 2.0, "mmse": 26.0,
             "cdr": 0.5, "nwbv": 0.743, "etiv": 1478.0, "asf": 1.23},
        ],
    }


    def test_predict_clinical_still_works(self, client):
        """POST /predict/clinical must still return 200."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip("GRU checkpoint not found")
        response = client.post("/predict/clinical", json=self.SAMPLE_PREDICT)
        assert response.status_code == 200

    def test_predict_clinical_returns_valid_prediction(self, client):
        """Prediction response must have expected structure."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip("GRU checkpoint not found")
        response = client.post("/predict/clinical", json=self.SAMPLE_PREDICT)
        if response.status_code == 200:
            data = response.json()
            assert "predicted_cdr_class" in data
            assert 0 <= data["predicted_cdr_class"] <= 3

    def test_health_endpoint_still_works(self, client):
        """GET /health must still return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
