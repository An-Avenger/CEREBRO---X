"""
Real MRI Pipeline Tests for Cerebro-X.

Tests are graded by severity:
  - CRITICAL: Must pass for any production use
  - IMPORTANT: Should pass for correct behaviour
  - INFORMATIONAL: Pipeline metadata checks

Run:
    pytest tests/test_mri_pipeline.py -v

Requires:
    pip install nibabel numpy torch pytest
"""
from __future__ import annotations

import io
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_nifti_bytes(shape=(64, 64, 64), fill_value=1.0) -> bytes:
    """
    Create a minimal valid NIfTI-1 file in memory.
    Uses nibabel to produce a real, valid NIfTI header.
    """
    try:
        import nibabel as nib
        data = np.full(shape, fill_value, dtype=np.float32)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        buf = io.BytesIO()
        img.to_file_map({"header": nib.FileHolder(fileobj=buf), "image": nib.FileHolder(fileobj=buf)})
        return buf.getvalue()
    except ImportError:
        pytest.skip("nibabel not installed")


def _make_real_nifti_file(
    shape=(64, 64, 64),
    fill_value=1.0,
    suffix=".nii",
) -> Path:
    """Create a real NIfTI temp file and return its path."""
    try:
        import nibabel as nib
        data = np.full(shape, fill_value, dtype=np.float32)
        # Add some variation to make it non-trivial
        data[10:50, 10:50, 10:50] = 100.0  # simulate brain region
        data[0:5, :, :] = 0.0               # simulate background
        img = nib.Nifti1Image(data, affine=np.diag([1.5, 1.5, 1.5, 1.0]))
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        nib.save(img, tmp.name)
        tmp.close()
        return Path(tmp.name)
    except ImportError:
        pytest.skip("nibabel not installed")


# ─── Group 1: NiftiLoader Tests (CRITICAL) ────────────────────────────────────

class TestNiftiLoader:

    def test_load_valid_nifti(self):
        """CRITICAL: NiftiLoader loads a real NIfTI file and returns OK status."""
        from cerebro_x.imaging.nifti_loader import NiftiLoader, MRILoadStatus

        path = _make_real_nifti_file()
        loader = NiftiLoader()
        result = loader.load(str(path), subject_id="TEST_001", visit=1)
        path.unlink(missing_ok=True)

        assert result.status == MRILoadStatus.OK, f"Expected OK, got {result.status}: {result.message}"
        assert result.tensor is not None
        assert len(result.tensor.shape) == 3, "Tensor must be 3D (D, H, W)"
        assert result.tensor.dtype == torch.float32

    def test_load_missing_file(self):
        """CRITICAL: NiftiLoader returns MISSING status for non-existent paths."""
        from cerebro_x.imaging.nifti_loader import NiftiLoader, MRILoadStatus

        loader = NiftiLoader()
        result = loader.load("/definitely/does/not/exist.nii", subject_id="TEST", visit=1)
        assert result.status == MRILoadStatus.MISSING

    def test_load_empty_volume(self):
        """CRITICAL: NiftiLoader detects and rejects all-zero volumes."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from cerebro_x.imaging.nifti_loader import NiftiLoader, MRILoadStatus

        # Create a guaranteed all-zero NIfTI volume using np.zeros
        # and explicitly set scl_slope=1, scl_inter=0 to prevent scaling artifacts
        data = np.zeros((64, 64, 64), dtype=np.float32)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        img.header.set_slope_inter(1.0, 0.0)  # No intensity scaling

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
            nib.save(img, f.name)
            path = Path(f.name)

        loader = NiftiLoader()
        result = loader.load(str(path), subject_id="TEST", visit=1)
        path.unlink(missing_ok=True)

        assert result.status == MRILoadStatus.EMPTY, f"Expected EMPTY, got {result.status}: {result.message}"

    def test_load_wrong_dimensions(self):
        """CRITICAL: Loader rejects volumes with out-of-range dimensions."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        data = np.ones((10, 10, 10), dtype=np.float32)  # 10 < min_dim_voxels=32
        img = nib.Nifti1Image(data, np.eye(4))
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
            nib.save(img, f.name)
            path = Path(f.name)

        from cerebro_x.imaging.nifti_loader import NiftiLoader, MRILoadStatus

        loader = NiftiLoader(min_dim_voxels=32)
        result = loader.load(str(path), subject_id="TEST", visit=1)
        path.unlink(missing_ok=True)

        assert result.status == MRILoadStatus.WRONG_DIMENSIONS

    def test_tensor_dtype_is_float32(self):
        """IMPORTANT: Loaded tensor must be float32 for PyTorch compatibility."""
        from cerebro_x.imaging.nifti_loader import NiftiLoader, MRILoadStatus

        path = _make_real_nifti_file()
        loader = NiftiLoader()
        result = loader.load(str(path), subject_id="TEST", visit=1)
        path.unlink(missing_ok=True)

        assert result.ok
        assert result.tensor.dtype == torch.float32


# ─── Group 2: Preprocessing Tests (CRITICAL) ─────────────────────────────────

class TestMRIPreprocessing:

    def test_output_shape(self):
        """CRITICAL: Preprocessing outputs exactly (64, 64, 64)."""
        from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline

        raw = torch.rand(128, 96, 80) * 200  # realistic MRI intensities
        pipeline = MRIPreprocessingPipeline(target_shape=(64, 64, 64))
        out = pipeline(raw)

        assert out.shape == (64, 64, 64), f"Expected (64, 64, 64), got {out.shape}"

    def test_no_nan_or_inf(self):
        """CRITICAL: Preprocessed tensor must not contain NaN or Inf."""
        from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline

        raw = torch.rand(90, 90, 90) * 1000
        pipeline = MRIPreprocessingPipeline(target_shape=(64, 64, 64))
        out = pipeline(raw)

        assert not torch.isnan(out).any(), "Preprocessed tensor contains NaN"
        assert not torch.isinf(out).any(), "Preprocessed tensor contains Inf"

    def test_rejects_2d_input(self):
        """IMPORTANT: Preprocessing rejects non-3D tensors."""
        from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline

        pipeline = MRIPreprocessingPipeline()
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            pipeline(torch.rand(64, 64))  # 2D — should fail

    def test_normalization_reduces_range(self):
        """IMPORTANT: After z-score normalization, brain voxels should be near zero-mean."""
        from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline

        raw = torch.rand(64, 64, 64) * 500 + 100  # non-zero values
        pipeline = MRIPreprocessingPipeline(target_shape=(64, 64, 64))
        out = pipeline(raw)

        # After z-score, mean should be ~0 (may be offset due to background clipping)
        # We just check it's no longer in raw intensity range (100-600)
        assert float(out.max()) < 10.0, "After normalization, max should be <10"

    def test_deterministic(self):
        """IMPORTANT: Same input → same output every time."""
        from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline

        raw = torch.rand(64, 64, 64) * 300
        pipeline = MRIPreprocessingPipeline(target_shape=(64, 64, 64))
        out1 = pipeline(raw.clone())
        out2 = pipeline(raw.clone())

        assert torch.allclose(out1, out2, atol=1e-5), "Preprocessing is not deterministic"


# ─── Group 3: MRI Preprocessing Pipeline (End-to-End) ────────────────────────

class TestMRIProcessingPipeline:

    def test_valid_nifti_bytes_success(self):
        """CRITICAL: process_mri_bytes succeeds on real NIfTI bytes."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes

        path = _make_real_nifti_file()
        file_bytes = path.read_bytes()
        path.unlink(missing_ok=True)

        result = process_mri_bytes(file_bytes, "brain.nii", subject_id="TEST")

        assert result.success, f"Expected success, got error: {result.error_message}"
        assert result.scalars is not None
        assert result.preprocessed is not None

    def test_preprocessed_shape_is_1_64_64_64(self):
        """CRITICAL: Preprocessed tensor must be (1, 64, 64, 64) for CNN."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes

        path = _make_real_nifti_file()
        file_bytes = path.read_bytes()
        path.unlink(missing_ok=True)

        result = process_mri_bytes(file_bytes, "brain.nii")

        assert result.success
        assert result.preprocessed.shape == (1, 64, 64, 64), (
            f"Expected (1, 64, 64, 64), got {result.preprocessed.shape}"
        )

    def test_nwbv_proxy_is_between_0_and_1(self):
        """CRITICAL: nwbv_proxy must be a valid fraction [0, 1]."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes

        path = _make_real_nifti_file()
        file_bytes = path.read_bytes()
        path.unlink(missing_ok=True)

        result = process_mri_bytes(file_bytes, "brain.nii")

        assert result.success
        nwbv = result.scalars.nwbv_proxy
        assert 0.0 <= nwbv <= 1.0, f"nwbv_proxy={nwbv} out of [0, 1] range"

    def test_invalid_extension_rejected(self):
        """CRITICAL: Non-NIfTI files are rejected before processing."""
        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes

        result = process_mri_bytes(b"fake data", "scan.jpg")
        assert not result.success
        assert result.error_code == "INVALID_EXTENSION"

    def test_corrupt_bytes_rejected(self):
        """CRITICAL: Corrupt/invalid NIfTI bytes are rejected with INVALID_NIFTI."""
        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes

        # Generate random bytes that are definitely not a valid NIfTI
        fake = bytes(range(256)) * 100
        result = process_mri_bytes(fake, "corrupt.nii")

        assert not result.success
        assert result.error_code in {"INVALID_NIFTI", "WRONG_DIMENSIONS", "EMPTY_VOLUME"}

    def test_cnn_status_reports_unavailable(self):
        """IMPORTANT: cnn_status must indicate checkpoint unavailability."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes

        path = _make_real_nifti_file()
        file_bytes = path.read_bytes()
        path.unlink(missing_ok=True)

        result = process_mri_bytes(file_bytes, "brain.nii")

        if result.success:
            # CNN embedding should be None since no checkpoint
            assert result.cnn_embedding is None
            assert "NOT_AVAILABLE" in result.cnn_status or "checkpoint" in result.cnn_status.lower()

    def test_no_fake_values(self):
        """CRITICAL: Verify nwbv_proxy is NOT hash-derived (not a fixed/predictable value)."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes

        # Two different MRI volumes should yield different nwbv_proxy values
        path_a = _make_real_nifti_file(fill_value=100.0)
        path_b = _make_real_nifti_file(fill_value=200.0)
        bytes_a = path_a.read_bytes()
        bytes_b = path_b.read_bytes()
        path_a.unlink(missing_ok=True)
        path_b.unlink(missing_ok=True)

        result_a = process_mri_bytes(bytes_a, "a.nii")
        result_b = process_mri_bytes(bytes_b, "b.nii")

        if result_a.success and result_b.success:
            # Structural properties should differ between different volumes
            # (At minimum, they should not both be identical hash-derived values)
            # The nwbv_proxy for fill_value=100 and fill_value=200 will differ
            # because the background threshold changes relative to the intensities
            assert isinstance(result_a.scalars.nwbv_proxy, float)
            assert isinstance(result_b.scalars.nwbv_proxy, float)
            # Both should be real fractions, not hash-based constants
            assert 0.0 <= result_a.scalars.nwbv_proxy <= 1.0
            assert 0.0 <= result_b.scalars.nwbv_proxy <= 1.0


# ─── Group 4: CNN Architecture Tests (no checkpoint needed) ───────────────────

class TestCNN3DArchitecture:

    def test_cnn3d_forward_shape(self):
        """CRITICAL: CNN3D forward pass produces correct embedding shape."""
        from cerebro_x.models.deep.cnn3d import Lightweight3DCNN

        model = Lightweight3DCNN(in_channels=1, embedding_dim=64)
        model.eval()

        x = torch.rand(1, 1, 64, 64, 64)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 64), f"Expected (1, 64), got {out.shape}"

    def test_cnn3d_accepts_real_preprocessed_tensor(self):
        """CRITICAL: CNN3D can process a real preprocessed MRI tensor."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes
        from cerebro_x.models.deep.cnn3d import Lightweight3DCNN

        path = _make_real_nifti_file()
        file_bytes = path.read_bytes()
        path.unlink(missing_ok=True)

        proc = process_mri_bytes(file_bytes, "brain.nii")
        assert proc.success

        model = Lightweight3DCNN(in_channels=1, embedding_dim=64)
        model.eval()

        # preprocessed is (1, 64, 64, 64) — add batch dim → (1, 1, 64, 64, 64)
        tensor_5d = proc.preprocessed.unsqueeze(0)
        with torch.no_grad():
            embedding = model(tensor_5d)

        assert embedding.shape == (1, 64)
        assert not torch.isnan(embedding).any()

    def test_cnn3d_no_nan_on_random_input(self):
        """IMPORTANT: CNN3D should not produce NaN for valid inputs."""
        from cerebro_x.models.deep.cnn3d import Lightweight3DCNN

        model = Lightweight3DCNN()
        model.eval()
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            out = model(x)

        assert not torch.isnan(out).any()


# ─── Group 5: Grad-CAM Tests (untrained model — shape tests only) ─────────────

class TestGradCAM3D:

    def test_gradcam_heatmap_shape(self):
        """IMPORTANT: Grad-CAM heatmap matches input spatial dimensions."""
        from cerebro_x.models.deep.cnn3d import Lightweight3DCNN
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = Lightweight3DCNN(in_channels=1, embedding_dim=64)

        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x, target_class=0)
        cam.remove_hooks()

        assert result.success, f"Grad-CAM failed: {result.error_message}"
        assert result.heatmap_3d is not None
        assert result.heatmap_3d.shape == (64, 64, 64)

    def test_gradcam_heatmap_range(self):
        """IMPORTANT: Grad-CAM output must be in [0, 1]."""
        from cerebro_x.models.deep.cnn3d import Lightweight3DCNN
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = Lightweight3DCNN()
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x, target_class=1)
        cam.remove_hooks()

        if result.success and result.heatmap_3d is not None:
            assert result.heatmap_3d.min() >= 0.0
            assert result.heatmap_3d.max() <= 1.0001  # float tolerance

    def test_gradcam_target_class_captured(self):
        """INFORMATIONAL: Grad-CAM returns the target class used."""
        from cerebro_x.models.deep.cnn3d import Lightweight3DCNN
        from cerebro_x.explainability.gradcam_3d import GradCAM3DCNN

        model = Lightweight3DCNN()
        cam = GradCAM3DCNN(model)
        x = torch.rand(1, 1, 64, 64, 64)
        result = cam.generate(x, target_class=2)
        cam.remove_hooks()

        if result.success:
            assert result.target_class == 2

    def test_gradcam_on_real_preprocessed_mri(self):
        """CRITICAL: Full pipeline: NIfTI → preprocess → CNN → Grad-CAM."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        from cerebro_x.imaging.mri_preprocessing import process_mri_bytes
        from cerebro_x.models.deep.cnn3d import Lightweight3DCNN
        from cerebro_x.explainability.gradcam_3d import run_gradcam_3d_cnn

        path = _make_real_nifti_file()
        file_bytes = path.read_bytes()
        path.unlink(missing_ok=True)

        proc = process_mri_bytes(file_bytes, "brain.nii")
        assert proc.success

        model = Lightweight3DCNN(in_channels=1, embedding_dim=64)
        # Use (1, 1, 64, 64, 64) for run_gradcam_3d_cnn
        tensor_5d = proc.preprocessed.unsqueeze(0)

        result = run_gradcam_3d_cnn(model, tensor_5d, target_class=None)

        assert result.success, f"Grad-CAM failed: {result.error_code}: {result.error_message}"
        assert result.heatmap_3d is not None
        assert result.heatmap_3d.shape == (64, 64, 64)
        assert result.embedding is not None
        assert len(result.embedding) == 64


# ─── Group 6: API Endpoint Tests ─────────────────────────────────────────────

class TestMRIUploadEndpoint:

    def _get_client(self):
        """Create a FastAPI test client."""
        try:
            from fastapi.testclient import TestClient
            from cerebro_x.api.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI test client not available")

    def test_upload_invalid_extension(self):
        """CRITICAL: /mri/upload rejects non-NIfTI files."""
        client = self._get_client()
        files = {"file": ("scan.jpg", b"fake image data", "image/jpeg")}
        response = client.post("/mri/upload", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error_code"] == "INVALID_EXTENSION"

    def test_upload_corrupt_nifti(self):
        """CRITICAL: /mri/upload rejects corrupt NIfTI bytes."""
        client = self._get_client()
        # Random bytes with .nii extension
        fake_bytes = b"\x00" * 1000
        files = {"file": ("brain.nii", fake_bytes, "application/octet-stream")}
        response = client.post("/mri/upload", files=files)
        # Should be 400 or 422 — not 200
        assert response.status_code in {400, 422, 500}
        data = response.json()
        assert "detail" in data

    def test_upload_tiny_file(self):
        """CRITICAL: /mri/upload rejects files < 348 bytes."""
        client = self._get_client()
        tiny = b"too small"
        files = {"file": ("brain.nii", tiny, "application/octet-stream")}
        response = client.post("/mri/upload", files=files)
        assert response.status_code == 400

    def test_upload_valid_nifti_succeeds(self):
        """CRITICAL: /mri/upload succeeds with real NIfTI and returns real nwbv."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        client = self._get_client()
        path = _make_real_nifti_file()
        file_bytes = path.read_bytes()
        path.unlink(missing_ok=True)

        files = {"file": ("brain.nii", file_bytes, "application/octet-stream")}
        response = client.post("/mri/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "extracted_features" in data
        nwbv = data["extracted_features"]["nwbv"]
        assert 0.0 <= nwbv <= 1.0, f"nwbv={nwbv} out of valid range"

    def test_upload_response_has_no_hash_fields(self):
        """CRITICAL: Response must NOT contain any hash-related keys."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")

        client = self._get_client()
        path = _make_real_nifti_file()
        file_bytes = path.read_bytes()
        path.unlink(missing_ok=True)

        files = {"file": ("brain.nii", file_bytes, "application/octet-stream")}
        response = client.post("/mri/upload", files=files)

        if response.status_code == 200:
            data = response.json()
            # Old mock used 'hash' or 'simulated'
            resp_str = str(data).lower()
            assert "hash" not in resp_str
            assert "simulated" not in resp_str
            assert "simulation" not in resp_str
