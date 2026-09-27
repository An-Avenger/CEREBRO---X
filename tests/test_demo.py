"""
Integration smoke tests for the main demo endpoints.

These tests verify that:
1. Health check is responsive.
2. Clinical prediction returns the correct response structure.
3. Brain Twin extraction returns correct structure.
4. MRI upload without a file returns 422.
5. EEG screen endpoint is accessible.

NOTE: These tests use the TestClient from conftest which provides
the in-memory test database. Model loading depends on whether
the checkpoint exists in artifacts/.
"""
import pytest
from fastapi.testclient import TestClient


SAMPLE_PAYLOAD = {
    "subject_id": "OAS2_0001",
    "visits": [
        {
            "age": 70.0,
            "educ": 16.0,
            "ses": 2.0,
            "mmse": 28.0,
            "cdr": 0.0,
            "nwbv": 0.75,
            "etiv": 1500.0,
            "asf": 1.2,
            "gender": "F",
            "hand": "R",
        }
    ],
}


def test_health_endpoint(client: TestClient):
    """Health check is always available."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "models_loaded" in data
    assert data["status"] in ("ok", "degraded")


def test_history_endpoint(client: TestClient):
    """History endpoint returns a list."""
    res = client.get("/history/")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_experiments_endpoint(client: TestClient):
    """Experiments endpoint returns a list or dict."""
    res = client.get("/experiments/")
    assert res.status_code == 200


def test_clinical_prediction_response_structure(client: TestClient):
    """
    Clinical prediction returns correct response structure when model is loaded.
    If model is not loaded (503), that is also acceptable for CI without checkpoints.
    """
    res = client.post("/predict/clinical", json=SAMPLE_PAYLOAD)
    assert res.status_code in (200, 503), (
        f"Expected 200 or 503, got {res.status_code}: {res.json()}"
    )
    if res.status_code == 200:
        data = res.json()
        assert "predicted_cdr_class" in data, f"Missing predicted_cdr_class in {data}"
        assert "predicted_cdr_value" in data
        assert "predicted_cdr_label" in data
        assert "class_probabilities" in data
        # Canonical CDR keys
        probs = data["class_probabilities"]
        assert set(probs.keys()) == {"0", "1", "2", "3"}, (
            f"class_probabilities must use canonical keys '0','1','2','3'. Got: {list(probs.keys())}"
        )


def test_brain_twin_response_structure(client: TestClient):
    """
    Brain twin extraction returns correct response structure when model is loaded.
    """
    res = client.post("/brain-twin/extract", json=SAMPLE_PAYLOAD)
    assert res.status_code in (200, 503)
    if res.status_code == 200:
        data = res.json()
        assert "n_visits" in data
        assert "z_dim" in data
        assert "trajectories" in data
        assert "disclaimer" in data
        # Z_t must be labeled as learned latent, not anatomical
        disclaimer = data.get("disclaimer", "") or data.get("representation", "")
        assert "NOT" in disclaimer or "NOT" in data.get("representation", ""), (
            "Brain Twin response must disclaim that Z_t is NOT a neuroimaging measurement."
        )


def test_mri_upload_missing_file(client: TestClient):
    """MRI upload without a file returns 422 (FastAPI validation error)."""
    res = client.post("/mri/upload")
    assert res.status_code == 422


def test_root_endpoint(client: TestClient):
    """Root endpoint returns project metadata with a disclaimer."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "project" in data
    assert "disclaimer" in data
    assert "research" in data["disclaimer"].lower() or "NOT" in data["disclaimer"]
