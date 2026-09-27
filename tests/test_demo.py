import pytest
from fastapi.testclient import TestClient
from cerebro_x.api.main import app

client = TestClient(app)

def test_demo_api_availability():
    """Verify that all APIs needed for the Demo Mode are responsive."""
    
    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    
    # 2. History loading
    res = client.get("/history/")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
    
    # 3. Experiments API
    res = client.get("/experiments/")
    assert res.status_code == 200
    
    # 4. Clinical prediction with sample data
    # Mock some basic valid input structure required by the API
    sample_payload = {
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
                "asf": 1.2
            }
        ]
    }
    res = client.post("/predict/clinical", json=sample_payload)
    assert res.status_code == 200
    assert "prediction" in res.json()
    
    # 5. Bimodal prediction
    res = client.post("/predict/bimodal", json=sample_payload)
    assert res.status_code == 200
    assert "prediction" in res.json()
    
    # 6. Brain Twin Z_t extraction
    res = client.post("/brain-twin/extract", json=sample_payload)
    assert res.status_code == 200
    assert "z_t" in res.json()

def test_mri_demo_status():
    """Verify that MRI upload gives honest error when no checkpoint is available."""
    # Since we know no NIfTI is actually passed, it should throw a 400 or handle properly.
    res = client.post("/mri/upload")
    # Missing file should be 422 Unprocessable Entity in FastAPI
    assert res.status_code == 422
