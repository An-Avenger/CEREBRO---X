import pytest

# Sample payload for testing prediction
MOCK_PATIENT_PAYLOAD = {
    "subject_id": "TEST_SUBJECT_001",
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

def test_health_check(client):
    """Test the /health endpoint to ensure API and models load properly."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "clinical_gru" in data["models_loaded"]
    assert data["models_loaded"]["clinical_gru"] is True

def test_predict_clinical(client):
    """Test clinical GRU prediction endpoint."""
    response = client.post("/predict/clinical", json=MOCK_PATIENT_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    
    assert data["subject_id"] == "TEST_SUBJECT_001"
    assert "predicted_cdr_class" in data
    assert data["predicted_cdr_class"] in [0, 1, 2, 3] # Valid CDR classes (0.0, 0.5, 1.0, 2.0 mapped to 0-3)
    assert "predicted_cdr_value" in data
    assert "class_probabilities" in data
    assert len(data["class_probabilities"]) == 4

def test_patient_history_saving(client):
    """Test that predicting clinical data automatically saves to history DB."""
    # Ensure patient is created and record saved
    res_predict = client.post("/predict/clinical", json=MOCK_PATIENT_PAYLOAD)
    assert res_predict.status_code == 200
    
    # Fetch history
    res_history = client.get(f"/patients/{MOCK_PATIENT_PAYLOAD['subject_id']}/history")
    assert res_history.status_code == 200
    history_data = res_history.json()
    
    assert history_data["subject_id"] == MOCK_PATIENT_PAYLOAD["subject_id"]
    assert len(history_data["history"]) >= 1
    
    latest_record = history_data["history"][0]
    assert latest_record["age"] == MOCK_PATIENT_PAYLOAD["visits"][0]["age"]
    assert latest_record["mmse"] == MOCK_PATIENT_PAYLOAD["visits"][0]["mmse"]

def test_history_not_found(client):
    """Test history for a non-existent patient returns 404."""
    res = client.get("/patients/NON_EXISTENT/history")
    assert res.status_code == 404

def test_brain_twin_extract(client):
    """Test extracting the abstract brain twin state."""
    response = client.post("/brain-twin/extract", json=MOCK_PATIENT_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    
    assert data["n_visits"] == 1
    assert data["z_dim"] == 64
    assert "trajectories" in data
    assert len(data["trajectories"]) == 1 # 1 step for 1 visit
    assert len(data["trajectories"][0]) == 64 # 64 dims

def test_invalid_payload_error(client):
    """Test invalid payload correctly throws 422 Unprocessable Entity."""
    # Missing required 'visits' field
    invalid_payload = {"subject_id": "TEST_SUBJECT_001"}
    response = client.post("/predict/clinical", json=invalid_payload)
    assert response.status_code == 422
