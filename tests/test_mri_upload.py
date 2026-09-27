import pytest
from fastapi.testclient import TestClient

def test_mri_upload_valid_file(client: TestClient):
    """Test uploading a valid mock NIfTI file."""
    # Create a mock file
    file_content = b"fake nifti content for testing"
    files = {"file": ("scan.nii", file_content, "application/octet-stream")}
    
    response = client.post("/mri/upload", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert data["filename"] == "scan.nii"
    assert "extracted_features" in data
    assert "etiv" in data["extracted_features"]
    assert "nwbv" in data["extracted_features"]
    assert "asf" in data["extracted_features"]
    
    # Ensure pseudo-random generation is deterministic for same content
    features = data["extracted_features"]
    assert 1300 <= features["etiv"] <= 1700
    assert 0.65 <= features["nwbv"] <= 0.85
    assert 1.0 <= features["asf"] <= 1.4

def test_mri_upload_invalid_file(client: TestClient):
    """Test uploading a file with an invalid extension."""
    file_content = b"this is a text file"
    files = {"file": ("document.txt", file_content, "text/plain")}
    
    response = client.post("/mri/upload", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]
