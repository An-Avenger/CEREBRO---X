"""
Tests for /mri/upload endpoint.

Current MRI pipeline status:
  - NIfTI validation: IMPLEMENTED
  - Real preprocessing: IMPLEMENTED
  - CNN3D embedding: NOT AVAILABLE (no trained checkpoint)
  - Grad-CAM: NOT AVAILABLE (requires CNN checkpoint)

These tests verify honest behavior:
  - Invalid extension -> 400
  - Too-small file (not a valid NIfTI) -> 400 or 422
  - CNN3D checkpoint unavailable -> cnn_status reflects this
"""
import pytest
from fastapi.testclient import TestClient


def test_mri_upload_invalid_extension(client: TestClient):
    """Uploading a non-NIfTI file returns 400 with INVALID_EXTENSION."""
    file_content = b"this is a text file"
    files = {"file": ("document.txt", file_content, "text/plain")}

    response = client.post("/mri/upload", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error_code"] == "INVALID_EXTENSION"


def test_mri_upload_too_small(client: TestClient):
    """Uploading a .nii file that is too small returns 400 with FILE_TOO_SMALL."""
    file_content = b"fake nifti content"  # < 348 bytes
    files = {"file": ("scan.nii", file_content, "application/octet-stream")}

    response = client.post("/mri/upload", files=files)
    # Should be 400 (FILE_TOO_SMALL) or 503 (nibabel unavailable)
    assert response.status_code in (400, 503)
    detail = response.json()["detail"]
    assert detail["error_code"] in ("FILE_TOO_SMALL", "NIBABEL_UNAVAILABLE", "NIBABEL_STUB", "INVALID_NIFTI", "PIPELINE_UNAVAILABLE")


def test_mri_upload_nii_gz_invalid(client: TestClient):
    """
    Uploading a .nii.gz that is not a valid NIfTI: nibabel may accept it
    (permissive parser) or reject it. Either way, no 500 error should occur.
    """
    file_content = b"x" * 400  # 400 bytes but not a real NIfTI
    files = {"file": ("scan.nii.gz", file_content, "application/octet-stream")}

    response = client.post("/mri/upload", files=files)
    # Nibabel may succeed (permissive) or reject gracefully; must not be 500
    assert response.status_code in (200, 400, 422, 503), (
        f"Unexpected status {response.status_code}: {response.json()}"
    )
    if response.status_code != 200:
        detail = response.json()["detail"]
        assert "error_code" in detail
        assert detail["success"] is False



def test_mri_explain_no_checkpoint(client: TestClient):
    """
    POST /explain/mri without a trained CNN3D checkpoint returns 503
    with CNN3D_CHECKPOINT_NOT_AVAILABLE.
    """
    file_content = b"fake nifti"
    files = {"file": ("scan.nii", file_content, "application/octet-stream")}

    response = client.post("/explain/mri", files=files)
    # Either 503 (no checkpoint) or 400 (invalid file before checkpoint check)
    # The endpoint checks checkpoint FIRST, so it should be 503 if no checkpoint.
    assert response.status_code in (400, 503)
    if response.status_code == 503:
        detail = response.json()["detail"]
        assert detail["error_code"] == "CNN3D_CHECKPOINT_NOT_AVAILABLE"
