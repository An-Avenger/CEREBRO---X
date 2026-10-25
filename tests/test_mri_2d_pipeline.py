"""
Tests for 2D MRI CNN Pipeline: Preprocessing, Model, Splitting, Training & API Integration.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from cerebro_x.api.main import app
from cerebro_x.api.services.model_loader import get_registry
from cerebro_x.explainability.mri_2d_shap import MRI2DExplainer
from cerebro_x.imaging.dicom_preprocessing import (
    SliceExtractionConfig,
    normalise_and_resize_slice,
)
from cerebro_x.imaging.mri_slice_dataset import MRISliceDataset
from cerebro_x.models.deep.cnn2d import MRI2DCNN


@pytest.fixture(scope="module")
def client():
    # Ensure registry loaded
    reg = get_registry()
    reg.load_all()
    with TestClient(app) as test_client:
        yield test_client


def test_normalise_and_resize_slice():
    raw_slice = np.random.uniform(100, 2000, size=(180, 200)).astype(np.float32)
    norm = normalise_and_resize_slice(raw_slice, target_size=(224, 224))

    assert norm.shape == (224, 224)
    assert norm.dtype == np.float32
    assert 0.0 <= norm.min() <= norm.max() <= 1.0


def test_mri_2d_cnn_forward():
    model = MRI2DCNN(num_classes=3, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 3)
    probs = torch.softmax(out, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_subject_split_isolation():
    split_file = Path("artifacts/EXP-MRI-2D-CNN-001/subject_splits.csv")
    assert split_file.exists(), "subject_splits.csv should exist"

    df = pd.read_csv(split_file)
    train_subjs = set(df[df["split"] == "train"]["subject_id"])
    val_subjs = set(df[df["split"] == "val"]["subject_id"])
    test_subjs = set(df[df["split"] == "test"]["subject_id"])

    # Strict subject-level exclusivity
    assert len(train_subjs.intersection(val_subjs)) == 0, "Train and Val subjects must not overlap"
    assert len(train_subjs.intersection(test_subjs)) == 0, "Train and Test subjects must not overlap"
    assert len(val_subjs.intersection(test_subjs)) == 0, "Val and Test subjects must not overlap"


def test_checkpoint_reloading():
    checkpoint_path = Path("artifacts/EXP-MRI-2D-CNN-001/best_checkpoint.pt")
    assert checkpoint_path.exists(), "Best checkpoint must exist"

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "model_state_dict" in ckpt
    assert ckpt["num_classes"] == 3

    model = MRI2DCNN(num_classes=3, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = torch.randn(1, 1, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3)


def test_mri_2d_explainer():
    model = MRI2DCNN(num_classes=3, pretrained=False)
    explainer = MRI2DExplainer(model, device="cpu")

    dummy_slice = torch.randn(1, 1, 224, 224)
    res = explainer.explain(dummy_slice, steps=5)

    assert "predicted_class" in res
    assert "probabilities" in res
    assert "attribution" in res
    assert res["attribution"].shape == (224, 224)
    assert np.all(np.isfinite(res["attribution"]))
    assert 0.0 <= res["attribution"].min() <= res["attribution"].max() <= 1.0


def test_api_predict_slice_endpoint(client):
    # Test with synthetic .npy slice
    dummy_arr = np.random.uniform(0, 1, size=(224, 224)).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, dummy_arr)
    buf.seek(0)

    response = client.post(
        "/mri/predict-slice?explain=true",
        files={"file": ("test_slice.npy", buf.getvalue(), "application/octet-stream")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["predicted_class"] in ["CN", "MCI", "AD"]
    assert "probabilities" in data
    assert "explanation" in data


def test_api_predict_slice_invalid_input(client):
    # Upload corrupt non-image bytes
    corrupt_bytes = b"NOT_A_VALID_IMAGE_OR_NPY_FILE"
    response = client.post(
        "/mri/predict-slice",
        files={"file": ("corrupt.png", corrupt_bytes, "image/png")},
    )
    assert response.status_code == 400


def test_api_predict_multislice_endpoint(client):
    # Upload 3 synthetic slices
    files = []
    for i in range(3):
        buf = io.BytesIO()
        arr = np.random.uniform(0, 1, size=(224, 224)).astype(np.float32)
        np.save(buf, arr)
        buf.seek(0)
        files.append(("files", (f"slice_{i}.npy", buf.getvalue(), "application/octet-stream")))

    response = client.post("/mri/predict-multislice", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["slices_evaluated"] == 3
    assert data["consensus_prediction"] in ["CN", "MCI", "AD"]
    assert len(data["per_slice_breakdown"]) == 3
    assert "consensus_probabilities" in data
