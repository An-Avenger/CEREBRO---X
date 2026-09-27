"""Tests for MRI integration pipeline (Phase B)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cerebro_x.imaging.subject_matcher import MRISubjectMatcher
from cerebro_x.imaging.nifti_loader import NiftiLoader, MRILoadStatus
from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline
from cerebro_x.imaging.qc import MRIQualityChecker

# Mock nibabel for tests so we don't require real NIfTI files or the library
# to run the core logic tests.
class MockNibabelImage:
    def __init__(self, data, affine=None):
        self.data = data
        self.shape = data.shape
        self.affine = affine if affine is not None else np.eye(4)
        
    def get_fdata(self):
        return self.data
        
    class MockHeader:
        def get_zooms(self):
            return (1.0, 1.0, 1.0)
            
    @property
    def header(self):
        return self.MockHeader()

class MockNibabel:
    @staticmethod
    def load(path):
        if "corrupted" in str(path):
            raise ValueError("Corrupted file")
        if "wrong_dim" in str(path):
            # 2D image instead of 3D
            return MockNibabelImage(np.zeros((64, 64)))
        # Return a valid 3D image
        return MockNibabelImage(np.random.rand(64, 64, 64))
        
    @staticmethod
    def as_closest_canonical(img):
        return img


# Install mock BEFORE tests in this module, and explicitly restore after.
# We use a function-scoped fixture (autouse=True) that patches per test,
# which avoids leaking into other test modules.
@pytest.fixture(autouse=True)
def _mock_nibabel_for_this_module(monkeypatch):
    """
    Patches nibabel in nifti_loader for EACH test in this module.
    monkeypatch auto-restores after each test, so no cross-module pollution.
    """
    import cerebro_x.imaging.nifti_loader as loader_module
    monkeypatch.setattr(loader_module, "nib", MockNibabel())
    monkeypatch.setattr(loader_module, "_NIB_AVAILABLE", True)
    yield
    # monkeypatch auto-restores here






@pytest.fixture
def mock_mri_dir(tmp_path):
    """Create a mock directory structure for subject matching tests."""
    mri_dir = tmp_path / "mri_data"
    mri_dir.mkdir()
    
    # Valid BIDS format
    (mri_dir / "sub-OAS20001_ses-d0001_T1w.nii.gz").touch()
    
    # Valid Legacy format
    (mri_dir / "OAS2_0002_MR1.nii").touch()
    
    # Ambiguous match setup
    (mri_dir / "OAS2_0003_MR1_run1.nii").touch()
    (mri_dir / "OAS2_0003_MR1_run2.nii").touch()
    
    # Test for nifti loader
    (mri_dir / "test_load_OAS2_0004.nii").touch()
    (mri_dir / "corrupted.nii").touch()
    (mri_dir / "wrong_dim.nii").touch()
    
    return mri_dir


class TestMRISubjectMatcher:
    def test_normalize_subject_id(self):
        matcher = MRISubjectMatcher()
        variants = matcher._normalize_subject_id("OAS2_0001")
        assert "OAS2_0001" in variants
        assert "OAS20001" in variants
        
    def test_find_bids_format(self, mock_mri_dir):
        matcher = MRISubjectMatcher(mock_mri_dir)
        path = matcher.find_mri_for_visit("OAS2_0001", 1)
        assert path is not None
        assert path.name == "sub-OAS20001_ses-d0001_T1w.nii.gz"
        
    def test_find_legacy_format(self, mock_mri_dir):
        matcher = MRISubjectMatcher(mock_mri_dir)
        path = matcher.find_mri_for_visit("OAS2_0002", 1)
        assert path is not None
        assert path.name == "OAS2_0002_MR1.nii"
        
    def test_ambiguous_match_returns_none(self, mock_mri_dir):
        matcher = MRISubjectMatcher(mock_mri_dir)
        path = matcher.find_mri_for_visit("OAS2_0003", 1)
        # Should return None to prevent contamination when multiple files match
        assert path is None
        
    def test_missing_subject_returns_none(self, mock_mri_dir):
        matcher = MRISubjectMatcher(mock_mri_dir)
        path = matcher.find_mri_for_visit("OAS2_9999", 1)
        assert path is None
        
    def test_validate_match(self):
        matcher = MRISubjectMatcher()
        assert matcher.validate_match("OAS2_0001", "path/to/sub-OAS20001_ses-d0000_T1w.nii.gz")
        assert matcher.validate_match("OAS2_0001", "path/to/OAS2_0001_MR1.nii")
        assert not matcher.validate_match("OAS2_0001", "path/to/OAS2_0002_MR1.nii")


class TestNiftiLoader:
    def test_load_success(self, mock_mri_dir):
        loader = NiftiLoader()
        file_path = mock_mri_dir / "test_load_OAS2_0004.nii"
        
        result = loader.load(file_path, "OAS2_0004", 1)
        
        assert result.ok
        assert result.status == MRILoadStatus.OK
        assert isinstance(result.tensor, torch.Tensor)
        assert result.tensor.shape == (64, 64, 64)
        
    def test_load_missing_file(self):
        loader = NiftiLoader()
        result = loader.load("non_existent.nii", "OAS2_0004", 1)
        
        assert not result.ok
        assert result.status == MRILoadStatus.MISSING
        assert result.tensor is None
        
    def test_load_wrong_subject_validation(self, mock_mri_dir):
        loader = NiftiLoader()
        file_path = mock_mri_dir / "test_load_OAS2_0004.nii"
        
        result = loader.load(
            file_path, 
            "OAS2_9999", 
            1, 
            validate_subject_in_filename=True
        )
        
        assert not result.ok
        assert result.status == MRILoadStatus.WRONG_SUBJECT
        
    def test_load_corrupted_file(self, mock_mri_dir):
        loader = NiftiLoader()
        file_path = mock_mri_dir / "corrupted.nii"
        
        result = loader.load(file_path, "OAS2_0004", 1)
        assert not result.ok
        assert result.status == MRILoadStatus.UNREADABLE
        
    def test_load_wrong_dimensions(self, mock_mri_dir):
        loader = NiftiLoader()
        file_path = mock_mri_dir / "wrong_dim.nii"
        
        result = loader.load(file_path, "OAS2_0004", 1)
        assert not result.ok
        assert result.status == MRILoadStatus.WRONG_DIMENSIONS


class TestMRIPreprocessingPipeline:
    def test_resize(self):
        pipeline = MRIPreprocessingPipeline(target_shape=(32, 32, 32))
        
        # Start with 64x64x64
        tensor = torch.ones((64, 64, 64))
        resized = pipeline._resize_volume(tensor)
        
        assert resized.shape == (32, 32, 32)
        
    def test_normalization(self):
        pipeline = MRIPreprocessingPipeline()
        
        # Create a dummy brain with distinct background
        tensor = torch.zeros((64, 64, 64))
        # Brain tissue in the center
        tensor[16:48, 16:48, 16:48] = torch.randn((32, 32, 32)) * 10 + 100
        # Extreme artifact
        tensor[32, 32, 32] = 1000
        
        normalized = pipeline._normalize_intensity(tensor)
        
        # Check brain voxels have roughly 0 mean and 1 std
        mask = normalized > normalized.min() + 0.1
        assert abs(normalized[mask].mean().item()) < 0.1
        assert abs(normalized[mask].std().item() - 1.0) < 0.1
        
        # Check artifact was clipped
        assert normalized.max() < 10.0  # Should be clipped heavily relative to z-score
        
    def test_full_pipeline(self):
        pipeline = MRIPreprocessingPipeline(target_shape=(32, 32, 32))
        tensor = torch.randn((64, 64, 64)) * 10 + 100
        
        result = pipeline(tensor)
        
        assert result.shape == (32, 32, 32)
        assert torch.isfinite(result).all()


class TestMRIQualityChecker:
    def test_qc_pass(self):
        qc = MRIQualityChecker()
        
        # Healthy simulated tensor (brain tissue + background)
        tensor = torch.zeros((64, 64, 64))
        tensor[16:48, 16:48, 16:48] = torch.rand((32, 32, 32)) * 100 + 10
        
        result = qc.check(tensor)
        assert result["passed"] is True
        
    def test_qc_fail_empty(self):
        qc = MRIQualityChecker()
        
        # Completely blank scan
        tensor = torch.zeros((64, 64, 64))
        
        result = qc.check(tensor)
        assert result["passed"] is False
        assert "below minimum" in result["reason"]
        
    def test_qc_fail_all_signal(self):
        qc = MRIQualityChecker()
        
        # No background (cropping error)
        tensor = torch.ones((64, 64, 64)) * 100
        
        result = qc.check(tensor)
        assert result["passed"] is False
        assert "above maximum" in result["reason"]
        
    def test_qc_fail_low_dynamic_range(self):
        """
        A near-uniform tensor (minimal variation, high signal everywhere) fails QC.
        The QC checker may flag it as 'above maximum' (too much signal, no background)
        or 'below minimum' (no real brain structure), depending on the threshold logic.
        """
        qc = MRIQualityChecker()

        # Signal variation is tiny across the whole tensor — nearly uniform high signal
        tensor = torch.ones((64, 64, 64)) * 10.0
        tensor[16:48, 16:48, 16:48] = torch.rand((32, 32, 32)) * 0.1 + 10.0

        result = qc.check(tensor)
        assert result["passed"] is False
        # The QC may report either "above maximum" or "below minimum" depending on
        # which threshold is violated first (brain fraction vs signal ratio)
        assert "below minimum" in result["reason"] or "above maximum" in result["reason"], (
            f"Unexpected QC failure reason: {result['reason']}"
        )

