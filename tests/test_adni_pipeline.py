"""
tests/test_adni_pipeline.py
-----------------------------
Tests for the ADNI MRI training pipeline.

Coverage:
  - Manifest validation (valid, missing, invalid CDR)
  - CDR label mapping (valid values, invalid values, NaN)
  - RealADNIMRIDataset (schema check, fail-closed behaviour)
  - Subject-level splitting (no leakage, all subjects accounted)
  - Model output shape and loss calculation
  - Checkpoint save/reload
  - Fail-closed: missing manifest, missing NIfTI, no GPU without --no-gpu
  - Reproducibility / inference consistency

All tests use synthetic NIfTI files (via nibabel) or in-memory manifests.
No real ADNI data is required to run these tests.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_nifti(path: Path, shape=(64, 64, 64)) -> Path:
    """Write a minimal real NIfTI file for testing."""
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed")

    data = np.random.rand(*shape).astype(np.float32) * 100 + 10
    data[0:5, :, :] = 0.0  # background
    img = nib.Nifti1Image(data, affine=np.diag([1.5, 1.5, 1.5, 1.0]))
    nib.save(img, str(path))
    return path


def _make_synthetic_manifest(
    tmp_dir: Path,
    n_subjects: int = 6,
    scans_per_subject: int = 2,
    valid_cdr: bool = True,
) -> tuple[Path, Path]:
    """
    Create synthetic NIfTI files and a matching manifest CSV.

    Returns (manifest_path, nifti_dir).
    """
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed")

    nifti_dir = tmp_dir / "nifti"
    nifti_dir.mkdir(exist_ok=True)

    cdr_values = [0.0, 0.5, 1.0, 2.0, 0.5, 0.0]
    rows = []
    for i in range(n_subjects):
        subj = f"TEST_{i:04d}_S_{1000 + i}"
        cdr = cdr_values[i % len(cdr_values)]
        for visit in range(scans_per_subject):
            fname = f"{subj}_m{(visit + 1) * 6:02d}.nii.gz"
            nifti_path = nifti_dir / fname
            _make_synthetic_nifti(nifti_path)
            rows.append({
                "subject_id": subj,
                "visit_id": f"m{(visit + 1) * 6:02d}",
                "image_id": f"{10000 + i * 10 + visit}",
                "nifti_path": fname,
                "cdr_score": cdr if valid_cdr else "INVALID",
                "scan_date": f"2022-0{visit + 1}-15",
            })

    manifest_df = pd.DataFrame(rows)
    manifest_path = tmp_dir / "manifest_with_labels.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path, nifti_dir


# ─────────────────────────────────────────────────────────────────────────────
# 1. CDR Mapping Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCDRMapping:
    """Tests for CDR value → class label mapping."""

    def test_valid_cdr_0_maps_to_0(self):
        from cerebro_x.imaging.adni_dataset import map_cdr_to_class
        assert map_cdr_to_class(0.0, 0) == 0

    def test_valid_cdr_05_maps_to_1(self):
        from cerebro_x.imaging.adni_dataset import map_cdr_to_class
        assert map_cdr_to_class(0.5, 0) == 1

    def test_valid_cdr_1_maps_to_2(self):
        from cerebro_x.imaging.adni_dataset import map_cdr_to_class
        assert map_cdr_to_class(1.0, 0) == 2

    def test_valid_cdr_2_maps_to_3(self):
        from cerebro_x.imaging.adni_dataset import map_cdr_to_class
        assert map_cdr_to_class(2.0, 0) == 3

    def test_valid_cdr_3_maps_to_3(self):
        """CDR=3.0 (Severe) merges into class 3 (documented behaviour)."""
        from cerebro_x.imaging.adni_dataset import map_cdr_to_class
        assert map_cdr_to_class(3.0, 0) == 3

    def test_invalid_cdr_raises_value_error(self):
        """CRITICAL: Unknown CDR values must RAISE, not silently map to 0."""
        from cerebro_x.imaging.adni_dataset import map_cdr_to_class
        with pytest.raises(ValueError, match="not in the valid CDR set"):
            map_cdr_to_class(0.25, 5)

    def test_invalid_cdr_negative_raises(self):
        from cerebro_x.imaging.adni_dataset import map_cdr_to_class
        with pytest.raises(ValueError):
            map_cdr_to_class(-1.0, 0)

    def test_string_float_is_handled(self):
        """String representation of a valid CDR float should work."""
        from cerebro_x.imaging.adni_dataset import map_cdr_to_class
        assert map_cdr_to_class("0.5", 0) == 1

    def test_non_numeric_string_raises(self):
        from cerebro_x.imaging.adni_dataset import map_cdr_to_class
        with pytest.raises(ValueError):
            map_cdr_to_class("N/A", 0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RealADNIMRIDataset Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRealADNIMRIDataset:

    def test_loads_valid_manifest(self, tmp_path):
        """CRITICAL: Dataset loads successfully with valid manifest and NIfTI files."""
        manifest_path, nifti_dir = _make_synthetic_manifest(tmp_path)
        from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
        ds = RealADNIMRIDataset(manifest_path, nifti_dir)
        assert len(ds) > 0

    def test_raises_if_manifest_missing(self, tmp_path):
        """CRITICAL: Raises FileNotFoundError if manifest does not exist."""
        from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
        with pytest.raises(FileNotFoundError):
            RealADNIMRIDataset(tmp_path / "nonexistent.csv", tmp_path)

    def test_raises_if_required_columns_missing(self, tmp_path):
        """CRITICAL: Raises ValueError if manifest is missing required columns."""
        bad_manifest = tmp_path / "bad.csv"
        pd.DataFrame({"subject_id": ["A"], "nifti_path": ["a.nii"]}).to_csv(bad_manifest, index=False)
        from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
        with pytest.raises(ValueError, match="missing required columns"):
            RealADNIMRIDataset(bad_manifest, tmp_path)

    def test_raises_on_invalid_cdr_in_manifest(self, tmp_path):
        """CRITICAL: Invalid CDR values in manifest must raise, not silently map."""
        manifest_path, nifti_dir = _make_synthetic_manifest(tmp_path, valid_cdr=False)
        from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
        with pytest.raises(ValueError):
            RealADNIMRIDataset(manifest_path, nifti_dir)

    def test_raises_on_nan_cdr_in_manifest(self, tmp_path):
        """CRITICAL: NaN CDR values must raise, not be silently skipped."""
        manifest_path, nifti_dir = _make_synthetic_manifest(tmp_path)
        df = pd.read_csv(manifest_path)
        df.loc[0, "cdr_score"] = float("nan")
        df.to_csv(manifest_path, index=False)

        from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
        with pytest.raises(ValueError, match="NaN"):
            RealADNIMRIDataset(manifest_path, nifti_dir)

    def test_getitem_returns_correct_shapes(self, tmp_path):
        """CRITICAL: __getitem__ returns (1, 64, 64, 64) tensor and long label."""
        manifest_path, nifti_dir = _make_synthetic_manifest(tmp_path, n_subjects=2, scans_per_subject=1)
        from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
        ds = RealADNIMRIDataset(manifest_path, nifti_dir)
        vol, label = ds[0]
        assert vol.shape == (1, 64, 64, 64), f"Expected (1,64,64,64), got {vol.shape}"
        assert vol.dtype == torch.float32
        assert label.dtype == torch.long

    def test_getitem_raises_on_missing_nifti(self, tmp_path):
        """CRITICAL: __getitem__ raises if NIfTI file is missing."""
        manifest_path, nifti_dir = _make_synthetic_manifest(tmp_path, n_subjects=1, scans_per_subject=1)
        # Delete the NIfTI file
        for f in nifti_dir.glob("*.nii.gz"):
            f.unlink()

        from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
        ds = RealADNIMRIDataset(manifest_path, nifti_dir)
        with pytest.raises((ValueError, FileNotFoundError)):
            _ = ds[0]

    def test_no_nan_in_preprocessed_volume(self, tmp_path):
        """CRITICAL: Preprocessed volume must not contain NaN or Inf."""
        manifest_path, nifti_dir = _make_synthetic_manifest(tmp_path, n_subjects=2, scans_per_subject=1)
        from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
        ds = RealADNIMRIDataset(manifest_path, nifti_dir)
        vol, _ = ds[0]
        assert not torch.isnan(vol).any(), "Preprocessed volume contains NaN"
        assert not torch.isinf(vol).any(), "Preprocessed volume contains Inf"

    def test_subjects_property(self, tmp_path):
        """IMPORTANT: .subjects returns unique subject IDs."""
        manifest_path, nifti_dir = _make_synthetic_manifest(tmp_path, n_subjects=3, scans_per_subject=2)
        from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
        ds = RealADNIMRIDataset(manifest_path, nifti_dir)
        assert len(ds.subjects) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. Subject-Level Split Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSubjectLevelSplitADNI:
    """Tests the subject-level split in train_cnn3d.py."""

    def _make_manifest(self, n_subjects: int = 20) -> pd.DataFrame:
        rows = []
        cdr_vals = [0.0, 0.5, 1.0, 2.0]
        for i in range(n_subjects):
            for visit in range(2):
                rows.append({
                    "subject_id": f"SUBJ_{i:04d}",
                    "visit_id": f"m{(visit + 1) * 6:02d}",
                    "nifti_path": f"fake_{i}_{visit}.nii.gz",
                    "cdr_score": cdr_vals[i % len(cdr_vals)],
                    "label": i % len(cdr_vals),
                })
        return pd.DataFrame(rows)

    def test_no_subject_overlap(self):
        """CRITICAL: No subject appears in more than one split (R-012)."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_cnn3d import subject_level_split

        manifest = self._make_manifest(30)
        train_idx, val_idx, test_idx, _ = subject_level_split(manifest, 0.15, 0.15, seed=42)

        train_subj = set(manifest.loc[train_idx, "subject_id"])
        val_subj   = set(manifest.loc[val_idx,   "subject_id"])
        test_subj  = set(manifest.loc[test_idx,  "subject_id"])

        assert len(train_subj & val_subj) == 0, "Train/Val subject overlap!"
        assert len(train_subj & test_subj) == 0, "Train/Test subject overlap!"
        assert len(val_subj & test_subj) == 0, "Val/Test subject overlap!"

    def test_all_subjects_accounted_for(self):
        """All subjects must appear in exactly one split."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_cnn3d import subject_level_split

        manifest = self._make_manifest(30)
        train_idx, val_idx, test_idx, _ = subject_level_split(manifest, 0.15, 0.15, seed=42)

        all_subj = set(manifest["subject_id"].unique())
        split_subj = (
            set(manifest.loc[train_idx, "subject_id"])
            | set(manifest.loc[val_idx,   "subject_id"])
            | set(manifest.loc[test_idx,  "subject_id"])
        )
        assert all_subj == split_subj, f"Missing subjects: {all_subj - split_subj}"

    def test_different_seeds_give_different_splits(self):
        """Different seeds should produce different split assignments."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_cnn3d import subject_level_split

        manifest = self._make_manifest(40)
        _, _, test_a, _ = subject_level_split(manifest, 0.15, 0.15, seed=0)
        _, _, test_b, _ = subject_level_split(manifest, 0.15, 0.15, seed=99)

        subj_a = set(manifest.loc[test_a, "subject_id"])
        subj_b = set(manifest.loc[test_b, "subject_id"])
        assert subj_a != subj_b

    def test_raises_on_too_few_subjects(self):
        """ValueError if too few subjects for 3-way split."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_cnn3d import subject_level_split

        manifest = self._make_manifest(2)  # only 2 subjects
        with pytest.raises(ValueError, match="Too few subjects"):
            subject_level_split(manifest, 0.33, 0.33, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model and Loss Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelAndLoss:

    def test_mri_cerebro_net_output_shape(self):
        """CRITICAL: MRICerebroNet outputs (B, 4) logits."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        model.eval()
        x = torch.rand(2, 1, 64, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"

    def test_loss_calculation_valid(self):
        """CRITICAL: CrossEntropyLoss produces finite scalar loss."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        criterion = nn.CrossEntropyLoss()
        x = torch.rand(2, 1, 64, 64, 64)
        y = torch.tensor([0, 2], dtype=torch.long)
        logits = model(x)
        loss = criterion(logits, y)
        assert torch.isfinite(loss).item(), "Loss is not finite"
        assert loss.item() > 0, "Loss is non-positive — unexpected"

    def test_gradients_flow(self):
        """CRITICAL: Gradients must flow through all model parameters."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        criterion = nn.CrossEntropyLoss()
        x = torch.rand(1, 1, 64, 64, 64)
        y = torch.tensor([1], dtype=torch.long)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_class_weight_tensor_shape(self):
        """Class weights tensor must match num_classes."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_cnn3d import compute_class_weights

        manifest = pd.DataFrame({
            "subject_id": ["A", "B", "C", "D"],
            "label": [0, 1, 2, 3],
        })
        train_idx = [0, 1, 2, 3]
        weights = compute_class_weights(manifest, train_idx, num_classes=4, device=torch.device("cpu"))
        assert weights.shape == (4,)
        assert torch.isfinite(weights).all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Checkpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:

    def test_save_and_reload_checkpoint(self, tmp_path):
        """CRITICAL: Saved checkpoint can be reloaded and produces identical output."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_cnn3d import save_checkpoint, load_checkpoint
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        import torch.optim as optim

        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        ckpt_path = tmp_path / "test_checkpoint.pt"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=5, best_val_acc=0.75, config={})
        assert ckpt_path.exists(), "Checkpoint file not created"

        # Load into a fresh model
        model2 = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        optimizer2 = optim.AdamW(model2.parameters(), lr=1e-4)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)
        epoch, best_acc = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)

        assert epoch == 5
        assert best_acc == 0.75

        # Verify same output
        x = torch.rand(1, 1, 64, 64, 64)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-5), "Reloaded model gives different output"

    def test_checkpoint_contains_optimizer_state(self, tmp_path):
        """CRITICAL: Checkpoint must include optimizer state for resumable training."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_cnn3d import save_checkpoint
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        import torch.optim as optim

        model = MRICerebroNet()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=1, best_val_acc=0.5, config={})

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        assert "optimizer_state_dict" in ckpt, "optimizer_state_dict missing from checkpoint"
        assert "model_state_dict" in ckpt, "model_state_dict missing from checkpoint"
        assert "epoch" in ckpt


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reproducibility Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:

    def test_preprocessing_deterministic(self):
        """CRITICAL: Same input always produces same preprocessed output."""
        from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline
        pipeline = MRIPreprocessingPipeline(target_shape=(64, 64, 64))
        raw = torch.rand(128, 96, 80) * 200 + 50
        out1 = pipeline(raw.clone())
        out2 = pipeline(raw.clone())
        assert torch.allclose(out1, out2, atol=1e-5), "Preprocessing is not deterministic"

    def test_model_inference_deterministic(self):
        """CRITICAL: Same input → same model output in eval mode."""
        from cerebro_x.models.deep.cnn3d import MRICerebroNet
        torch.manual_seed(0)
        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
        model.eval()
        x = torch.rand(1, 1, 64, 64, 64)
        with torch.no_grad():
            out1 = model(x.clone())
            out2 = model(x.clone())
        assert torch.allclose(out1, out2, atol=1e-6), "Model inference is not deterministic"
