"""
tests/test_robustness.py
=========================
Phase 11 — Validation & Robustness

Stress-tests the Cerebro X inference pipeline against 8 adversarial scenarios
without requiring real model checkpoints (uses architecture-only forward passes).

Scenarios:
    R-001  Missing clinical values (NaN injection)
    R-002  Single-visit sequences (length = 1)
    R-003  Corrupted / zero MRI scalars
    R-004  Extreme outlier MRI feature values
    R-005  All-same-class inputs (class imbalance simulation)
    R-006  Very long temporal gap sequences (5+ visits, large time jump)
    R-007  Feature distribution shift (scaled 2x / 0.5x)
    R-008  Determinism / random seed sensitivity

Each test:
    - Constructs an untrained (randomly initialised) model identical to production.
    - Calls the same inference helpers used by the FastAPI layer.
    - Asserts that the system does NOT crash and returns a structurally valid response.
    - Does NOT assert accuracy — this tests robustness, not performance.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.models.deep.temporal import TemporalCerebroNet
from cerebro_x.models.deep.bimodal_fusion import BimodalCerebroNet
from cerebro_x.api.services.inference import (
    run_clinical_inference,
    run_bimodal_inference,
    visits_to_feature_tensor,
)
from cerebro_x.features.preprocessor import CDR_CLASS_TO_KEY, CDR_CLASS_TO_VALUE

# Canonical probability keys ("0", "1", "2", "3")
CDR_LABELS = CDR_CLASS_TO_KEY  # int -> "0"/"1"/"2"/"3"
CDR_VALUES = CDR_CLASS_TO_VALUE  # int -> 0.0/0.5/1.0/2.0



# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clinical_gru() -> TemporalCerebroNet:
    """Randomly initialised (untrained) GRU — same architecture as production."""
    torch.manual_seed(42)
    model = TemporalCerebroNet(
        input_dim=19, gru_hidden_dim=64, gru_num_layers=1,
        num_classes=4, dropout=0.3
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def bimodal_model() -> BimodalCerebroNet:
    """Randomly initialised bimodal fusion model."""
    torch.manual_seed(42)
    model = BimodalCerebroNet(
        clinical_input_dim=19, mri_input_dim=4,
        clinical_hidden_dim=64, mri_embed_dim=32,
        num_classes=4, dropout=0.3
    )
    model.eval()
    return model


def _make_visit(
    age: float = 72.0,
    educ: float = 12.0,
    ses: float = 2.0,
    mmse: float = 28.0,
    cdr: float = 0.0,
    nwbv: float = 0.75,
    etiv: float = 1500.0,
    asf: float = 1.2,
) -> dict:
    return dict(age=age, educ=educ, ses=ses, mmse=mmse, cdr=cdr,
                nwbv=nwbv, etiv=etiv, asf=asf, gender="F", hand="R")


def _standard_visits(n: int = 3) -> list[dict]:
    """Return n healthy baseline visits."""
    return [_make_visit(age=70.0 + i * 2) for i in range(n)]


def _validate_result(result: dict) -> None:
    """Assert that inference output is structurally valid."""
    assert "predicted_cdr_class" in result
    assert "predicted_cdr_value" in result
    assert "predicted_cdr_label" in result
    assert "class_probabilities" in result

    cls = result["predicted_cdr_class"]
    assert cls in (0, 1, 2, 3), f"CDR class out of range: {cls}"

    probs = list(result["class_probabilities"].values())
    assert len(probs) == 4, f"Expected 4 class probabilities, got {len(probs)}"
    total = sum(probs)
    assert abs(total - 1.0) < 1e-3, f"Probabilities don't sum to 1: {total}"

    for p in probs:
        assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"
        assert not math.isnan(p), "NaN probability detected"


# ─── R-001: Missing clinical values (NaN injection) ──────────────────────────

class TestR001MissingValues:
    """R-001: Inject NaN into visit dicts — fillna(0) in feature builder must prevent crash."""

    def test_nan_mmse(self, clinical_gru):
        visits = _standard_visits(3)
        visits[1]["mmse"] = float("nan")
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_nan_nwbv(self, clinical_gru):
        visits = _standard_visits(3)
        visits[0]["nwbv"] = float("nan")
        visits[2]["nwbv"] = float("nan")
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_all_nan_visit(self, clinical_gru):
        """Completely empty / NaN visit — must fall back to defaults gracefully."""
        visits = [
            _make_visit(),
            dict(age=float("nan"), educ=float("nan"), ses=float("nan"),
                 mmse=float("nan"), cdr=float("nan"), nwbv=float("nan"),
                 etiv=float("nan"), asf=float("nan")),
            _make_visit(age=76.0),
        ]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_none_values_coerced(self, clinical_gru):
        """None values should not crash the feature builder."""
        visits = [_make_visit()]
        visits[0]["ses"] = None   # type: ignore[assignment]
        # visits_to_feature_tensor uses dict.get() which handles None
        # fillna(0.0) in pandas handles None after conversion
        tensor = visits_to_feature_tensor(visits, None)
        assert tensor.shape == (1, 1, 19)
        assert not torch.any(torch.isnan(tensor))


# ─── R-002: Single-visit sequences ───────────────────────────────────────────

class TestR002SingleVisit:
    """R-002: Model must handle sequence length 1 (no visit history)."""

    def test_single_visit_clinical(self, clinical_gru):
        visits = [_make_visit()]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)
        assert result["n_visits_used"] == 1

    def test_single_visit_high_cdr(self, clinical_gru):
        visits = [_make_visit(mmse=18.0, cdr=2.0, nwbv=0.65)]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_single_visit_bimodal(self, bimodal_model):
        visits = [_make_visit()]
        mri = {"nwbv": 0.75, "etiv": 1500.0, "asf": 1.2, "nwbv_delta": 0.0}
        result = run_bimodal_inference(bimodal_model, visits, mri)
        _validate_result(result)

    def test_sequence_length_preserved_in_tensor(self):
        visits = [_make_visit()]
        tensor = visits_to_feature_tensor(visits, None)
        assert tensor.shape == (1, 1, 19), f"Unexpected tensor shape: {tensor.shape}"


# ─── R-003: Corrupted / zero MRI scalars ─────────────────────────────────────

class TestR003CorruptedMRI:
    """R-003: Zero-filled MRI scalars (simulating missing/failed acquisition)."""

    def test_zero_nwbv(self, bimodal_model):
        visits = _standard_visits(2)
        mri = {"nwbv": 0.0, "etiv": 0.0, "asf": 0.0, "nwbv_delta": 0.0}
        result = run_bimodal_inference(bimodal_model, visits, mri)
        _validate_result(result)

    def test_negative_mri_scalars(self, bimodal_model):
        """After normalization, negative values are valid tensors."""
        visits = _standard_visits(2)
        mri = {"nwbv": -0.1, "etiv": -500.0, "asf": -0.5, "nwbv_delta": -0.05}
        result = run_bimodal_inference(bimodal_model, visits, mri)
        _validate_result(result)

    def test_zero_clinical_features(self, clinical_gru):
        """All-zero clinical visit (e.g., default/missing patient record)."""
        visits = [dict(age=0, educ=0, ses=0, mmse=0, cdr=0,
                       nwbv=0, etiv=0, asf=0, gender="F", hand="R")]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_tensor_not_nan_on_zero_mri(self):
        """Feature tensor must never contain NaN even on zero-filled input."""
        visits = [dict(age=0, educ=0, ses=0, mmse=0, cdr=0,
                       nwbv=0, etiv=0, asf=0)]
        tensor = visits_to_feature_tensor(visits, None)
        assert not torch.any(torch.isnan(tensor)), "NaN detected in zero-MRI tensor"
        assert not torch.any(torch.isinf(tensor)), "Inf detected in zero-MRI tensor"


# ─── R-004: Extreme outlier MRI values ───────────────────────────────────────

class TestR004OutlierMRI:
    """R-004: Extreme out-of-distribution MRI acquisition characteristics."""

    def test_very_large_etiv(self, clinical_gru):
        visits = [_make_visit(etiv=9999.0)]  # ~6.7 sigma above mean
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_very_small_nwbv(self, clinical_gru):
        visits = [_make_visit(nwbv=0.4)]  # far below normal range
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_very_large_age(self, clinical_gru):
        visits = [_make_visit(age=120.0)]  # physiologically impossible
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_extreme_mmse(self, clinical_gru):
        """MMSE=0 (total cognitive loss) and MMSE=30 (perfect)."""
        for mmse_val in [0.0, 30.0]:
            visits = [_make_visit(mmse=mmse_val)]
            result = run_clinical_inference(clinical_gru, visits)
            _validate_result(result)

    def test_no_inf_in_tensor_for_outliers(self):
        """After normalization, outlier values must not produce Inf tensors."""
        visits = [_make_visit(age=200.0, etiv=50000.0, nwbv=5.0)]
        tensor = visits_to_feature_tensor(visits, None)
        assert not torch.any(torch.isinf(tensor)), "Inf detected in outlier tensor"


# ─── R-005: Class imbalance simulation ───────────────────────────────────────

class TestR005ClassImbalance:
    """R-005: Feed only one CDR class — model must not crash, output valid probs."""

    @pytest.mark.parametrize("cdr_val", [0.0, 0.5, 1.0, 2.0])
    def test_single_cdr_class_input(self, clinical_gru, cdr_val):
        """All visits same CDR value — simulates extreme class imbalance in input."""
        visits = [
            _make_visit(cdr=cdr_val, mmse=max(0, 30 - int(cdr_val * 10)))
            for _ in range(3)
        ]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_all_normal_cdr(self, clinical_gru):
        """5 healthy visits — should return a valid prediction."""
        visits = [_make_visit(cdr=0.0, mmse=29, nwbv=0.78) for _ in range(5)]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_all_severe_cdr(self, clinical_gru):
        """5 severe dementia visits."""
        visits = [_make_visit(cdr=2.0, mmse=10, nwbv=0.62) for _ in range(5)]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)


# ─── R-006: Temporal gaps ────────────────────────────────────────────────────

class TestR006TemporalGaps:
    """R-006: Very long sequences and large age jumps between visits."""

    def test_five_visits_long_sequence(self, clinical_gru):
        """Maximum typical OASIS-2 visit count."""
        visits = [_make_visit(age=65.0 + i * 5) for i in range(5)]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)
        assert result["n_visits_used"] == 5

    def test_large_age_jump(self, clinical_gru):
        """Two visits 20 years apart — extreme temporal gap."""
        visits = [
            _make_visit(age=60.0, mmse=29),
            _make_visit(age=80.0, mmse=18, cdr=1.0, nwbv=0.65),
        ]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_visits_same_age(self, clinical_gru):
        """Same-age visits (zero time gap) — edge case."""
        visits = [_make_visit(age=72.0) for _ in range(3)]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_seven_visits_beyond_dataset(self, clinical_gru):
        """More visits than seen during training (OOD sequence length)."""
        visits = [_make_visit(age=60.0 + i * 2, cdr=min(2.0, i * 0.25))
                  for i in range(7)]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)
        assert result["n_visits_used"] == 7


# ─── R-007: Dataset / distribution shift ─────────────────────────────────────

class TestR007DistributionShift:
    """R-007: Scaled features — simulate a different scanner / acquisition protocol."""

    def test_features_scaled_2x(self, clinical_gru):
        """Double all feature values — simulate high-value scanner."""
        visits = [_make_visit(
            age=144.0,       # 2x
            educ=24.0,       # 2x
            mmse=56.0,       # 2x (OOD, but must not crash)
            nwbv=1.5,        # 2x
            etiv=3000.0,     # 2x
        )]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_features_scaled_half(self, clinical_gru):
        """Half all feature values — simulate low-signal acquisition."""
        visits = [_make_visit(
            age=36.0,        # 0.5x
            mmse=14.0,       # 0.5x
            nwbv=0.375,      # 0.5x
            etiv=750.0,      # 0.5x
        )]
        result = run_clinical_inference(clinical_gru, visits)
        _validate_result(result)

    def test_different_gender_distribution(self, clinical_gru):
        """All-male vs all-female sequence."""
        for gender in ["M", "F"]:
            visits = [dict(age=72.0, educ=12.0, ses=2.0, mmse=28.0, cdr=0.0,
                           nwbv=0.75, etiv=1500.0, asf=1.2, gender=gender, hand="R")
                      for _ in range(3)]
            result = run_clinical_inference(clinical_gru, visits)
            _validate_result(result)

    def test_feature_tensor_shape_unchanged_under_shift(self):
        """Tensor shape is ALWAYS (1, T, 19) regardless of feature values."""
        for scale in [0.1, 1.0, 10.0, 100.0]:
            visits = [_make_visit(
                age=72.0 * scale,
                mmse=28.0 * scale,
                nwbv=0.75 * scale,
                etiv=1500.0 * scale,
            ) for _ in range(3)]
            tensor = visits_to_feature_tensor(visits, None)
            assert tensor.shape == (1, 3, 19), f"Shape mismatch at scale={scale}"


# ─── R-008: Random seed / determinism ────────────────────────────────────────

class TestR008Determinism:
    """R-008: Same input must produce identical output on repeated calls."""

    def test_clinical_gru_is_deterministic(self, clinical_gru):
        """With eval() + no_grad(), GRU output must be identical across calls."""
        visits = _standard_visits(3)
        result1 = run_clinical_inference(clinical_gru, visits)
        result2 = run_clinical_inference(clinical_gru, visits)
        result3 = run_clinical_inference(clinical_gru, visits)

        assert result1["predicted_cdr_class"] == result2["predicted_cdr_class"] == result3["predicted_cdr_class"]
        assert result1["predicted_cdr_value"] == result2["predicted_cdr_value"]

        for i in CDR_LABELS:
            label = CDR_LABELS[i]
            p1 = result1["class_probabilities"][label]
            p2 = result2["class_probabilities"][label]
            p3 = result3["class_probabilities"][label]
            assert abs(p1 - p2) < 1e-6, f"Non-deterministic prob for {label}: {p1} vs {p2}"
            assert abs(p1 - p3) < 1e-6, f"Non-deterministic prob for {label}: {p1} vs {p3}"


    def test_bimodal_is_deterministic(self, bimodal_model):
        visits = _standard_visits(2)
        mri = {"nwbv": 0.74, "etiv": 1480.0, "asf": 1.18, "nwbv_delta": -0.01}
        result1 = run_bimodal_inference(bimodal_model, visits, mri)
        result2 = run_bimodal_inference(bimodal_model, visits, mri)
        assert result1["predicted_cdr_class"] == result2["predicted_cdr_class"]

    def test_different_seeds_can_differ(self):
        """Two models with different seeds may produce different outputs (sanity check)."""
        torch.manual_seed(0)
        m1 = TemporalCerebroNet(input_dim=19, gru_hidden_dim=64, num_classes=4)
        m1.eval()

        torch.manual_seed(999)
        m2 = TemporalCerebroNet(input_dim=19, gru_hidden_dim=64, num_classes=4)
        m2.eval()

        visits = _standard_visits(3)
        r1 = run_clinical_inference(m1, visits)
        r2 = run_clinical_inference(m2, visits)

        # Both results must still be valid
        _validate_result(r1)
        _validate_result(r2)
        # (We don't assert they're different — with unlikely random init they could match)


# ─── Additional: Tensor-level robustness ─────────────────────────────────────

class TestTensorRobustness:
    """Low-level tensor sanity checks independent of full inference pipeline."""

    def test_feature_tensor_shape(self):
        for n in range(1, 8):
            tensor = visits_to_feature_tensor(_standard_visits(n), None)
            assert tensor.shape == (1, n, 19), f"Shape mismatch for n={n}: {tensor.shape}"

    def test_feature_tensor_dtype(self):
        tensor = visits_to_feature_tensor(_standard_visits(3), None)
        assert tensor.dtype == torch.float32

    def test_gru_forward_no_crash_on_length_1(self):
        model = TemporalCerebroNet(input_dim=19, gru_hidden_dim=64, num_classes=4)
        model.eval()
        x = torch.zeros(1, 1, 19)
        lengths = torch.tensor([1])
        with torch.no_grad():
            out = model(x, lengths)
        assert out.shape == (1, 4)

    def test_gru_output_shape_multiple_batch(self):
        model = TemporalCerebroNet(input_dim=19, gru_hidden_dim=64, num_classes=4)
        model.eval()
        x = torch.randn(4, 5, 19)   # batch=4, seq=5
        lengths = torch.tensor([5, 3, 2, 1])
        with torch.no_grad():
            out = model(x, lengths)
        assert out.shape == (4, 4), f"Unexpected output shape: {out.shape}"

    def test_softmax_probs_sum_to_one(self, clinical_gru):
        import torch.nn.functional as F
        visits = _standard_visits(3)
        tensor = visits_to_feature_tensor(visits, None)
        lengths = torch.tensor([3])
        with torch.no_grad():
            logits = clinical_gru(tensor, lengths)
            probs = F.softmax(logits, dim=-1).squeeze(0)
        assert abs(probs.sum().item() - 1.0) < 1e-5
