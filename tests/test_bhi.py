"""
Tests for BrainHealthIndex.

Verifies:
1. BHI receives non-zero probabilities using correct canonical keys.
2. BHI raises ValueError if required keys are missing.
3. BHI raises ValueError if human-readable label keys are passed.
4. BHI output is in valid range.
5. BHI includes research disclaimer.
"""
from __future__ import annotations

import pytest

from cerebro_x.brain_twin.bhi import BrainHealthIndex


# ── BHI correctness ───────────────────────────────────────────────────────────

def test_bhi_with_valid_canonical_keys():
    """BHI computes correctly with canonical string keys '0','1','2','3'."""
    bhi = BrainHealthIndex()
    probs = {"0": 0.7, "1": 0.2, "2": 0.08, "3": 0.02}

    result = bhi.compute(
        prediction_probs=probs,
        mmse=27.0,
        nwbv=0.76,
        age=74.0,
        educ=16.0,
    )

    assert "bhi" in result
    assert 0.0 <= result["bhi"] <= 100.0
    assert "progression_score" in result
    assert "cognitive_score" in result
    assert "structural_score" in result
    assert "expected_cdr" in result
    assert "disclaimer" in result
    assert "NOT clinically validated" in result["disclaimer"]


def test_bhi_expected_cdr_computation():
    """Expected CDR should be weighted sum of CDR values."""
    bhi = BrainHealthIndex()
    probs = {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0}

    result = bhi.compute(probs, mmse=28.0, nwbv=0.78, age=70.0, educ=16.0)
    assert result["expected_cdr"] == pytest.approx(0.0)
    # Full CDR-0 probability → high progression score
    assert result["progression_score"] == pytest.approx(100.0)


def test_bhi_raises_on_missing_required_keys():
    """BHI raises ValueError if any canonical key is missing."""
    bhi = BrainHealthIndex()
    # Missing key "3"
    bad_probs = {"0": 0.7, "1": 0.2, "2": 0.1}

    with pytest.raises(ValueError, match="Missing"):
        bhi.compute(bad_probs, mmse=27.0, nwbv=0.76, age=74.0, educ=16.0)


def test_bhi_raises_on_human_readable_label_keys():
    """
    BHI must reject human-readable label keys like 'Normal (CDR 0.0)'.
    This was the original bug — old inference.py keyed by label strings.
    """
    bhi = BrainHealthIndex()
    bad_probs = {
        "Normal (CDR 0.0)": 0.7,
        "Very Mild Dementia (CDR 0.5)": 0.2,
        "Mild Dementia (CDR 1.0)": 0.08,
        "Moderate Dementia (CDR 2.0)": 0.02,
    }

    with pytest.raises(ValueError, match="Missing"):
        bhi.compute(bad_probs, mmse=27.0, nwbv=0.76, age=74.0, educ=16.0)


def test_bhi_raises_on_zero_total_probability():
    """BHI raises ValueError if probabilities sum near zero."""
    bhi = BrainHealthIndex()
    zero_probs = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0}

    with pytest.raises(ValueError, match="sum to"):
        bhi.compute(zero_probs, mmse=27.0, nwbv=0.76, age=74.0, educ=16.0)


def test_bhi_output_range_all_cases():
    """BHI output must always be between 0 and 100."""
    bhi = BrainHealthIndex()

    worst_case = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0}
    result_worst = bhi.compute(worst_case, mmse=0.0, nwbv=0.60, age=95.0, educ=8.0)
    assert 0.0 <= result_worst["bhi"] <= 100.0

    best_case = {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0}
    result_best = bhi.compute(best_case, mmse=30.0, nwbv=0.85, age=60.0, educ=20.0)
    assert 0.0 <= result_best["bhi"] <= 100.0
    assert result_best["bhi"] > result_worst["bhi"]
