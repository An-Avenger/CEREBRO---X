"""
Brain Health Index (BHI) Module.

Defines the quantitative Brain Health Index (BHI) derived from the
longitudinal Brain Twin (Z_t) representation.

WARNING: RESEARCH PROTOTYPE ONLY — NOT CLINICALLY VALIDATED.
This score is a synthetic composite designed for research purposes.
It has NOT been validated for clinical diagnosis or patient management.
Do NOT use for medical decision-making.

CDR class probability keys must be the canonical short keys:
    "0"  -> CDR 0.0  (Normal)
    "1"  -> CDR 0.5  (Very Mild)
    "2"  -> CDR 1.0  (Mild)
    "3"  -> CDR 2.0  (Moderate)

These match the class_probabilities dict returned by run_clinical_inference().
"""
from __future__ import annotations

import logging
import numpy as np

from cerebro_x.features.preprocessor import BHI_PROB_KEYS

logger = logging.getLogger(__name__)

# Required canonical probability keys
_REQUIRED_KEYS = BHI_PROB_KEYS  # ("0", "1", "2", "3")


class BrainHealthIndex:
    """
    Computes a composite Brain Health Index (0-100 scale, higher is healthier).

    RESEARCH PROTOTYPE — NOT CLINICALLY VALIDATED.

    The BHI is derived from three main components:
    1. Clinical progression risk (from the trained GRU class probabilities)
    2. Cognitive state (MMSE relative to age/education norms)
    3. Structural preservation (nWBV relative to healthy baseline)

    CDR probability keys must be the canonical short keys "0", "1", "2", "3".
    """

    DISCLAIMER = (
        "Research prototype composite score. NOT clinically validated. "
        "Do not use for medical decision-making."
    )

    def __init__(
        self,
        weight_progression: float = 0.50,
        weight_cognitive: float = 0.30,
        weight_structural: float = 0.20,
    ):
        """
        Initialize BHI calculator with component weights.
        Weights are normalized so they sum to 1.0.
        """
        total = weight_progression + weight_cognitive + weight_structural
        self.w_prog = weight_progression / total
        self.w_cog = weight_cognitive / total
        self.w_struct = weight_structural / total

        logger.debug(
            "BHI initialized with weights: Prog=%.2f, Cog=%.2f, Struct=%.2f",
            self.w_prog, self.w_cog, self.w_struct
        )

    def compute(
        self,
        prediction_probs: dict[str, float],
        mmse: float,
        nwbv: float,
        age: float,
        educ: float,
    ) -> dict:
        """
        Compute the BHI and its individual components.

        Args:
            prediction_probs: Dict mapping canonical CDR class keys
                              ("0", "1", "2", "3") to probability.
                              Keys MUST use the short canonical form,
                              not human labels like "Normal (CDR 0.0)".
            mmse:  Current MMSE score (0-30).
            nwbv:  Normalized Whole Brain Volume (typically 0.65 - 0.85).
            age:   Current age in years.
            educ:  Years of education.

        Returns:
            Dictionary containing the final BHI and sub-component scores.

        Raises:
            ValueError: If any required canonical key is missing from prediction_probs.
        """
        # Validate canonical keys are present
        missing = [k for k in _REQUIRED_KEYS if k not in prediction_probs]
        if missing:
            raise ValueError(
                f"BHI requires canonical CDR probability keys {_REQUIRED_KEYS}. "
                f"Missing: {missing}. "
                f"Got: {list(prediction_probs.keys())}. "
                "Ensure the prediction endpoint returns class_probabilities with "
                "keys '0', '1', '2', '3' (not human-readable labels)."
            )

        total_prob = sum(prediction_probs.values())
        if total_prob < 0.01:
            raise ValueError(
                f"Prediction probabilities sum to {total_prob:.4f}. "
                "This indicates a preprocessing or model error."
            )

        # 1. Progression Score (0-100)
        p0  = prediction_probs["0"]  # CDR 0.0
        p05 = prediction_probs["1"]  # CDR 0.5
        p1  = prediction_probs["2"]  # CDR 1.0
        p2  = prediction_probs["3"]  # CDR 2.0

        expected_cdr = 0.0 * p0 + 0.5 * p05 + 1.0 * p1 + 2.0 * p2
        progression_score = max(0.0, 100.0 * (1.0 - (expected_cdr / 2.0)))

        # 2. Cognitive Score (0-100)
        ideal_mmse = 30.0
        if age > 80:
            ideal_mmse -= 1.0
        if educ < 12:
            ideal_mmse -= 1.0
        cognitive_score = max(0.0, min(100.0, (mmse / ideal_mmse) * 100.0))

        # 3. Structural Score (0-100)
        healthy_nwbv = 0.82 - max(0.0, age - 60) * 0.002
        structural_ratio = nwbv / healthy_nwbv
        structural_score = max(0.0, min(100.0, (structural_ratio - 0.8) * 500.0))

        bhi = (
            self.w_prog * progression_score
            + self.w_cog * cognitive_score
            + self.w_struct * structural_score
        )

        return {
            "bhi": round(bhi, 1),
            "progression_score": round(progression_score, 1),
            "cognitive_score": round(cognitive_score, 1),
            "structural_score": round(structural_score, 1),
            "expected_cdr": round(expected_cdr, 3),
            "disclaimer": self.DISCLAIMER,
        }
