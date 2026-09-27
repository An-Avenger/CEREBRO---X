"""
Brain Health Index (BHI) Module.

Defines the quantitative Brain Health Index (BHI) derived from the 
longitudinal Brain Twin (Z_t) representation.

> [!WARNING]
> RESEARCH PROTOTYPE ONLY.
> This score is a synthetic composite designed for research purposes.
> It has NOT been clinically validated and should not be used for diagnosis.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

class BrainHealthIndex:
    """
    Computes a composite Brain Health Index (0-100 scale, higher is healthier).
    
    The BHI is derived from three main components:
    1. Clinical progression risk (from the trained BimodalCerebroNet logits)
    2. Cognitive state (MMSE relative to age/education norms)
    3. Structural preservation (nWBV relative to healthy baseline)
    """
    
    def __init__(
        self,
        weight_progression: float = 0.50,
        weight_cognitive: float = 0.30,
        weight_structural: float = 0.20,
    ):
        """
        Initialize BHI calculator with component weights.
        Weights should sum to 1.0.
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
        educ: float
    ) -> dict[str, float]:
        """
        Compute the BHI and its individual components.
        
        Args:
            prediction_probs: Dict mapping CDR class string ("0", "1", "2", "3") to probability.
                              Derived from the multimodal model output.
            mmse: Current Mini-Mental State Examination score (0-30).
            nwbv: Normalized Whole Brain Volume (typically 0.65 - 0.85).
            age: Current age in years.
            educ: Years of education.
            
        Returns:
            Dictionary containing the final BHI and sub-component scores.
        """
        
        # 1. Progression Score (0-100)
        # Higher probability of CDR=0 -> higher score
        # Probabilities for CDR 0, 0.5, 1.0, 2.0 (mapped to keys "0", "1", "2", "3")
        p0 = prediction_probs.get("0", 0.0)
        p05 = prediction_probs.get("1", 0.0)
        p1 = prediction_probs.get("2", 0.0)
        p2 = prediction_probs.get("3", 0.0)
        
        # Expected CDR (0 to 2.0)
        expected_cdr = 0.0 * p0 + 0.5 * p05 + 1.0 * p1 + 2.0 * p2
        # Normalize so 0 CDR = 100, 2.0 CDR = 0
        progression_score = max(0.0, 100.0 * (1.0 - (expected_cdr / 2.0)))
        
        # 2. Cognitive Score (0-100)
        # MMSE is 0-30. We apply a slight age/education adjustment for the "ideal" score.
        # This is a heuristic adjustment for research purposes.
        ideal_mmse = 30.0
        if age > 80: ideal_mmse -= 1.0
        if educ < 12: ideal_mmse -= 1.0
        
        # Normalize MMSE against the ideal score.
        cognitive_score = max(0.0, min(100.0, (mmse / ideal_mmse) * 100.0))
        
        # 3. Structural Score (0-100)
        # nWBV typically ranges from 0.65 (severe atrophy) to 0.85 (healthy young).
        # We define a "healthy" baseline that declines slightly with age.
        # Healthy baseline nWBV roughly = 0.82 - (age - 60) * 0.002
        healthy_nwbv = 0.82 - max(0.0, age - 60) * 0.002
        
        # Calculate ratio of actual to expected healthy volume
        structural_ratio = nwbv / healthy_nwbv
        
        # Map ratio to 0-100. 
        # A ratio of 1.0 or higher is 100. A ratio of 0.8 is roughly 0.
        structural_score = max(0.0, min(100.0, (structural_ratio - 0.8) * 500.0))
        
        # Calculate final composite BHI
        bhi = (
            self.w_prog * progression_score + 
            self.w_cog * cognitive_score + 
            self.w_struct * structural_score
        )
        
        return {
            "bhi": round(bhi, 1),
            "progression_score": round(progression_score, 1),
            "cognitive_score": round(cognitive_score, 1),
            "structural_score": round(structural_score, 1),
            "expected_cdr": round(expected_cdr, 3),
        }
