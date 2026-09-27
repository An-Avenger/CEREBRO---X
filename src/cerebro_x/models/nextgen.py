from .deep.bimodal_fusion import BimodalCerebroNet
import torch

class NextGenMultimodalModel:
    """
    Mock wrapper around the standard BimodalCerebroNet that simulates
    the integration of Genetics (APOE4) and Blood Biomarkers (p-tau).
    
    Since the base model was only trained on 8 features from OASIS-2, this wrapper
    adjusts the output probabilities synthetically to demonstrate clinical utility.
    """
    
    def __init__(self, base_fusion_model):
        self.base_model = base_fusion_model
        
    def predict(self, clinical_seq, mri_tensor, apoe4: bool = False, p_tau: float = None):
        """
        Runs the base model, then adjusts the CDR probabilities based on biomarkers.
        """
        # Run base prediction
        base_logits = self.base_model(clinical_seq, mri_tensor)
        probs = torch.softmax(base_logits, dim=1).detach().numpy()[0]
        
        # Simulated Biomarker Adjustment
        # If APOE4 carrier, increase risk of higher CDR by ~15%
        # If p-tau > 21.7 pg/mL, increase risk of higher CDR by ~20%
        
        risk_multiplier = 1.0
        if apoe4:
            risk_multiplier += 0.15
        if p_tau is not None and p_tau > 21.7:
            risk_multiplier += 0.20
            
        if risk_multiplier > 1.0:
            # Shift mass from CDR 0 to CDR 0.5, 1, 2
            shift_amount = probs[0] * (risk_multiplier - 1.0)
            shift_amount = min(shift_amount, probs[0]) # Don't go below 0
            
            probs[0] -= shift_amount
            probs[1] += shift_amount * 0.5
            probs[2] += shift_amount * 0.3
            probs[3] += shift_amount * 0.2
            
            # Re-normalize just in case
            probs = probs / probs.sum()
            
        # Determine class and expected value
        cdr_values = [0.0, 0.5, 1.0, 2.0]
        pred_class = int(probs.argmax())
        pred_val = sum(p * v for p, v in zip(probs, cdr_values))
        
        return {
            "predicted_cdr_class": pred_class,
            "predicted_cdr_value": pred_val,
            "class_probabilities": probs.tolist(),
            "apoe4_adjusted": apoe4,
            "ptau_adjusted": p_tau is not None,
            "risk_multiplier": risk_multiplier
        }
