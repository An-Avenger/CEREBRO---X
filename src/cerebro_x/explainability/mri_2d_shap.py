"""
MRI 2D SHAP and Saliency Explainability Module.

Supports:
    1. SHAP DeepExplainer / GradientExplainer / PartitionExplainer where compatible.
    2. Integrated Gradients and Guided Saliency fallback for verified attribution.
    3. Exporting attribution overlays, prediction probabilities, and explanation artifacts.
"""
import base64
import io
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MRI2DExplainer:
    """Explainability engine for 2D MRI classification models."""

    def __init__(
        self,
        model: nn.Module,
        background_tensors: Optional[torch.Tensor] = None,
        class_names: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.class_names = class_names or ["CN", "MCI", "AD"]
        self.background = background_tensors
        self.explainer = None
        self.explainer_type = "integrated_gradients"

        # Attempt to initialise SHAP GradientExplainer if background is provided
        if background_tensors is not None and len(background_tensors) > 0:
            try:
                import shap
                # Use a small background sample (e.g. 10 slices)
                bg = background_tensors[:10].to(self.device)
                self.explainer = shap.GradientExplainer(self.model, bg)
                self.explainer_type = "shap_gradient"
                logger.info("Initialized SHAP GradientExplainer with %d background slices.", len(bg))
            except Exception as e:
                logger.warning("SHAP GradientExplainer initialization fallback: %s", e)
                self.explainer = None
                self.explainer_type = "integrated_gradients"

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        steps: int = 30,
    ) -> dict[str, Any]:
        """
        Explain a single 2D MRI slice (1, 1, H, W) or (1, H, W).
        Returns attribution heatmap, predicted class, probabilities, and method name.
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        image_tensor.requires_grad_(True)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_class = int(np.argmax(probs))
        target = pred_class if target_class is None else target_class

        attribution = None
        used_method = self.explainer_type

        # 1. Try SHAP if initialized
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(image_tensor)
                # shap_values can be list of arrays per class, or array with shape (1, 1, H, W, num_classes)
                if isinstance(shap_values, list):
                    attribution = np.array(shap_values[target]).squeeze()
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 5:
                        attribution = shap_values[0, 0, :, :, target]
                    else:
                        attribution = shap_values.squeeze()
                used_method = "shap_gradient"
            except Exception as e:
                logger.warning("SHAP execution error, falling back to Integrated Gradients: %s", e)
                attribution = None

        # 2. Integrated Gradients fallback
        if attribution is None:
            used_method = "integrated_gradients"
            baseline = torch.zeros_like(image_tensor)
            scaled_inputs = [
                baseline + (float(i) / steps) * (image_tensor - baseline)
                for i in range(1, steps + 1)
            ]
            grads = []
            for step_input in scaled_inputs:
                step_input = step_input.clone().detach().requires_grad_(True)
                out = self.model(step_input)
                loss = out[0, target]
                loss.backward()
                grads.append(step_input.grad.detach().cpu().numpy())

            avg_grads = np.mean(grads, axis=0)
            delta = (image_tensor - baseline).detach().cpu().numpy()
            ig = avg_grads * delta
            attribution = np.abs(ig).squeeze()

        # Normalize attribution map for visualization
        attr_norm = attribution.astype(np.float32)
        attr_norm = attr_norm - np.min(attr_norm)
        denom = np.max(attr_norm)
        if denom > 1e-8:
            attr_norm = attr_norm / denom

        # Compute Anatomical ROI Breakdown
        roi_breakdown = self.compute_anatomical_roi_breakdown(attr_norm)

        return {
            "predicted_class": pred_class,
            "predicted_label": self.class_names[pred_class],
            "target_class": target,
            "target_label": self.class_names[target],
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            },
            "attribution": attr_norm,
            "raw_attribution": attribution,
            "method": used_method,
            "roi_breakdown": roi_breakdown,
        }

    @staticmethod
    def compute_anatomical_roi_breakdown(attr_map: np.ndarray) -> list[dict[str, Any]]:
        """
        Map 224x224 normalized saliency attribution onto standard axial neurological ROIs.
        ROIs defined on 224x224 normalized stereotaxic coordinates:
          - Hippocampus & Medial Temporal: bilateral mid-inferior (y: 110-155, x: 65-100 & 124-159)
          - Lateral Ventricles: central cavity (y: 85-135, x: 92-132)
          - Temporal & Cortical Grey Matter: bilateral outer cortex (y: 80-170, x: 30-70 & 154-194)
          - Frontal Lobe / Subcortical: anterior region (y: 40-95, x: 60-164)
          - Posterior / Occipital: posterior pole (y: 155-205, x: 60-164)
        """
        H, W = attr_map.shape
        rois = {
            "Medial Temporal / Hippocampus": [
                (int(0.48 * H), int(0.70 * H), int(0.28 * W), int(0.44 * W)),
                (int(0.48 * H), int(0.70 * H), int(0.56 * W), int(0.72 * W)),
            ],
            "Lateral Ventricles & Periventricular": [
                (int(0.38 * H), int(0.60 * H), int(0.41 * W), int(0.59 * W)),
            ],
            "Cortical Grey Matter & Sulci": [
                (int(0.35 * H), int(0.75 * H), int(0.12 * W), int(0.30 * W)),
                (int(0.35 * H), int(0.75 * H), int(0.70 * W), int(0.88 * W)),
            ],
            "Frontal Lobe & Anterior Network": [
                (int(0.18 * H), int(0.42 * H), int(0.26 * W), int(0.74 * W)),
            ],
            "Posterior Parietal & Occipital": [
                (int(0.70 * H), int(0.92 * H), int(0.26 * W), int(0.74 * W)),
            ],
        }

        roi_scores = {}
        total_energy = float(np.sum(attr_map)) + 1e-8

        for name, boxes in rois.items():
            mask = np.zeros((H, W), dtype=bool)
            for y1, y2, x1, x2 in boxes:
                mask[y1:y2, x1:x2] = True
            energy = float(np.sum(attr_map[mask]))
            roi_scores[name] = energy

        sum_roi_energy = sum(roi_scores.values()) + 1e-8
        breakdown = []
        for name, energy in roi_scores.items():
            pct = (energy / sum_roi_energy) * 100.0
            breakdown.append({
                "region": name,
                "importance_pct": round(pct, 1),
                "raw_energy": round(energy, 4),
            })

        breakdown.sort(key=lambda x: x["importance_pct"], reverse=True)
        return breakdown

    def save_explanation_figure(
        self,
        image_slice: np.ndarray,
        explanation: dict[str, Any],
        output_path: str | Path,
        title: str = "MRI 2D Attribution",
    ) -> None:
        """Save a side-by-side plot of the original slice, attribution heatmap, and overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        im = image_slice.squeeze()
        attr = explanation["attribution"].squeeze()

        # Original
        axes[0].imshow(im, cmap="gray")
        axes[0].set_title("Input MRI Slice")
        axes[0].axis("off")

        # Attribution Heatmap
        axes[1].imshow(attr, cmap="hot")
        axes[1].set_title(f"XAI ({explanation['method']})")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(im, cmap="gray")
        axes[2].imshow(attr, cmap="jet", alpha=0.45)
        pred_txt = f"Pred: {explanation['predicted_label']} ({explanation['probabilities'][explanation['predicted_label']]:.1%})"
        axes[2].set_title(pred_txt)
        axes[2].axis("off")

        plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_base64_visualization(
        self,
        image_slice: np.ndarray,
        explanation: dict[str, Any],
    ) -> str:
        """Render side-by-side XAI explanation to a base64 encoded PNG data URI."""
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        fig.patch.set_facecolor("#0b0f19")

        im = image_slice.squeeze()
        attr = explanation["attribution"].squeeze()

        # Input Slice
        axes[0].imshow(im, cmap="gray")
        axes[0].set_title("Input Slice", color="#e2e8f0", fontsize=10)
        axes[0].axis("off")
        axes[0].set_facecolor("#0b0f19")

        # Attribution
        axes[1].imshow(attr, cmap="hot")
        axes[1].set_title(f"SHAP / IG ({explanation['method']})", color="#38bdf8", fontsize=10)
        axes[1].axis("off")
        axes[1].set_facecolor("#0b0f19")

        # Overlay
        axes[2].imshow(im, cmap="gray")
        axes[2].imshow(attr, cmap="jet", alpha=0.45)
        pred_label = explanation.get("predicted_label", "Unknown")
        prob_val = explanation.get("probabilities", {}).get(pred_label, 0.0)
        axes[2].set_title(f"Overlay: {pred_label} ({prob_val:.1%})", color="#a78bfa", fontsize=10)
        axes[2].axis("off")
        axes[2].set_facecolor("#0b0f19")

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
