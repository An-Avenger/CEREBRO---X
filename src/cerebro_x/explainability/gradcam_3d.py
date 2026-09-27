"""
Real 3D Grad-CAM for Cerebro-X MRI Explainability.

Generates genuine Grad-CAM heatmaps from the Lightweight3DCNN
by capturing layer activations and gradients via PyTorch hooks.

Status:
  - GradCAM3D class       : IMPLEMENTED (real hook-based Grad-CAM)
  - 2D slice visualization: IMPLEMENTED (axial / sagittal / coronal)
  - CNN3D checkpoint       : NOT AVAILABLE (no trained model yet)
  - Heatmap output         : Returns base64-encoded PNG slices when model is loaded

Usage:
    from cerebro_x.explainability.gradcam_3d import run_gradcam_3d_cnn

    result = run_gradcam_3d_cnn(
        model=cnn3d_model,          # Lightweight3DCNN instance
        preprocessed=tensor_4d,    # (1, D, H, W)
        target_class=1,             # None = use model's top prediction
    )
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not available — slice visualization disabled.")


@dataclass
class GradCAMResult:
    """Result of a 3D Grad-CAM computation."""

    success: bool
    target_class: Optional[int] = None
    target_class_probability: Optional[float] = None
    heatmap_3d: Optional[np.ndarray] = None     # (D, H, W) float32, range [0, 1]
    slices_base64: Optional[dict] = None          # {"axial": str, "sagittal": str, "coronal": str}
    embedding: Optional[list[float]] = None       # 64-dim CNN embedding
    error_code: str = ""
    error_message: str = ""
    method: str = "3D-GradCAM"
    layer_name: str = ""


class GradCAM3DCNN:
    """
    Real 3D Grad-CAM for Lightweight3DCNN.

    Captures:
      - Forward activations from the last Conv3d block (before AdaptiveAvgPool3d)
      - Backward gradients from the same block
      - Computes weighted combination to produce a 3D heatmap

    Target layer: features[-3] — the last Conv3d in the feature extractor
    (the layer at index 11 = Conv3d(32, 64) in Lightweight3DCNN).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []

        # Find the last Conv3d layer in the features sequential
        self._target_layer, self._layer_name = self._find_target_layer(model)
        if self._target_layer is not None:
            self._register_hooks()

    def _find_target_layer(self, model: nn.Module):
        """Find the last Conv3d layer — this is where Grad-CAM captures gradients."""
        last_conv = None
        last_name = ""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv3d):
                last_conv = module
                last_name = name
        return last_conv, last_name

    def _register_hooks(self):
        def _fwd(module, inp, out):
            self._activations = out.detach().clone()

        def _bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach().clone()

        h1 = self._target_layer.register_forward_hook(_fwd)
        h2 = self._target_layer.register_full_backward_hook(_bwd)
        self._hooks = [h1, h2]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __del__(self):
        self.remove_hooks()

    def _get_embedding(self, x: torch.Tensor) -> list[float]:
        """Extract 64-dim embedding without gradient tracking."""
        with torch.no_grad():
            emb = self.model(x)
        return emb.squeeze(0).cpu().numpy().tolist()

    def generate(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> GradCAMResult:
        """
        Run Grad-CAM on a preprocessed MRI tensor.

        Args:
            x:            Shape (1, 1, D, H, W) — batched, channel-first.
            target_class: Which CDR class to explain (0–3).
                          None = use model's top prediction.

        Returns:
            GradCAMResult with heatmap_3d, slices_base64, and embedding.
        """
        if self._target_layer is None:
            return GradCAMResult(
                success=False,
                error_code="NO_CONV_LAYER",
                error_message="No Conv3d layer found in model — cannot run Grad-CAM.",
            )

        if x.dim() == 4:  # (1, D, H, W) → (1, 1, D, H, W)
            x = x.unsqueeze(0)

        x = x.float()
        input_shape = x.shape[2:]   # (D, H, W)

        # ── Forward pass (with gradient enabled) ──────────────────────────────
        self.model.zero_grad()
        x.requires_grad_(False)

        # Enable grad for backward
        with torch.enable_grad():
            x_grad = x.clone().requires_grad_(True)
            logits = self.model(x_grad)
            probs = F.softmax(logits, dim=-1).squeeze(0)

            if target_class is None:
                target_class = int(probs.argmax().item())

            target_prob = float(probs[target_class].item())

            # ── Backward ──────────────────────────────────────────────────────
            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

        if self._gradients is None or self._activations is None:
            return GradCAMResult(
                success=False,
                error_code="HOOKS_FAILED",
                error_message="Gradient/activation hooks did not fire. Check model architecture.",
            )

        # ── Compute Grad-CAM weights ───────────────────────────────────────────
        # weights: (C,) — average gradient per channel across spatial dims
        weights = self._gradients.mean(dim=(2, 3, 4))   # (1, C)
        weights = weights.squeeze(0)                      # (C,)

        activations = self._activations.squeeze(0)       # (C, D', H', W')

        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only positive influence matters
        cam = F.relu(cam)

        # ── Upsample to original input size ───────────────────────────────────
        cam_5d = cam.unsqueeze(0).unsqueeze(0)   # (1, 1, D', H', W')
        cam_up = F.interpolate(
            cam_5d, size=input_shape, mode="trilinear", align_corners=False
        )
        cam_np = cam_up.squeeze().cpu().numpy()   # (D, H, W)

        # Normalize to [0, 1]
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # ── Extract embedding ──────────────────────────────────────────────────
        try:
            with torch.no_grad():
                embedding = self.model(x).squeeze(0).cpu().numpy().tolist()
        except Exception:
            embedding = None

        # ── Generate 2D slice visualizations ──────────────────────────────────
        slices_b64 = _generate_slice_pngs(
            mri=x.squeeze().cpu().numpy(),   # (D, H, W)
            heatmap=cam_np,
        )

        return GradCAMResult(
            success=True,
            target_class=target_class,
            target_class_probability=target_prob,
            heatmap_3d=cam_np.astype(np.float32),
            slices_base64=slices_b64,
            embedding=embedding,
            layer_name=self._layer_name,
        )


def _generate_slice_pngs(
    mri: np.ndarray,
    heatmap: np.ndarray,
) -> Optional[dict]:
    """
    Generate base64-encoded PNG images overlaying Grad-CAM on MRI slices.

    Returns dict with keys: "axial", "sagittal", "coronal".
    Each value is a base64 string (data:image/png;base64,...).
    """
    if not _MPL_AVAILABLE:
        return None

    D, H, W = mri.shape
    results = {}

    slices = {
        "axial":    (mri[D // 2, :, :],   heatmap[D // 2, :, :]),
        "sagittal": (mri[:, :, W // 2],   heatmap[:, :, W // 2]),
        "coronal":  (mri[:, H // 2, :],   heatmap[:, H // 2, :]),
    }

    for plane, (mri_slice, cam_slice) in slices.items():
        try:
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
            fig.patch.set_facecolor("#0f0f14")

            # Normalize MRI slice for display
            mri_min, mri_max = mri_slice.min(), mri_slice.max()
            mri_disp = (mri_slice - mri_min) / (mri_max - mri_min + 1e-8)

            titles = ["MRI", "Grad-CAM", "Overlay"]
            imgs = [
                (mri_disp, "gray", 1.0),
                (cam_slice, "hot", 1.0),
                (mri_disp, "gray", 1.0),
            ]

            for ax, (img, cmap, alpha) in zip(axes, imgs):
                ax.imshow(img.T, cmap=cmap, origin="lower", aspect="equal")
                ax.set_facecolor("#0f0f14")
                ax.axis("off")

            # Overlay on third panel
            axes[2].imshow(cam_slice.T, cmap="hot", origin="lower",
                           alpha=0.55, aspect="equal")

            for ax, title in zip(axes, titles):
                ax.set_title(title, color="white", fontsize=9, pad=4)

            fig.suptitle(f"{plane.capitalize()} slice — Grad-CAM overlay",
                         color="#aaa", fontsize=9, y=1.02)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)

            b64 = base64.b64encode(buf.read()).decode("utf-8")
            results[plane] = f"data:image/png;base64,{b64}"

        except Exception as e:
            logger.warning("Could not generate %s slice: %s", plane, e)
            results[plane] = None

    return results


def run_gradcam_3d_cnn(
    model: nn.Module,
    preprocessed: torch.Tensor,
    target_class: Optional[int] = None,
) -> GradCAMResult:
    """
    Convenience function to run Grad-CAM on a Lightweight3DCNN.

    Args:
        model:        Loaded Lightweight3DCNN instance (eval mode).
        preprocessed: Tensor (1, D, H, W) from the MRI preprocessing pipeline.
        target_class: CDR class to explain (0–3). None = top prediction.

    Returns:
        GradCAMResult.
    """
    cam = GradCAM3DCNN(model)
    result = cam.generate(preprocessed, target_class=target_class)
    cam.remove_hooks()
    return result
