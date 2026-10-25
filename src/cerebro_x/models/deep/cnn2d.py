"""
2D CNN for MRI Slice Classification — ResNet-18 Transfer Learning.

Accepts single-channel (grayscale) MRI slices and classifies them
into CDR-based cognitive status categories.

Architecture:
    - ResNet-18 backbone (ImageNet pretrained when available)
    - Modified first conv layer: accepts 1-channel input (replicated to 3ch)
    - Custom classifier head for configurable number of classes
    - CPU-friendly design

Usage:
    model = MRI2DCNN(num_classes=3, pretrained=True)
    logits = model(batch)   # batch: (B, 1, 224, 224) → logits: (B, 3)
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_resnet18(pretrained: bool = True) -> nn.Module:
    """Load ResNet-18 with or without pretrained weights."""
    try:
        import torchvision.models as models

        if pretrained:
            try:
                # torchvision >= 0.13 API
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                backbone = models.resnet18(weights=weights)
                logger.info("ResNet-18 loaded with ImageNet pretrained weights.")
            except AttributeError:
                # Older torchvision
                backbone = models.resnet18(pretrained=True)
                logger.info("ResNet-18 loaded with pretrained=True (legacy API).")
        else:
            backbone = models.resnet18(weights=None)
            logger.info("ResNet-18 loaded WITHOUT pretrained weights.")
        return backbone
    except ImportError:
        logger.warning(
            "torchvision not available. Falling back to custom lightweight 2D CNN."
        )
        return None


class LightweightCNN2D(nn.Module):
    """
    Fallback lightweight 2D CNN when torchvision is not available.
    Designed for 224×224 grayscale input.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (1, 224, 224) → (32, 112, 112)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: (32, 112, 112) → (64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: (64, 56, 56) → (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: (128, 28, 28) → (256, 14, 14)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Adaptive pool to fixed size
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MRI2DCNN(nn.Module):
    """
    2D CNN for MRI slice classification.

    Uses ResNet-18 transfer learning when torchvision is available,
    otherwise falls back to a lightweight custom CNN.

    Input:  (B, 1, H, W) — single-channel grayscale MRI slice
    Output: (B, num_classes) — class logits

    Args:
        num_classes: Number of output classes (e.g. 3 for CN/MCI/AD).
        pretrained: Use ImageNet pretrained ResNet-18 weights.
        input_size: Expected input spatial size (default 224).
        dropout: Dropout rate for the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        input_size: int = 224,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self._use_resnet = False

        backbone = _get_resnet18(pretrained) if pretrained else _get_resnet18(False)

        if backbone is not None:
            self._use_resnet = True
            # Modify first conv to accept 1 channel by averaging pretrained weights
            original_conv = backbone.conv1
            new_conv = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )
            if pretrained and original_conv.weight is not None:
                # Average the 3-channel weights into 1 channel
                with torch.no_grad():
                    new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
                    if original_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.copy_(original_conv.bias)
            backbone.conv1 = new_conv

            # Get the feature dimension from the original FC
            feature_dim = backbone.fc.in_features
            # Replace with identity to extract features
            backbone.fc = nn.Identity()

            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
            logger.info(
                "MRI2DCNN: ResNet-18 backbone (features=%d) → %d classes",
                feature_dim,
                num_classes,
            )
        else:
            self._use_resnet = False
            self.backbone = LightweightCNN2D(in_channels=1, embedding_dim=128)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
            logger.info(
                "MRI2DCNN: Lightweight custom backbone → %d classes", num_classes
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 1, H, W) grayscale MRI slice tensor.

        Returns:
            (B, num_classes) logits.
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def forward_multislice(self, slices_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates multiple slices from the same subject scan.

        Args:
            slices_tensor: (K, 1, H, W) or (B, K, 1, H, W)

        Returns:
            aggregated_probs: Mean softmax probability distribution.
            slice_probs: Softmax probabilities per individual slice.
        """
        if slices_tensor.ndim == 4:
            # (K, 1, H, W)
            logits = self.forward(slices_tensor)
            probs = torch.softmax(logits, dim=-1)
            mean_probs = probs.mean(dim=0, keepdim=True)
            return mean_probs, probs
        elif slices_tensor.ndim == 5:
            # (B, K, 1, H, W)
            B, K, C, H, W = slices_tensor.shape
            flat_x = slices_tensor.view(B * K, C, H, W)
            logits = self.forward(flat_x)
            probs = torch.softmax(logits, dim=-1).view(B, K, -1)
            mean_probs = probs.mean(dim=1)
            return mean_probs, probs
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {slices_tensor.shape}")

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before the classifier head."""
        return self.backbone(x)


# Class labels matching CDR-based classification
CDR_CLASS_LABELS_2D = {
    0: "CN (Normal)",
    1: "MCI (Mild Cognitive Impairment)",
    2: "AD (Alzheimer's Disease)",
}

# CDR → 3-class mapping for the 2D pipeline
# CDR 0.0 → CN, CDR 0.5 → MCI, CDR ≥ 1.0 → AD
CDR_TO_3CLASS = {
    0.0: 0,   # CN
    0.5: 1,   # MCI
    1.0: 2,   # AD
    2.0: 2,   # AD
    3.0: 2,   # AD
}
