"""Geospatial demo model: U-Net for binary segmentation.

Uses segmentation-models-pytorch with a ResNet18 encoder for CPU-friendly
land/water (or urban/rural) segmentation on 256x256 tiles.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from segmentation_models_pytorch import Unet

DEFAULT_ENCODER = "resnet18"
DEFAULT_ENCODER_WEIGHTS = None  # no pretrained on CPU demo; avoids download


def build_model(
    in_channels: int = 3,
    device: str = "cpu",
    encoder: str = DEFAULT_ENCODER,
) -> nn.Module:
    """Build a U-Net segmentation model."""
    model = Unet(
        encoder_name=encoder,
        encoder_weights=DEFAULT_ENCODER_WEIGHTS,
        in_channels=in_channels,
        classes=1,
        activation="sigmoid",
    )
    return model.to(device)


def load_checkpoint(path: str, device: str = "cpu", in_channels: int = 3) -> nn.Module:
    """Load a trained checkpoint."""
    model = build_model(in_channels=in_channels, device=device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
