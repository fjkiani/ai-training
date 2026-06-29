"""Medical imaging demo model.

MONAI U-Net for 2D image classification (MedNIST: 6 hand radiograph classes).
The same architecture supports segmentation; here we use a classification head
for a fast, reliable CPU demo on the MedNIST dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# MedNIST classes
MEDNIST_CLASSES = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]
NUM_CLASSES = len(MEDNIST_CLASSES)


class SimpleUNetClassifier(nn.Module):
    """Lightweight U-Net-style encoder with a classification head.

    Designed for small (64x64) grayscale images and CPU training.
    Encoder: 3 downsample blocks; head: global avg pool -> linear.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES, base_features: int = 32):
        super().__init__()
        f = base_features
        self.enc1 = self._conv_block(in_channels, f)
        self.enc2 = self._conv_block(f, f * 2)
        self.enc3 = self._conv_block(f * 2, f * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(f * 4, f * 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(f * 8, num_classes)

    @staticmethod
    def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


def build_model(device: str = "cpu") -> SimpleUNetClassifier:
    """Instantiate the model and move to device."""
    model = SimpleUNetClassifier()
    return model.to(device)


def load_checkpoint(path: str, device: str = "cpu") -> SimpleUNetClassifier:
    """Load a trained checkpoint into a model instance."""
    model = build_model(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
