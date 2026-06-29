"""Tests for the medical imaging pipeline and model."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from domains.medical.pipeline import hu_window, center_crop_or_pad, preprocess_2d_image
from domains.medical.model import build_model, SimpleUNetClassifier, MEDNIST_CLASSES, NUM_CLASSES


def test_hu_window_range():
    """HU windowing should produce values in [0,1]."""
    arr = np.array([-2048, -1024, 0, 800, 1600, 3000], dtype=np.float32)
    out = hu_window(arr, window=(-1024, 1600))
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6
    assert out.dtype == np.float32


def test_center_crop_pad():
    """Crop large, pad small, identity for exact match."""
    big = np.ones((80, 80, 80), dtype=np.float32)
    out = center_crop_or_pad(big, (64, 64, 64))
    assert out.shape == (64, 64, 64)

    small = np.ones((40, 40, 40), dtype=np.float32)
    out = center_crop_or_pad(small, (64, 64, 64))
    assert out.shape == (64, 64, 64)
    # center should be 1, edges 0
    assert out[32, 32, 32] == 1.0
    assert out[0, 0, 0] == 0.0

    exact = np.ones((64, 64, 64), dtype=np.float32)
    out = center_crop_or_pad(exact, (64, 64, 64))
    assert np.allclose(out, exact)


def test_model_forward_shape():
    """Model should output (batch, num_classes) for (batch, 1, 64, 64) input."""
    model = build_model("cpu")
    model.eval()
    x = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, NUM_CLASSES)
    assert len(MEDNIST_CLASSES) == NUM_CLASSES


def test_preprocess_2d_shape(tmp_path):
    """preprocess_2d_image should return (1, 64, 64) in [0,1]."""
    from PIL import Image
    img = Image.new("L", (128, 128), color=128)
    p = tmp_path / "test.png"
    img.save(p)
    arr = preprocess_2d_image(p, size=64)
    assert arr.shape == (1, 64, 64)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 - 1e-6
    assert arr.max() <= 1.0 + 1e-6
