"""Tests for the geospatial pipeline and model."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from domains.geospatial.pipeline import normalize_bands, tile_raster
from domains.geospatial.model import build_model
from domains.geospatial.train import generate_synthetic_dataset, TileDataset


def test_normalize_bands_range():
    arr = np.random.rand(3, 64, 64).astype(np.float32) * 100
    out = normalize_bands(arr)
    assert out.shape == (3, 64, 64)
    assert out.dtype == np.float32
    for i in range(3):
        assert out[i].min() >= 0.0 - 1e-6
        assert out[i].max() <= 1.0 + 1e-6


def test_model_forward_shape():
    model = build_model(in_channels=3, device="cpu")
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid output


def test_synthetic_dataset(tmp_path):
    """Generate a tiny synthetic dataset and verify it loads."""
    out = generate_synthetic_dataset(tmp_path / "prepared", n_tiles=20, tile_size=64)
    assert (out / "manifest.json").exists()
    ds = TileDataset(out, "train")
    img, mask = ds[0]
    assert img.shape[0] == 3  # 3 bands
    assert mask.shape == (1, 64, 64)
    assert mask.dtype == torch.float32
