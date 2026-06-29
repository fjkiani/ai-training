"""Inference for the geospatial segmentation model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import load_checkpoint

HERE = Path(__file__).parent
DEFAULT_CKPT = HERE / "models" / "geo_unet.pth"


def predict_tile(image: np.ndarray, ckpt_path: str | Path = DEFAULT_CKPT, device: str = "cpu") -> np.ndarray:
    """Run segmentation on a (C, H, W) or (H, W) numpy array.

    If the input is already in [0,1] range, it is used directly (matching training).
    Otherwise, per-band percentile normalization is applied.

    Returns a binary mask (H, W) uint8.
    """
    model = load_checkpoint(str(ckpt_path), device, in_channels=image.shape[0] if image.ndim == 3 else 1)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=0)
    img = image.astype(np.float32)

    # If data is outside [0,1], apply percentile normalization (raw GeoTIFF values)
    if img.min() < 0 or img.max() > 1:
        from .pipeline import normalize_bands
        img = normalize_bands(img)
    else:
        # Already normalized — just clip to [0,1]
        img = np.clip(img, 0, 1)

    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    mask = (out.cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
    return mask
