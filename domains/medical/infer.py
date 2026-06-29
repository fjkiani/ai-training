"""Inference for the medical imaging demo model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import load_checkpoint, MEDNIST_CLASSES
from .pipeline import preprocess_2d_image

HERE = Path(__file__).parent
DEFAULT_CKPT = HERE / "models" / "medical_unet.pth"


def predict(image_path: str | Path, ckpt_path: str | Path = DEFAULT_CKPT, device: str = "cpu") -> dict:
    """Run inference on a single 2D medical image.

    Returns dict with predicted class, confidence, and per-class probabilities.
    """
    model = load_checkpoint(str(ckpt_path), device)
    arr = preprocess_2d_image(image_path, size=64)  # (1, 64, 64) in [0,1]
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 1, 64, 64)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    return {
        "predicted_class": MEDNIST_CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {MEDNIST_CLASSES[i]: float(p) for i, p in enumerate(probs)},
    }
