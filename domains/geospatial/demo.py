"""Gradio demo for the geospatial segmentation pipeline.

Upload an image (PNG/JPG representing a satellite tile) -> see land/water mask.

Run locally:
    python -m domains.geospatial.demo
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .infer import predict_tile

HERE = Path(__file__).parent
CKPT = HERE / "models" / "geo_unet.pth"
SAMPLES_DIR = HERE / "samples"


def run_inference(image):
    if image is None:
        return None, "Please upload an image."
    if not CKPT.exists():
        return None, f"No trained model found at {CKPT}. Run training first: python -m domains.geospatial.train"

    img = np.array(Image.open(image).convert("RGB").resize((256, 256))).transpose(2, 0, 1) / 255.0
    mask = predict_tile(img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img.transpose(1, 2, 0))
    axes[0].set_title("Input (256x256)")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    overlay = img.transpose(1, 2, 0).copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.6 + np.array([0.0, 1.0, 0.0]) * 0.4
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay (green=land)")
    axes[2].axis("off")
    plt.tight_layout()

    land_pct = float(mask.mean()) * 100
    summary = f"Land coverage: {land_pct:.1f}%\nWater coverage: {100 - land_pct:.1f}%"
    return fig, summary


def _get_examples():
    """Return list of sample image paths, or None if samples don't exist."""
    if not SAMPLES_DIR.exists():
        return None
    samples = sorted(SAMPLES_DIR.glob("*.png"))
    return [[str(s)] for s in samples] if samples else None


def build_demo() -> gr.Interface:
    return gr.Interface(
        fn=run_inference,
        inputs=gr.Image(type="filepath", label="Upload a satellite tile (PNG/JPG)"),
        outputs=[gr.Image(label="Prediction"), gr.Textbox(label="Summary")],
        title="Geospatial — Land/Water Segmentation",
        description=(
            "A U-Net (ResNet18 encoder) trained on synthetic land/water tiles. "
            "Upload a 256x256 satellite-like image to see the predicted segmentation mask."
        ),
        examples=_get_examples(),
        cache_examples=False,
    )


if __name__ == "__main__":
    build_demo().launch()
