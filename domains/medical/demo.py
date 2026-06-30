"""Gradio demo for the medical imaging pipeline.

Upload a medical image (PNG/JPG of a radiograph) -> see predicted class + confidence.

Run locally:
    python -m domains.medical.demo

Deploy to HuggingFace Spaces:
    See demos/huggingface_spaces.md
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .infer import predict
from .model import MEDNIST_CLASSES
from .pipeline import preprocess_2d_image

HERE = Path(__file__).parent
CKPT = HERE / "models" / "medical_unet.pth"
SAMPLES_DIR = HERE / "samples"


def run_inference(image):
    """Gradio callback: image -> prediction plot + JSON."""
    if image is None:
        return None, "Please upload an image."
    if not CKPT.exists():
        return None, f"No trained model found at {CKPT}. Run training first: python -m domains.medical.train"

    result = predict(image)
    arr = preprocess_2d_image(image, size=64)[0]

    # Build a figure: input image + bar chart of probabilities
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("Input (64x64)")
    axes[0].axis("off")
    probs = [result["probabilities"][c] for c in MEDNIST_CLASSES]
    colors = ["#FF9400" if c == result["predicted_class"] else "#0279EE" for c in MEDNIST_CLASSES]
    axes[1].barh(MEDNIST_CLASSES, probs, color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title(f"Prediction: {result['predicted_class']} ({result['confidence']:.1%})")
    plt.tight_layout()

    import json
    summary = json.dumps(result, indent=2)
    return fig, summary


def _get_examples():
    """Return list of sample image paths, or None if samples don't exist."""
    if not SAMPLES_DIR.exists():
        return None
    samples = sorted(SAMPLES_DIR.glob("*.jpeg"))
    return [[str(s)] for s in samples] if samples else None


def build_demo() -> gr.Interface:
    return gr.Interface(
        fn=run_inference,
        inputs=gr.Image(type="filepath", label="Upload a radiograph (PNG/JPG)"),
        outputs=[gr.Image(label="Prediction"), gr.Textbox(label="Details")],
        title="Medical Imaging — MedNIST Classifier",
        description=(
            "A MONAI U-Net classifier trained on MedNIST (6 radiograph classes: "
            "AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT). "
            "Upload a radiograph to see the predicted class and confidence."
        ),
        examples=_get_examples(),
        cache_examples=False,
    )


if __name__ == "__main__":
    build_demo().launch()
