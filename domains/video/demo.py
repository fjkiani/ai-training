"""Gradio demo for the video analysis pipeline.

Upload a video (MP4) -> see detected scenes, keyframes, and CLIP tags.

Run locally:
    python -m domains.video.demo
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .infer import analyze_video

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "outputs"
SAMPLES_DIR = HERE / "samples"


def run_inference(video_path):
    if video_path is None:
        return None, "Please upload a video file."
    try:
        manifest = analyze_video(video_path, OUTPUT_DIR, classify=True)
    except Exception as e:
        return None, f"Error: {e}"

    n_scenes = manifest["n_scenes"]
    n_keyframes = manifest["n_keyframes"]
    classifications = manifest.get("classifications", [])

    # Build a summary figure showing keyframes with their top label
    if classifications:
        n_show = min(len(classifications), 8)
        ncols = 4
        nrows = int(np.ceil(n_show / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = np.atleast_1d(axes).ravel()
        for i in range(n_show):
            c = classifications[i]
            img = Image.open(c["path"])
            axes[i].imshow(img)
            top_label, top_score = c["labels"][0]
            axes[i].set_title(f"{top_label}\n({top_score:.1%})", fontsize=9)
            axes[i].axis("off")
        for i in range(n_show, len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No keyframes extracted", ha="center", va="center")
        ax.axis("off")

    summary = f"Scenes detected: {n_scenes}\nKeyframes extracted: {n_keyframes}\n\n"
    if classifications:
        summary += "Top tags per keyframe:\n"
        for i, c in enumerate(classifications[:10]):
            top_label, top_score = c["labels"][0]
            summary += f"  Scene {i+1}: {top_label} ({top_score:.1%})\n"
    return fig, summary


def _get_examples():
    """Return list of sample video paths, or None if samples don't exist."""
    if not SAMPLES_DIR.exists():
        return None
    samples = sorted(SAMPLES_DIR.glob("*.mp4"))
    return [[str(s)] for s in samples] if samples else None


def build_demo() -> gr.Interface:
    return gr.Interface(
        fn=run_inference,
        inputs=gr.Video(label="Upload a video (MP4)"),
        outputs=[gr.Image(label="Keyframes + Tags"), gr.Textbox(label="Summary")],
        title="Video — Scene Detection + CLIP Tagging",
        description=(
            "Detects scenes using PySceneDetect, extracts keyframes, and tags them "
            "using CLIP zero-shot classification. Upload a short video to see results."
        ),
        examples=_get_examples(),
    )


if __name__ == "__main__":
    build_demo().launch()
