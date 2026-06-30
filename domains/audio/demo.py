"""Gradio demo for the audio classification pipeline.

Upload an audio file (WAV/MP3) -> see waveform, mel-spectrogram, predicted class.

Run locally:
    python -m domains.audio.demo
"""
from __future__ import annotations

import io
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from PIL import Image

from .infer import predict
from .pipeline import load_audio, get_mel_spectrogram_image

HERE = Path(__file__).parent
CKPT = HERE / "models" / "audio_rf.pkl"
LE = HERE / "models" / "label_encoder.pkl"
SAMPLES_DIR = HERE / "samples"


def _fig_to_pil(fig) -> Image.Image:
    """Render a matplotlib Figure to a PIL Image for Gradio v6 compatibility."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def run_inference(audio_path):
    if audio_path is None:
        return None, None, "Please upload an audio file."
    if not CKPT.exists():
        return None, None, f"No trained model found at {CKPT}. Run training first: python -m domains.audio.train --download-esc50"

    result = predict(audio_path)
    y, sr = load_audio(audio_path)

    # Waveform
    fig1, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("")
    plt.tight_layout()

    # Mel-spectrogram
    mel = get_mel_spectrogram_image(audio_path)
    fig2, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title("Mel-Spectrogram")
    fig2.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    summary = f"Predicted: {result['predicted_class']} ({result['confidence']:.1%})\n\nTop-5:\n"
    for name, prob in result["top5"]:
        summary += f"  {name}: {prob:.1%}\n"
    return _fig_to_pil(fig1), _fig_to_pil(fig2), summary


def _get_examples():
    """Return list of sample audio paths, or None if samples don't exist."""
    if not SAMPLES_DIR.exists():
        return None
    samples = sorted(SAMPLES_DIR.glob("*.wav"))
    return [[str(s)] for s in samples] if samples else None


def build_demo() -> gr.Interface:
    return gr.Interface(
        fn=run_inference,
        inputs=gr.Audio(type="filepath", label="Upload an audio file (WAV/MP3)"),
        outputs=[
            gr.Image(label="Waveform"),
            gr.Image(label="Mel-Spectrogram"),
            gr.Textbox(label="Prediction"),
        ],
        title="Audio — Environmental Sound Classification",
        description=(
            "A Random Forest classifier trained on ESC-50 (50 environmental sound classes). "
            "Upload an audio file to see the waveform, mel-spectrogram, and predicted class."
        ),
        examples=_get_examples(),
        cache_examples=False,
    )


if __name__ == "__main__":
    build_demo().launch()
