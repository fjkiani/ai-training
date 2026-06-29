"""Inference for the audio classifier."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import load_model, load_label_encoder
from .pipeline import get_feature_vector, get_mel_spectrogram_image

HERE = Path(__file__).parent
DEFAULT_CKPT = HERE / "models" / "audio_rf.pkl"
DEFAULT_LE = HERE / "models" / "label_encoder.pkl"


def predict(audio_path: str | Path, ckpt_path: str | Path = DEFAULT_CKPT, le_path: str | Path = DEFAULT_LE) -> dict:
    """Run audio classification on a single file.

    Returns dict with predicted class, confidence, and top-5 predictions.
    """
    model = load_model(ckpt_path)
    le = load_label_encoder(le_path)
    class_names = le["class_names"]
    mean, std = le["mean"], le["std"]

    feat = get_feature_vector(audio_path)
    feat = (feat - mean) / std
    feat = feat.reshape(1, -1)

    probs = model.predict_proba(feat)[0]
    pred_idx = int(probs.argmax())
    top5_idx = np.argsort(probs)[::-1][:5]

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "top5": [(class_names[i], float(probs[i])) for i in top5_idx],
    }
