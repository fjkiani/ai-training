"""Tests for the audio pipeline and model."""
from __future__ import annotations

import numpy as np
from pathlib import Path

from domains.audio.pipeline import extract_features, aggregate_features, load_audio, get_feature_vector
from domains.audio.model import build_model


def test_extract_features_shapes():
    """Feature extraction should return expected shapes."""
    sr = 22050
    y = np.random.randn(sr * 2).astype(np.float32)  # 2 seconds of noise
    feats = extract_features(y, sr)
    assert feats["mfcc"].shape[0] == 40
    assert feats["mel_spec"].shape[0] == 128
    assert feats["chroma"].shape[0] == 12
    assert feats["spectral_contrast"].shape[0] == 7
    assert feats["tonnetz"].shape[0] == 6
    # All should have frames > 0
    for key, val in feats.items():
        assert val.shape[1] > 0, f"{key} has no frames"


def test_aggregate_features_fixed_length():
    """Aggregated features should be a fixed-length 1D vector regardless of input length."""
    sr = 22050
    y1 = np.random.randn(sr * 1).astype(np.float32)
    y2 = np.random.randn(sr * 3).astype(np.float32)
    v1 = aggregate_features(extract_features(y1, sr))
    v2 = aggregate_features(extract_features(y2, sr))
    assert v1.ndim == 1
    assert v1.shape == v2.shape, "Feature vectors should have same length for different durations"
    # Expected dim: (40+12+7+6) * 2 (mean+std) = 130
    assert v1.shape[0] == 130


def test_rf_model_fit_predict():
    """Random Forest should train and predict on synthetic data."""
    X = np.random.randn(50, 130).astype(np.float32)
    y = np.random.randint(0, 5, 50)
    model = build_model(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset(set(range(5)))
