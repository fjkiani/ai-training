"""Audio preprocessing pipeline.

WAV/MP3 -> feature arrays (MFCC, mel-spectrogram, chroma, spectral contrast, tonnetz).

Pipeline steps:
  1. Load audio (librosa)
  2. Resample to target rate (22050 Hz)
  3. Convert to mono
  4. Extract features (MFCC, mel-spec, chroma, spectral contrast, tonnetz)
  5. Aggregate to fixed-length vectors (mean + std)
  6. Normalize

Built on librosa, inspired by danilodsp/AFX (MIT) and MaxHilsdorf/SLAPP (MIT).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import librosa

from shared import get_logger

log = get_logger("audio.pipeline")

DEFAULT_SR = 22050
DEFAULT_N_MFCC = 40
DEFAULT_N_MELS = 128


def load_audio(path: str | Path, sr: int = DEFAULT_SR, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load an audio file, resampled to sr, mono by default."""
    y, sr = librosa.load(str(path), sr=sr, mono=mono)
    return y, sr


def extract_features(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_mels: int = DEFAULT_N_MELS,
) -> dict:
    """Extract a comprehensive feature set from an audio waveform.

    Returns dict of feature arrays:
      - mfcc: (n_mfcc, frames)
      - mel_spec: (n_mels, frames)
      - chroma: (12, frames)
      - spectral_contrast: (7, frames)
      - tonnetz: (6, frames)
    """
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Chroma (STFT-based)
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    # Tonnetz
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

    return {
        "mfcc": mfcc,
        "mel_spec": mel_spec_db,
        "chroma": chroma,
        "spectral_contrast": spectral_contrast,
        "tonnetz": tonnetz,
    }


def aggregate_features(features: dict) -> np.ndarray:
    """Aggregate time-varying features into a fixed-length vector (mean + std per coefficient)."""
    parts = []
    for key in ["mfcc", "chroma", "spectral_contrast", "tonnetz"]:
        feat = features[key]
        parts.append(np.mean(feat, axis=1))
        parts.append(np.std(feat, axis=1))
    return np.concatenate(parts)


def get_feature_vector(path: str | Path, sr: int = DEFAULT_SR) -> np.ndarray:
    """Full pipeline: load -> extract -> aggregate. Returns a 1D feature vector."""
    y, sr = load_audio(path, sr=sr)
    features = extract_features(y, sr)
    return aggregate_features(features)


def get_mel_spectrogram_image(path: str | Path, sr: int = DEFAULT_SR, n_mels: int = DEFAULT_N_MELS) -> np.ndarray:
    """Return a mel-spectrogram as a normalized (n_mels, frames) array for CNN input or visualization."""
    y, sr = load_audio(path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # normalize to [0,1]
    mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
    if mel_max > mel_min:
        mel_spec_db = (mel_spec_db - mel_min) / (mel_max - mel_min)
    return mel_spec_db.astype(np.float32)


def prepare_dataset(
    audio_dir: str | Path,
    output_dir: str | Path,
    sr: int = DEFAULT_SR,
    seed: int = 42,
) -> dict:
    """Process a folder of audio files organized by class (one subfolder per class).

    Returns a manifest with feature array paths and label encoder.
    """
    import json
    import pickle
    from shared import ensure_dir, split_indices

    audio_dir = Path(audio_dir)
    out = ensure_dir(output_dir)

    class_names = sorted([d.name for d in audio_dir.iterdir() if d.is_dir()])
    label_map = {name: i for i, name in enumerate(class_names)}

    samples = []  # (path, label_idx)
    for name in class_names:
        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            for p in sorted((audio_dir / name).glob(f"*{ext}")):
                samples.append((p, label_map[name]))

    log.info(f"Found {len(samples)} audio files across {len(class_names)} classes: {class_names}")

    train_idx, val_idx, test_idx = split_indices(len(samples), seed=seed)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    all_features = []
    all_labels = []
    for path, label in samples:
        feat = get_feature_vector(path, sr=sr)
        all_features.append(feat)
        all_labels.append(label)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Normalize features (z-score) using train stats
    train_mask = np.zeros(len(samples), dtype=bool)
    train_mask[train_idx] = True
    mean, std = X[train_mask].mean(0), X[train_mask].std(0) + 1e-8
    X = (X - mean) / std

    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    with open(out / "label_encoder.pkl", "wb") as f:
        pickle.dump({"class_names": class_names, "label_map": label_map, "mean": mean, "std": std}, f)

    manifest = {
        "n_samples": len(samples),
        "n_classes": len(class_names),
        "class_names": class_names,
        "feature_dim": X.shape[1],
        "splits": {s: {"count": len(idxs)} for s, idxs in splits.items()},
        "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Dataset prepared at {out}: {X.shape}")
    return manifest
