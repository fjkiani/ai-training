"""Audio demo model: Random Forest classifier on aggregated MFCC features.

Fast, interpretable, CPU-friendly. A small CNN on mel-spectrograms is included
as an optional alternative but the RF is the primary demo model.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_model(n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    """Build a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def save_model(model: RandomForestClassifier, path: str | Path) -> None:
    """Pickle the model and label encoder together."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> RandomForestClassifier:
    """Load a pickled model."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_label_encoder(path: str | Path) -> dict:
    """Load the label encoder (class names + normalization stats)."""
    with open(path, "rb") as f:
        return pickle.load(f)
