"""Shared utilities for ai-training pipelines.

Common I/O, logging, dataset splitting, and reproducibility helpers used across
all four domains (medical, geospatial, audio, video).
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_indices(n: int, ratios=(0.7, 0.15, 0.15), seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """Split n indices into train/val/test lists with the given ratios.

    Returns (train_idx, val_idx, test_idx).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"ratios must sum to 1, got {sum(ratios)}"
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def list_files(directory: str | Path, exts: List[str]) -> List[Path]:
    """List files in directory matching any of the given extensions (case-insensitive)."""
    exts_lower = {e.lower() for e in exts}
    return sorted(
        p for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in exts_lower
    )
