"""Shared visualization helpers for ai-training pipelines."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np


def save_image_grid(
    images: list[np.ndarray],
    titles: Optional[list[str]] = None,
    ncols: int = 4,
    cmap: str = "gray",
    out_path: str | Path = "grid.png",
) -> Path:
    """Save a grid of images to out_path (PNG). Returns the path."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, i in zip(axes, images, range(n)):
        ax.imshow(np.squeeze(img), cmap=cmap)
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def save_overlay(
    base: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple = (1, 0, 0),
    out_path: str | Path = "overlay.png",
) -> Path:
    """Overlay a binary mask on a base image and save PNG."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.squeeze(base), cmap="gray")
    if np.any(mask):
        m = np.zeros((*np.squeeze(mask).shape, 4))
        m[..., 0] = color[0]
        m[..., 1] = color[1]
        m[..., 2] = color[2]
        m[..., 3] = (np.squeeze(mask) > 0).astype(float) * alpha
        ax.imshow(m)
    ax.axis("off")
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
