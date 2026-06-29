"""Inference for the video pipeline: process a video end-to-end."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .pipeline import process_video
from .model import DEFAULT_LABELS

HERE = Path(__file__).parent


def analyze_video(
    video_path: str | Path,
    output_dir: str | Path = HERE / "outputs",
    candidate_labels: List[str] | None = None,
    classify: bool = True,
) -> dict:
    """Full video analysis: scenes -> keyframes -> CLIP tags.

    Returns the manifest dict.
    """
    if candidate_labels is None:
        candidate_labels = DEFAULT_LABELS
    return process_video(
        video_path, output_dir,
        candidate_labels=candidate_labels,
        classify=classify,
    )
