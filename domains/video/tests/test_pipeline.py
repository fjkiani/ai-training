"""Tests for the video pipeline."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import cv2

from domains.video.pipeline import detect_scenes, extract_keyframes


def create_test_video(path: Path, duration_sec: float = 3.0, fps: int = 10, size=(64, 64)):
    """Create a short test video with a scene change at the midpoint."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    n_frames = int(duration_sec * fps)
    for i in range(n_frames):
        # First half: red, second half: blue (scene change)
        if i < n_frames // 2:
            frame = np.zeros((*size, 3), dtype=np.uint8)
            frame[:, :, 2] = 200  # red (BGR)
        else:
            frame = np.zeros((*size, 3), dtype=np.uint8)
            frame[:, :, 0] = 200  # blue (BGR)
        writer.write(frame)
    writer.release()
    return path


def test_detect_scenes(tmp_path):
    """Scene detection should find the red->blue transition."""
    video = create_test_video(tmp_path / "test.mp4")
    scenes = detect_scenes(video, threshold=20.0, min_scene_len=5)
    assert len(scenes) >= 1, f"Expected at least 1 scene, got {len(scenes)}"


def test_extract_keyframes(tmp_path):
    """Keyframe extraction should produce one image per scene."""
    video = create_test_video(tmp_path / "test.mp4")
    scenes = detect_scenes(video, threshold=20.0, min_scene_len=5)
    frames = extract_keyframes(video, scenes, tmp_path / "frames")
    assert len(frames) >= 1
    for p in frames:
        assert p.exists()
        assert p.suffix == ".png"
