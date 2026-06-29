"""Video preprocessing pipeline.

MP4 -> scene-split keyframes + CLIP zero-shot tags.

Pipeline steps:
  1. Probe video (ffmpeg)
  2. Scene detection (PySceneDetect ContentDetector)
  3. Extract keyframes per scene
  4. Deduplicate near-identical frames (pHash)
  5. CLIP zero-shot classification of keyframes

Built on PySceneDetect + OpenCV + HuggingFace Transformers (CLIP).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from scenedetect import detect, ContentDetector

from shared import get_logger

log = get_logger("video.pipeline")


def detect_scenes(video_path: str | Path, threshold: float = 27.0, min_scene_len: int = 15) -> List[tuple]:
    """Detect scene boundaries in a video.

    Returns a list of (start_time, end_time) tuples in seconds.
    """
    video_path = str(video_path)
    detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    scenes = detect(video_path, detector, show_progress=False)
    log.info(f"Detected {len(scenes)} scenes in {video_path}")
    return [(s.seconds, e.seconds) for s, e in scenes]


def extract_keyframes(video_path: str | Path, scenes: List[tuple], output_dir: str | Path) -> List[Path]:
    """Extract one keyframe (middle frame) per scene.

    Saves frames as PNG and returns list of paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_paths = []

    for i, (start, end) in enumerate(scenes):
        mid_time = (start + end) / 2
        frame_num = int(mid_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            out_path = output_dir / f"scene_{i:03d}_t{mid_time:.1f}s.png"
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(out_path)

    cap.release()
    log.info(f"Extracted {len(frame_paths)} keyframes to {output_dir}")
    return frame_paths


def dedup_keyframes(frame_paths: List[Path], threshold: float = 0.9) -> List[Path]:
    """Remove near-duplicate frames using perceptual hashing.

    threshold: similarity above which frames are considered duplicates (0-1).
    Returns filtered list of paths.
    """
    from PIL import Image
    import imagehash

    if len(frame_paths) <= 1:
        return frame_paths

    hashes = []
    kept = []
    for p in frame_paths:
        img = Image.open(p)
        h = imagehash.phash(img)
        is_dup = False
        for existing in hashes:
            similarity = 1 - (h - existing) / 64.0
            if similarity > threshold:
                is_dup = True
                break
        if not is_dup:
            hashes.append(h)
            kept.append(p)

    log.info(f"Dedup: {len(frame_paths)} -> {len(kept)} frames")
    return kept


def classify_frames_clip(
    frame_paths: List[Path],
    candidate_labels: List[str],
    model_name: str = "openai/clip-vit-base-patch32",
) -> List[dict]:
    """Zero-shot CLIP classification of frames against candidate labels.

    Returns list of {path, labels: [(label, score), ...]} dicts.
    """
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import torch

    log.info(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    results = []
    with torch.no_grad():
        for p in frame_paths:
            image = Image.open(p).convert("RGB")
            inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            ranked = sorted(zip(candidate_labels, probs), key=lambda x: -x[1])
            results.append({
                "path": str(p),
                "labels": [(label, float(score)) for label, score in ranked[:5]],
            })

    log.info(f"Classified {len(results)} frames")
    return results


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    candidate_labels: Optional[List[str]] = None,
    scene_threshold: float = 27.0,
    dedup: bool = True,
    classify: bool = True,
) -> dict:
    """Full pipeline: detect scenes -> extract keyframes -> dedup -> classify.

    Returns a manifest dict with scene info and classification results.
    """
    if candidate_labels is None:
        candidate_labels = [
            "outdoor", "indoor", "person", "vehicle", "landscape",
            "building", "animal", "text", "food", "sky",
        ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "keyframes"

    scenes = detect_scenes(video_path, threshold=scene_threshold)
    frame_paths = extract_keyframes(video_path, scenes, frames_dir)
    if dedup:
        frame_paths = dedup_keyframes(frame_paths)

    classifications = []
    if classify and frame_paths:
        classifications = classify_frames_clip(frame_paths, candidate_labels)

    manifest = {
        "video": str(video_path),
        "n_scenes": len(scenes),
        "scenes": [{"start": s, "end": e} for s, e in scenes],
        "n_keyframes": len(frame_paths),
        "keyframes": [str(p) for p in frame_paths],
        "classifications": classifications,
        "candidate_labels": candidate_labels,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Video processed: {len(scenes)} scenes, {len(frame_paths)} keyframes -> {output_dir}")
    return manifest
