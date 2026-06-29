# Video Pipeline

MP4 → scene detection → keyframe extraction → CLIP zero-shot tagging

## Overview

A video preprocessing pipeline built on PySceneDetect, OpenCV, and HuggingFace Transformers (CLIP). The pipeline detects scene boundaries, extracts representative keyframes, deduplicates near-identical frames, and performs zero-shot classification using CLIP. No training required for the demo — CLIP's pretrained weights provide zero-shot tagging.

## Pipeline

```
MP4 → probe (ffmpeg) → scene detection (PySceneDetect ContentDetector)
→ extract keyframes (middle frame per scene) → dedup (pHash, threshold 0.9)
→ CLIP zero-shot classification → scene list JSON + keyframe images
```

## Demo Model

- **Architecture:** CLIP ViT-Base-Patch32 (zero-shot, no training)
- **Task:** Frame content tagging against candidate labels
- **Default labels:** outdoor, indoor, person, vehicle, landscape, building, animal, text, food, sky
- **Optional:** Small CNN fine-tuning on CLIP-tagged frames (`train.py`)

## Quickstart

```bash
# Install dependencies
pip install scenedetect opencv-python-headless transformers torch gradio imagehash

# Process a video
python -c "from domains.video.infer import analyze_video; print(analyze_video('video.mp4'))"

# Launch the Gradio demo
python -m domains.video.demo
```

## Custom Labels

```python
from domains.video.infer import analyze_video

manifest = analyze_video(
    "video.mp4",
    candidate_labels=["ocean", "forest", "city", "desert", "mountain"],
)
```

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | Scene detection, keyframe extraction, dedup, CLIP classification |
| `model.py` | CLIP model loading and frame classification |
| `train.py` | Optional CNN fine-tuning on CLIP-tagged frames |
| `infer.py` | End-to-end video analysis |
| `demo.py` | Gradio web interface |
| `tests/` | Unit tests for scene detection and keyframe extraction |

## Credits

- Scene detection by [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) (BSD-3)
- Zero-shot classification by [HuggingFace Transformers](https://github.com/huggingface/transformers) (Apache-2.0)
- CLIP model by [OpenAI](https://github.com/openai/CLIP) (MIT)

## License

MIT
