"""Video demo model: CLIP zero-shot classifier.

No training required — uses pretrained CLIP weights for zero-shot frame
classification. A small CNN fine-tuning option is included in train.py for
those who want to train on extracted frames.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

DEFAULT_LABELS = [
    "outdoor", "indoor", "person", "vehicle", "landscape",
    "building", "animal", "text", "food", "sky",
]


def load_clip_model(model_name: str = DEFAULT_CLIP_MODEL):
    """Load CLIP model and processor. Returns (model, processor)."""
    from transformers import CLIPModel, CLIPProcessor
    import torch

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def classify_frame(image_path: str | Path, labels: List[str], model=None, processor=None) -> List[tuple]:
    """Classify a single frame. Returns sorted [(label, score), ...]."""
    from PIL import Image
    import torch

    if model is None or processor is None:
        model, processor = load_clip_model()

    image = Image.open(str(image_path)).convert("RGB")
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    ranked = sorted(zip(labels, probs), key=lambda x: -x[1])
    return [(label, float(score)) for label, score in ranked]
