"""Optional fine-tuning for the video domain.

The primary demo uses CLIP zero-shot (no training needed). This script provides
an optional small CNN that can be fine-tuned on extracted+tagged frames for
those who want a trained model. For the portfolio demo, zero-shot is sufficient.

Usage (optional):
    python -m domains.video.train --frames-dir domains/video/data/keyframes --epochs 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from shared import set_seed, ensure_dir, get_logger

log = get_logger("video.train")

HERE = Path(__file__).parent
MODEL_DIR = ensure_dir(HERE / "models")


class FrameDataset(Dataset):
    """Loads keyframe images with CLIP-assigned pseudo-labels for fine-tuning."""

    def __init__(self, manifest_path: str | Path, size: int = 224):
        self.size = size
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.samples = []
        self.class_names = manifest.get("candidate_labels", [])
        for item in manifest.get("classifications", []):
            if item["labels"]:
                top_label = item["labels"][0][0]
                if top_label in self.class_names:
                    self.samples.append((item["path"], self.class_names.index(top_label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1), label


class SmallCNN(nn.Module):
    """Lightweight CNN for frame classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


def train(manifest_path: str | Path, epochs: int = 10, lr: float = 1e-3, device: str = "cpu"):
    """Optional: fine-tune a small CNN on CLIP-tagged frames."""
    set_seed(42)
    ds = FrameDataset(manifest_path)
    if len(ds) < 10:
        log.warning(f"Only {len(ds)} frames available — skipping fine-tuning (need >= 10). Use zero-shot instead.")
        return None
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    model = SmallCNN(num_classes=len(ds.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        log.info(f"Epoch {epoch}/{epochs} | loss={running_loss/total:.4f} acc={correct/total:.4f}")

    ckpt = MODEL_DIR / "video_cnn.pth"
    torch.save(model.state_dict(), ckpt)
    log.info(f"Saved CNN to {ckpt}")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Optional: fine-tune CNN on CLIP-tagged video frames")
    parser.add_argument("--manifest", type=str, required=True, help="Path to process_video manifest.json")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.manifest, epochs=args.epochs)


if __name__ == "__main__":
    main()
