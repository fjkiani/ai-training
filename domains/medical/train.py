"""Train the medical imaging demo model on MONAI MedNIST.

MedNIST: ~47,164 hand radiographs across 6 classes (AbdomenCT, BreastMLO, CXR,
ChestCT, Hand, HeadCT). Auto-downloads via MONAI. For a fast CPU demo we train
on a 2,000-image subset for 5 epochs.

Usage:
    python -m domains.medical.train --epochs 5 --subset 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

from monai.apps import download_and_extract

from shared import set_seed, ensure_dir, get_logger, split_indices
from .model import build_model, MEDNIST_CLASSES, NUM_CLASSES

log = get_logger("medical.train")

HERE = Path(__file__).parent
DATA_ROOT = ensure_dir(HERE / "data")
MEDNIST_DIR = DATA_ROOT / "MedNIST"
MODEL_DIR = ensure_dir(HERE / "models")
DEFAULT_CKPT = MODEL_DIR / "medical_unet.pth"
IMG_SIZE = 64


def ensure_mednist():
    """Download and extract MedNIST if not present. Returns the class directories."""
    if not MEDNIST_DIR.exists() or not any(MEDNIST_DIR.iterdir()):
        log.info("Downloading MedNIST (~70MB)...")
        download_and_extract(
            str(DATA_ROOT),
            "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz",
            "MedNIST.tar.gz",
            "0bc7306e7427e00ad1c5526a6677552d",
        )
    class_dirs = sorted([d for d in MEDNIST_DIR.iterdir() if d.is_dir()])
    log.info(f"MedNIST classes: {[d.name for d in class_dirs]}")
    return class_dirs


class MedNISTImageDataset(Dataset):
    """Loads MedNIST images as tensors (1, 64, 64) in [0,1] with integer labels."""

    def __init__(self, class_dirs, size: int = IMG_SIZE):
        self.size = size
        self.samples = []  # (path, label_idx)
        self.class_names = [d.name for d in class_dirs]
        for idx, d in enumerate(class_dirs):
            for p in sorted(d.glob("*.jpeg")):
                self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("L").resize((self.size, self.size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0), label


def get_dataloaders(subset_size: int = 2000, batch_size: int = 32, val_frac: float = 0.15):
    """Build train/val dataloaders from a MedNIST subset."""
    class_dirs = ensure_mednist()
    full_ds = MedNISTImageDataset(class_dirs)
    n_total = len(full_ds)
    log.info(f"MedNIST total: {n_total} images, {NUM_CLASSES} classes")

    train_idx, val_idx, _ = split_indices(n_total, ratios=(1.0, 0.0, 0.0))  # shuffle all
    keep = train_idx[:subset_size]
    n_val = int(len(keep) * val_frac)
    val_idx, train_idx = keep[:n_val], keep[n_val:]

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train(epochs: int = 5, subset_size: int = 2000, lr: float = 1e-3, device: str = "cpu", batch_size: int = 32):
    """Train the classifier and save the best checkpoint + metrics."""
    set_seed(42)
    train_loader, val_loader = get_dataloaders(subset_size=subset_size, batch_size=batch_size)
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                v_loss += criterion(out, labels).item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total += imgs.size(0)
        val_loss = v_loss / max(v_total, 1)
        val_acc = v_correct / max(v_total, 1)

        log.info(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), DEFAULT_CKPT)
            log.info(f"  -> saved best model (val_acc={val_acc:.4f}) to {DEFAULT_CKPT}")

    # Save metrics
    metrics = {
        "best_val_acc": best_val_acc,
        "epochs": epochs,
        "subset_size": subset_size,
        "history": history,
        "classes": MEDNIST_CLASSES,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Training complete. Best val_acc={best_val_acc:.4f}")
    return history, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train medical imaging demo model on MedNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset", type=int, default=2000, help="number of images to train on")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(epochs=args.epochs, subset_size=args.subset, lr=args.lr, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
