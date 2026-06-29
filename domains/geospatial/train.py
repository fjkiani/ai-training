"""Train the geospatial segmentation model on tiled data.

Usage:
    python -m domains.geospatial.train --epochs 10

If no prepared dataset exists, generates a synthetic land/water one for the demo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from shared import set_seed, ensure_dir, get_logger, split_indices
from .model import build_model

log = get_logger("geo.train")

HERE = Path(__file__).parent
DATA_DIR = ensure_dir(HERE / "data")
PREPARED_DIR = DATA_DIR / "prepared"
MODEL_DIR = ensure_dir(HERE / "models")
DEFAULT_CKPT = MODEL_DIR / "geo_unet.pth"
TILE_SIZE = 256


class TileDataset(Dataset):
    """Loads image/mask .npy pairs from a prepared dataset directory."""

    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        with open(self.data_dir / "manifest.json") as f:
            manifest = json.load(f)
        self.indices = manifest["splits"][split]["indices"]
        self.split = split

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        img = np.load(self.img_dir / f"{self.split}_{i:05d}.npy")
        img = torch.from_numpy(img).float()
        if img.ndim == 2:
            img = img.unsqueeze(0)
        mask_path = self.mask_dir / f"{self.split}_{i:05d}.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        else:
            mask = torch.zeros(1, img.shape[-2], img.shape[-1])
        return img, mask


def generate_synthetic_dataset(out_dir: str | Path, n_tiles: int = 500, tile_size: int = TILE_SIZE, seed: int = 42):
    """Generate a synthetic land/water segmentation dataset for the demo.

    Creates RGB tiles where 'land' is textured green/brown and 'water' is blue,
    with a wavy boundary. Masks are binary (1=land, 0=water).
    """
    rng = np.random.RandomState(seed)
    out = ensure_dir(out_dir)
    (out / "images").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)

    train_idx, val_idx, test_idx = split_indices(n_tiles, seed=seed)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split, idxs in splits.items():
        for i in idxs:
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.1, 0.3)
            ys = np.linspace(0, 1, tile_size)
            boundary = (0.5 + amp * np.sin(ys * 2 * np.pi + phase)) * tile_size
            mask = np.zeros((tile_size, tile_size), dtype=np.float32)
            for col in range(tile_size):
                b = int(boundary[col])
                mask[:b, col] = 1.0  # land above boundary
            land_noise = rng.uniform(-0.05, 0.05, (tile_size, tile_size))
            water_noise = rng.uniform(-0.03, 0.03, (tile_size, tile_size))
            img = np.zeros((3, tile_size, tile_size), dtype=np.float32)
            img[0] = np.where(mask > 0, 0.3 + land_noise, 0.05 + water_noise)
            img[1] = np.where(mask > 0, 0.5 + land_noise, 0.15 + water_noise)
            img[2] = np.where(mask > 0, 0.2 + land_noise, 0.6 + water_noise)
            img = np.clip(img, 0, 1)
            np.save(out / "images" / f"{split}_{i:05d}.npy", img)
            np.save(out / "masks" / f"{split}_{i:05d}.npy", mask)

    manifest = {
        "tile_size": tile_size,
        "num_bands": 3,
        "splits": {s: {"count": len(idxs), "indices": idxs} for s, idxs in splits.items()},
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Synthetic dataset generated at {out}: {n_tiles} tiles")
    return out


def train(epochs: int = 10, lr: float = 1e-3, device: str = "cpu", batch_size: int = 8):
    """Train the U-Net on the (synthetic) tiled dataset."""
    set_seed(42)
    if not (PREPARED_DIR / "manifest.json").exists():
        log.info("No prepared dataset found, generating synthetic land/water dataset...")
        generate_synthetic_dataset(PREPARED_DIR, n_tiles=500)

    train_ds = TileDataset(PREPARED_DIR, "train")
    val_ds = TileDataset(PREPARED_DIR, "val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    in_ch = train_ds[0][0].shape[0]
    model = build_model(in_channels=in_ch, device=device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_iou = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        inter, union, v_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                v_loss += criterion(out, masks).item() * imgs.size(0)
                pred = (out > 0.5).float()
                inter += (pred * masks).sum().item()
                union += ((pred + masks) > 0).float().sum().item()
        val_iou = inter / (union + 1e-8)
        val_loss = v_loss / len(val_ds)
        log.info(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} val_iou={val_iou:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_iou": val_iou})

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), DEFAULT_CKPT)
            log.info(f"  -> saved best model (val_iou={val_iou:.4f}) to {DEFAULT_CKPT}")

    metrics = {"best_val_iou": best_iou, "epochs": epochs, "history": history}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Training complete. Best val_iou={best_iou:.4f}")
    return history, best_iou


def main():
    parser = argparse.ArgumentParser(description="Train geospatial segmentation model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train(epochs=args.epochs, lr=args.lr, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
