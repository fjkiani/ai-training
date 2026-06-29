"""Train the audio classifier on ESC-50 or a local dataset.

ESC-50: 2,000 environmental sound clips, 50 classes, ~600MB, CC-licensed.
For a fast CPU demo, the Random Forest trains in seconds.

Usage:
    python -m domains.audio.train --download-esc50 --epochs 1

Or with a local dataset:
    python -m domains.audio.train --audio-dir /path/to/audio_classes
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from shared import set_seed, ensure_dir, get_logger
from .model import build_model, save_model, load_label_encoder
from .pipeline import prepare_dataset, get_feature_vector

log = get_logger("audio.train")

HERE = Path(__file__).parent
DATA_DIR = ensure_dir(HERE / "data")
ESC50_DIR = DATA_DIR / "ESC-50"
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.tar.gz"
MODEL_DIR = ensure_dir(HERE / "models")
DEFAULT_CKPT = MODEL_DIR / "audio_rf.pkl"
DEFAULT_LABEL_ENCODER = MODEL_DIR / "label_encoder.pkl"


def download_esc50():
    """Download and extract ESC-50 dataset."""
    if ESC50_DIR.exists() and any(ESC50_DIR.iterdir()):
        log.info(f"ESC-50 already exists at {ESC50_DIR}")
        return ESC50_DIR

    log.info("Downloading ESC-50 (~600MB)...")
    tar_path = DATA_DIR / "ESC-50.tar.gz"
    urllib.request.urlretrieve(ESC50_URL, tar_path)
    log.info("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    # The extracted folder is ESC-50-master
    extracted = DATA_DIR / "ESC-50-master"
    if extracted.exists():
        extracted.rename(ESC50_DIR)
    tar_path.unlink(missing_ok=True)
    log.info(f"ESC-50 ready at {ESC50_DIR}")
    return ESC50_DIR


def organize_esc50_by_class(esc50_dir: Path) -> Path:
    """ESC-50 stores all audio in one 'audio' folder with filenames like 1-100032-A-0.wav.

    The 5th field (after splitting by '-') is the class ID. We create symlinks
    organized by class name for the pipeline.
    """
    import csv

    audio_dir = esc50_dir / "audio"
    meta_path = esc50_dir / "meta" / "esc50.csv"

    # Read class mapping
    class_names = {}
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names[row["target"]] = row["category"]

    organized = DATA_DIR / "esc50_organized"
    organized.mkdir(exist_ok=True)
    for class_id, name in class_names.items():
        (organized / name).mkdir(exist_ok=True)

    # Symlink files into class folders
    for wav in audio_dir.glob("*.wav"):
        parts = wav.stem.split("-")
        class_id = parts[3]
        class_name = class_names.get(class_id, "unknown")
        link = organized / class_name / wav.name
        if not link.exists():
            link.symlink_to(wav)

    log.info(f"Organized ESC-50 into {organized} ({len(class_names)} classes)")
    return organized


def train(audio_dir: str | Path | None = None, download: bool = False, seed: int = 42):
    """Train the RF classifier on ESC-50 or a local dataset."""
    set_seed(seed)

    if audio_dir is None:
        if download or not (DATA_DIR / "prepared" / "manifest.json").exists():
            esc50 = download_esc50()
            audio_dir = organize_esc50_by_class(esc50)
        else:
            audio_dir = DATA_DIR / "esc50_organized"

    prepared_dir = DATA_DIR / "prepared"
    if not (prepared_dir / "manifest.json").exists():
        log.info(f"Preparing dataset from {audio_dir}...")
        manifest = prepare_dataset(audio_dir, prepared_dir, seed=seed)
    else:
        with open(prepared_dir / "manifest.json") as f:
            manifest = json.load(f)
        log.info(f"Using existing prepared dataset: {manifest['n_samples']} samples, {manifest['n_classes']} classes")

    X = np.load(prepared_dir / "X.npy")
    y = np.load(prepared_dir / "y.npy")
    split_idx = manifest["split_indices"]

    X_train, y_train = X[split_idx["train"]], y[split_idx["train"]]
    X_val, y_val = X[split_idx["val"]], y[split_idx["val"]]
    X_test, y_test = X[split_idx["test"]], y[split_idx["test"]]

    log.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    log.info("Training Random Forest...")
    model = build_model(random_state=seed)
    model.fit(X_train, y_train)

    # Evaluate
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    log.info(f"Val accuracy: {val_acc:.4f}")
    log.info(f"Test accuracy: {test_acc:.4f}")

    # Classification report (top classes for readability)
    report = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)

    save_model(model, DEFAULT_CKPT)
    # Copy label encoder to model dir
    import shutil
    shutil.copy(prepared_dir / "label_encoder.pkl", DEFAULT_LABEL_ENCODER)

    metrics = {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "n_samples": manifest["n_samples"],
        "n_classes": manifest["n_classes"],
        "class_names": manifest["class_names"],
        "feature_dim": manifest["feature_dim"],
        "report": report,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Model saved to {DEFAULT_CKPT}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train audio classifier on ESC-50")
    parser.add_argument("--audio-dir", type=str, default=None, help="Local audio directory (one subfolder per class)")
    parser.add_argument("--download-esc50", action="store_true", help="Download ESC-50 dataset")
    args = parser.parse_args()
    train(audio_dir=args.audio_dir, download=args.download_esc50)


if __name__ == "__main__":
    main()
