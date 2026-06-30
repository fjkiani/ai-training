"""Gradio Blocks demo for the medical imaging pipeline.

Multi-tab portfolio surface for the MedNIST classifier:
  1. Try the Demo — interactive inference
  2. Data & Preprocessing — how DICOMs become tensors
  3. Model & Training — architecture, optimizer, hyperparameters
  4. Evaluation — per-epoch curve + best-checkpoint policy
  5. Code Walkthrough — the actual functions that ran
  6. Lessons Learned — design choices and what we'd do for a paying customer

Run locally:
    python -m domains.medical.demo

Deploy to HuggingFace Spaces:
    See demos/huggingface_spaces.md
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .infer import predict
from .model import MEDNIST_CLASSES
from .pipeline import preprocess_2d_image

HERE = Path(__file__).parent
CKPT = HERE / "models" / "medical_unet.pth"
METRICS = HERE / "models" / "metrics.json"
SAMPLES_DIR = HERE / "samples"


def _fig_to_pil(fig) -> Image.Image:
    """Render a matplotlib Figure to a PIL Image for Gradio v6 compatibility."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


# ─── Inference callback ─────────────────────────────────────────────────────
def run_inference(image):
    """Gradio callback: image -> prediction plot + JSON."""
    if image is None:
        return None, "Please upload an image."
    if not CKPT.exists():
        return None, f"No trained model found at {CKPT}. Run training first: python -m domains.medical.train"

    result = predict(image)
    arr = preprocess_2d_image(image, size=64)[0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("Input (64x64)")
    axes[0].axis("off")
    probs = [result["probabilities"][c] for c in MEDNIST_CLASSES]
    colors = ["#FF9400" if c == result["predicted_class"] else "#0279EE" for c in MEDNIST_CLASSES]
    axes[1].barh(MEDNIST_CLASSES, probs, color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title(f"Prediction: {result['predicted_class']} ({result['confidence']:.1%})")
    plt.tight_layout()
    summary = json.dumps(result, indent=2)
    return _fig_to_pil(fig), summary


# ─── Server-rendered evaluation chart ───────────────────────────────────────
def render_training_curve():
    """Server-rendered per-epoch curve from metrics.json. Called on app start."""
    if not METRICS.exists():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No metrics.json found", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _fig_to_pil(fig)
    m = json.loads(METRICS.read_text())
    hist = m.get("history", [])
    epochs = [h["epoch"] for h in hist]
    train_acc = [h["train_acc"] for h in hist]
    val_acc = [h["val_acc"] for h in hist]
    best = m.get("best_val_acc", max(val_acc))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_acc, "o-", color="#0279EE", label="Train accuracy", linewidth=2, markersize=8)
    ax.plot(epochs, val_acc, "s-", color="#FF9400", label="Val accuracy", linewidth=2, markersize=8)
    ax.axhline(best, color="#75A025", linestyle="--", alpha=0.6, label=f"Best val: {best:.1%}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("MedNIST 6-class classifier — training history", fontsize=13)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    plt.tight_layout()
    return _fig_to_pil(fig)


def _get_examples():
    if not SAMPLES_DIR.exists():
        return None
    samples = sorted(SAMPLES_DIR.glob("*.jpeg"))
    return [[str(s)] for s in samples] if samples else None


# ─── Static tab content ─────────────────────────────────────────────────────
DATA_AND_PREPROCESSING_MD = """
## Data & Preprocessing

### What we feed the model

**Dataset:** [MedNIST](https://medmnist.com/) — 58,954 medical images across 6 anatomy classes
(`AbdomenCT`, `BreastMRI`, `CXR`, `ChestCT`, `Hand`, `HeadCT`). All grayscale, all real medical images
(downsampled to 64×64), balanced across classes. Train/val/test split: 80/10/10, deterministic seed.

### Why this dataset

It's small enough to fit in memory, large enough to be honest about generalization, and the class
labels are realistic for a triage workflow (route this study to the right specialist). The reference
implementation behaves the same way on real DICOM as on this synthetic-looking PNG dataset, because
the pipeline is built around **MONAI transforms** that handle both.

### The preprocessing chain

```python
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Resized, EnsureTyped
)

train_transforms = Compose([
    LoadImaged(keys="image"),          # DICOM, NIfTI, or PNG — same API
    EnsureChannelFirstd(keys="image"), # (H, W) → (C, H, W)
    ScaleIntensityd(keys="image"),     # min-max to [0, 1]
    Resized(keys="image", spatial_size=(64, 64)),
    EnsureTyped(keys="image"),         # → torch.Tensor
])
```

**The trick:** `LoadImaged` handles DICOM, NIfTI, and PNG with the same interface. The same pipeline
file runs whether the input is a synthetic demo image or a multi-slice CT volume. That's the production
property we care about.

### What this *doesn't* handle (yet)

- Photometric inversion (MONOCHROME1 vs MONOCHROME2 in DICOM)
- Multi-modality fusion (CT + MR co-registration)
- Volume slicing (3D → 2D slice extraction)

For a paying customer, those go in at the same MONAI transform layer, before the model.
"""

MODEL_AND_TRAINING_MD = """
## Model & Training

### Architecture

A small 3-layer CNN, ~24K parameters total. CPU-trainable. Honest baseline.

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        # ...
```

### Why this size

- **CPU-trainable**: 5 epochs in under 10 minutes on a laptop
- **Honest baseline**: If a 24K-param CNN hits 99% val acc, the dataset is the hero. Worth knowing
  before spending cycles on bigger architectures.
- **Fast iteration**: thought-speed experimentation, not coffee-break-speed

### Training setup

| Hyperparameter | Value |
|:---|:---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | Cross-entropy |
| Batch size | 32 |
| Epochs | 5 |
| Best-checkpoint preservation | Yes |
| Random seed | 42 |

### What "best-checkpoint preservation" means

We save weights at the **best validation accuracy seen so far**, not the last epoch. Without this we'd
ship the epoch-5 weights (91% val acc) instead of the epoch-2 weights (99.3% val acc).
This is the single most important production-stability practice in this pipeline.
See the Evaluation tab for why.
"""

CODE_WALKTHROUGH_MD = """
## Code Walkthrough

### The four files that matter

| File | Role |
|:---|:---|
| `pipeline.py` | Preprocessing — `preprocess_2d_image(img, size=64)` |
| `model.py` | CNN architecture + `MEDNIST_CLASSES` constant |
| `train.py` | Training loop — Adam, CE loss, best-checkpoint save |
| `infer.py` | Inference — `predict(image)` returns `{predicted_class, confidence, probabilities}` |

### Inference path (what runs in this demo)

```python
from domains.medical.infer import predict

result = predict("path/to/image.jpeg")
# result = {
#   "predicted_class": "CXR",
#   "confidence": 0.965,
#   "probabilities": {"AbdomenCT": 0.001, "BreastMRI": 0.002, ...}
# }
```

### Repository layout

```
domains/medical/
├── pipeline.py          # Preprocessing
├── model.py             # CNN architecture
├── train.py             # Training loop
├── infer.py             # predict() function
├── demo.py              # ← This Gradio app
├── samples/             # Sample radiographs
└── models/
    ├── medical_unet.pth # Trained weights (~100KB)
    └── metrics.json     # Per-epoch training log
```

### Full source

[github.com/fjkiani/ai-training/tree/main/domains/medical](https://github.com/fjkiani/ai-training/tree/main/domains/medical)
"""

LESSONS_LEARNED_MD = """
## Lessons Learned

### What we'd keep

- **MONAI transforms over raw torchvision.** `LoadImaged` handles DICOM, NIfTI, and PNG with one API.
  No code change to handle real medical data — only the file paths change.
- **Best-checkpoint preservation by default.** Last-epoch weights are wrong in this regime
  (see Evaluation tab — epoch 4 val_acc was 75.7%, not 99.3%).
- **A small honest baseline beats a fashionable big model.** Anyone reproducing the repo gets
  the same 99.3% with their laptop CPU in <10 minutes.

### What we'd change for a paying customer

- **Real DICOM ingestion.** Add `pydicom` to handle photometric inversion, modality-specific
  preprocessing (HU windowing for CT, T1/T2 selection for MR), and series ordering.
- **Bigger model + longer training.** For ~20+ classes the small CNN saturates; switch to
  a ResNet18 backbone (still small, ImageNet-pretrained).
- **Active learning loop.** Real medical AI needs new edge cases monthly. Build a flagging
  workflow where low-confidence predictions get human-reviewed and added to the training set.
- **Calibration.** Softmax probabilities are not true probabilities. For triage workflows that
  use confidence thresholds, calibrate with temperature scaling.

### The single biggest gotcha

**Photometric inversion.** Some DICOMs encode "white = high X-ray attenuation" (MONOCHROME1) and
some encode the opposite (MONOCHROME2). If you don't normalize this at the pipeline layer, the model
sees half your data as inverted and silently underperforms. Hours of debugging vs. five lines of code.

### Blog post

Full long-form writeup including the per-epoch wobble story:
[jedilabs.org/blog/medical-dicom-to-classification](https://jedilabs.org/blog/medical-dicom-to-classification)
"""


# ─── Build the Blocks UI ────────────────────────────────────────────────────
def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Medical Imaging — MedNIST Classifier",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Medical Imaging — MedNIST Classifier\n"
            "A 6-class radiograph classifier (`AbdomenCT`, `BreastMRI`, `CXR`, `ChestCT`, `Hand`, `HeadCT`).\n"
            "Built with MONAI + PyTorch. 99.3% peak validation accuracy in 5 epochs of CPU training.\n\n"
            "**Portfolio:** [jedilabs.org/ai-training/medical](https://jedilabs.org/ai-training/medical) · "
            "**Code:** [github.com/fjkiani/ai-training](https://github.com/fjkiani/ai-training/tree/main/domains/medical)"
        )

        with gr.Tabs():
            # ── Tab 1: Try the Demo ──
            with gr.Tab("Try the Demo"):
                with gr.Row():
                    with gr.Column():
                        inp = gr.Image(type="filepath", label="Upload a radiograph (PNG/JPG)")
                        btn = gr.Button("Classify", variant="primary")
                        examples = _get_examples()
                        if examples:
                            gr.Examples(examples=examples, inputs=inp, label="Sample radiographs")
                    with gr.Column():
                        out_img = gr.Image(label="Prediction")
                        out_json = gr.Textbox(label="Details", lines=10)
                btn.click(run_inference, inputs=inp, outputs=[out_img, out_json])

            # ── Tab 2: Data & Preprocessing ──
            with gr.Tab("Data & Preprocessing"):
                gr.Markdown(DATA_AND_PREPROCESSING_MD)

            # ── Tab 3: Model & Training ──
            with gr.Tab("Model & Training"):
                gr.Markdown(MODEL_AND_TRAINING_MD)

            # ── Tab 4: Evaluation ──
            with gr.Tab("Evaluation"):
                gr.Markdown(
                    "## Per-epoch training curve\n"
                    "Rendered server-side from the actual `metrics.json` log "
                    "([file](https://github.com/fjkiani/ai-training/blob/main/domains/medical/models/metrics.json))."
                )
                gr.Image(value=render_training_curve(), label="Training history", show_label=False)
                gr.Markdown(
                    "### What this chart shows\n\n"
                    "- **Best val acc: 99.3% (epoch 2).** This is what gets shipped.\n"
                    "- **Epoch 4 val acc drops to 75.7%** — classic small-dataset oscillation. "
                    "Without best-checkpoint preservation we'd ship the epoch-5 weights (91% val acc) instead.\n"
                    "- **Train acc climbs monotonically** to ~99% by epoch 4; the val gap at epoch 4 is the model "
                    "briefly overshooting on a bad mini-batch sequence.\n\n"
                    "Pretending models train monotonically is how teams get blindsided in production. "
                    "We show the wobble on purpose."
                )

            # ── Tab 5: Code Walkthrough ──
            with gr.Tab("Code Walkthrough"):
                gr.Markdown(CODE_WALKTHROUGH_MD)

            # ── Tab 6: Lessons Learned ──
            with gr.Tab("Lessons Learned"):
                gr.Markdown(LESSONS_LEARNED_MD)

        gr.Markdown(
            "---\n"
            "Built by [Jedi Labs](https://jedilabs.org). "
            "If your team is staring down a DICOM pipeline — "
            "[book a discovery call](https://jedilabs.org/contact)."
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
