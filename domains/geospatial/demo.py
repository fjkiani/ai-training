"""Gradio Blocks demo for the geospatial segmentation pipeline.

Multi-tab portfolio surface for the land/water U-Net:
  1. Try the Demo — interactive inference
  2. Data & Preprocessing — synthetic tiles, CRS, label rasterization
  3. Model & Training — U-Net ResNet18 architecture + hyperparameters
  4. Evaluation — per-epoch IoU curve + sanity-check framing
  5. Code Walkthrough — the actual functions that ran
  6. Lessons Learned — design choices and the real-data delta

Run locally:
    python -m domains.geospatial.demo
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

from .infer import predict_tile

HERE = Path(__file__).parent
CKPT = HERE / "models" / "geo_unet.pth"
METRICS = HERE / "models" / "metrics.json"
SAMPLES_DIR = HERE / "samples"


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def run_inference(image):
    if image is None:
        return None, "Please upload an image."
    if not CKPT.exists():
        return None, f"No trained model found at {CKPT}. Run training first: python -m domains.geospatial.train"

    img = np.array(Image.open(image).convert("RGB").resize((256, 256))).transpose(2, 0, 1) / 255.0
    mask = predict_tile(img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img.transpose(1, 2, 0))
    axes[0].set_title("Input (256x256)")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    overlay = img.transpose(1, 2, 0).copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.6 + np.array([0.0, 1.0, 0.0]) * 0.4
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay (green=land)")
    axes[2].axis("off")
    plt.tight_layout()

    land_pct = float(mask.mean()) * 100
    summary = f"Land coverage: {land_pct:.1f}%\nWater coverage: {100 - land_pct:.1f}%"
    return _fig_to_pil(fig), summary


def render_training_curve():
    """Server-rendered per-epoch IoU curve from metrics.json."""
    if not METRICS.exists():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No metrics.json found", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _fig_to_pil(fig)
    m = json.loads(METRICS.read_text())
    hist = m.get("history", [])
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_iou = [h["val_iou"] for h in hist]
    best = m.get("best_val_iou", max(val_iou))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    line1 = ax.plot(epochs, train_loss, "o-", color="#0279EE", label="Train loss (BCE)", linewidth=2, markersize=7)
    line2 = ax2.plot(epochs, val_iou, "s-", color="#FF9400", label="Val IoU", linewidth=2, markersize=7)
    line3 = ax2.axhline(best, color="#75A025", linestyle="--", alpha=0.6, label=f"Best IoU: {best:.4f}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train loss", fontsize=12, color="#0279EE")
    ax2.set_ylabel("Val IoU", fontsize=12, color="#FF9400")
    ax.set_title("Land/water U-Net — training history (synthetic data)", fontsize=13)
    ax2.set_ylim(0.85, 1.005)
    # Combined legend (lines from both axes + the axhline)
    lines = line1 + line2 + [line3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="center right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    plt.tight_layout()
    return _fig_to_pil(fig)


def _get_examples():
    if not SAMPLES_DIR.exists():
        return None
    samples = sorted(SAMPLES_DIR.glob("*.png"))
    return [[str(s)] for s in samples] if samples else None


DATA_AND_PREPROCESSING_MD = """
## Data & Preprocessing

### What we feed the model

**Dataset:** Synthetic coastline tiles — generated programmatically with `rasterio` + `numpy`.
256×256 RGB images with binary land/water masks. Controlled noise levels, balanced class distribution.

### Why synthetic, not Sentinel-2 or Landsat

Three stacked reasons:

1. **Reproducibility.** Anyone running the repo gets identical data with the same seed.
2. **No licensing tangle.** Real satellite data is mostly open-license, but not always; demo data
   should be unambiguous.
3. **The pipeline is the lesson, not the data.** The transforms and model code are identical whether
   you feed it synthetic tiles or real GeoTIFFs.

For paying-customer work, the data step swaps in `rasterio.open()` over your actual scenes and
a `WindowedReader` for memory-bounded tile extraction. The rest of the pipeline is unchanged.

### CRS alignment (the real-data gotcha)

```python
import rasterio
from rasterio.warp import calculate_default_transform, reproject

# Reproject all scenes to a common CRS (e.g., EPSG:3857 for web mercator)
with rasterio.open("scene.tif") as src:
    transform, width, height = calculate_default_transform(
        src.crs, "EPSG:3857", src.width, src.height, *src.bounds
    )
    # ... reproject into target CRS
```

Every tile must be in the same CRS before the model sees it. This is the silent bug source on
production geospatial pipelines.

### Label rasterization

Vector labels (polygons of "this is water, this is land") become pixel masks at the model's
resolution via `rasterio.features.rasterize`:

```python
from rasterio.features import rasterize
mask = rasterize(
    [(geom, 1) for geom in water_geometries],
    out_shape=(256, 256),
    transform=tile_transform,
    fill=0
)
```

### Augmentation

Albumentations — not torchvision — because it handles segmentation masks correctly:

```python
import albumentations as A
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),  # ImageNet stats
])
```

A horizontal flip on the image *also* flips the mask. Torchvision doesn't do this by default;
Albumentations does.
"""

MODEL_AND_TRAINING_MD = """
## Model & Training

### Architecture

A U-Net with a ResNet18 encoder (ImageNet-pretrained), 2-class output head. Built with
`segmentation-models-pytorch` rather than rolling our own.

```python
import segmentation_models_pytorch as smp
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
```

### Why ResNet18 (not ResNet50 or EfficientNet)

- **Smaller**: 11M parameters vs 25M+. Faster training on CPU.
- **ImageNet pretraining is enough**: Binary segmentation on clear visual boundaries doesn't need
  fine-grained low-level features. The first few conv layers transfer fine.
- **Bigger encoders are for fine-grained problems**: A 9-class land-cover task would benefit from
  EfficientNet-B3+. Binary land/water does not.

### Why `segmentation-models-pytorch`

It gives you pretrained encoders + decoder pairs out of the box. The library has been battle-tested
across thousands of segmentation projects on Kaggle and in industry. Reinventing U-Net in pure
PyTorch is a research exercise, not a production decision.

### Training setup

| Hyperparameter | Value |
|:---|:---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | Binary cross-entropy with logits |
| Batch size | 8 |
| Epochs | 10 |
| Best-checkpoint preservation | Yes |
| Random seed | 42 |

### Metric

IoU (Intersection over Union) — the standard for binary segmentation. Best epoch checkpoint is
saved; final model weights are the best, not the last.
"""

CODE_WALKTHROUGH_MD = """
## Code Walkthrough

### The four files that matter

| File | Role |
|:---|:---|
| `pipeline.py` | Synthetic tile generation + augmentation |
| `model.py` | U-Net architecture wrapper |
| `train.py` | Training loop + IoU tracking + best-checkpoint save |
| `infer.py` | `predict_tile(img)` returns the binary mask |

### Inference path (what runs in this demo)

```python
from domains.geospatial.infer import predict_tile

# Load + normalize a 256x256 RGB tile
img = np.array(Image.open("tile.png").convert("RGB").resize((256, 256))).transpose(2, 0, 1) / 255.0

# Predict
mask = predict_tile(img)
# mask: (256, 256) ndarray of {0, 1}
```

### Repository layout

```
domains/geospatial/
├── pipeline.py          # Tile generation + augmentation
├── model.py             # U-Net wrapper
├── train.py             # Training loop
├── infer.py             # predict_tile() function
├── demo.py              # ← This Gradio app
├── samples/             # 2 sample coastline tiles
└── models/
    ├── geo_unet.pth     # Trained weights
    └── metrics.json     # Per-epoch training log
```

### Full source

[github.com/fjkiani/ai-training/tree/main/domains/geospatial](https://github.com/fjkiani/ai-training/tree/main/domains/geospatial)
"""

LESSONS_LEARNED_MD = """
## Lessons Learned

### What we'd keep

- **`segmentation-models-pytorch` as the starting point.** Pretrained encoders + battle-tested
  decoders. Don't reinvent U-Net.
- **Albumentations over torchvision** for segmentation augmentation. Mask-aware transforms are the
  whole reason it exists.
- **Show the IoU as a sanity check, not a flex.** Our 99.997% on synthetic data is the *correct*
  expected behavior; if the model didn't hit ~100% on clean synthetic boundaries, something is
  wrong with the loss, the metric, or the data loader.

### What we'd change for a paying customer

- **Real Sentinel-2 or Landsat scenes.** The same model code, trained on real labels from
  OpenStreetMap or the Copernicus Land Service.
- **Atmospheric correction.** Sen2Cor or similar. Without it, the model learns the sensor's
  atmospheric noise instead of the actual surface.
- **Cloud masking.** Even the best classifier can't see through opaque clouds; mask them out
  upstream of the model with a separate cloud detector.
- **Temporal compositing.** Multi-date composites (median over the last 90 days) handle clouds,
  shadows, and seasonal variation better than single-scene inference.
- **Multi-class.** Binary is the warm-up. Production land-cover work uses 7-15 classes from
  the IPCC or ESA classification schemes.

### The single biggest gotcha

**Coordinate reference systems.** Sentinel-2 scenes ship in UTM zones — different zones for
different scenes within the same project. If you don't reproject to a common CRS at the pipeline
layer, you'll get tile-edge artifacts and misaligned labels. `rasterio.warp.reproject` is the
five-line fix.

### Expected real-world IoU range

| Task | State-of-the-art IoU |
|:---|:---:|
| Binary land/water (Sentinel-2) | 0.92–0.95 |
| 9-class land cover (Sentinel-2) | 0.78–0.85 |
| Building footprint (high-res aerial) | 0.80–0.88 |
| Crop type (multi-spectral) | 0.65–0.80 |

Our 99.997% on synthetic data is *not* the production number. Real data degrades gracefully into
this 0.78–0.95 range. The drop is from data difficulty, not architecture choice.

### Blog post

Full long-form writeup including the synthetic-vs-real framing:
[jedilabs.org/blog/geospatial-geotiff-to-segmentation](https://jedilabs.org/blog/geospatial-geotiff-to-segmentation)
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Geospatial — Land/Water Segmentation",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Geospatial — Land/Water Segmentation\n"
            "A U-Net (ResNet18 encoder) trained for binary land/water segmentation on synthetic "
            "coastline tiles. 10 epochs, CPU-trainable, ~100% IoU on the synthetic validation set.\n\n"
            "**Portfolio:** [jedilabs.org/ai-training/geospatial](https://jedilabs.org/ai-training/geospatial) · "
            "**Code:** [github.com/fjkiani/ai-training](https://github.com/fjkiani/ai-training/tree/main/domains/geospatial)"
        )

        with gr.Tabs():
            with gr.Tab("Try the Demo"):
                with gr.Row():
                    with gr.Column():
                        inp = gr.Image(type="filepath", label="Upload a 256x256 satellite-like tile")
                        btn = gr.Button("Segment", variant="primary")
                        examples = _get_examples()
                        if examples:
                            gr.Examples(examples=examples, inputs=inp, label="Sample tiles")
                    with gr.Column():
                        out_img = gr.Image(label="Segmentation")
                        out_txt = gr.Textbox(label="Coverage summary", lines=3)
                btn.click(run_inference, inputs=inp, outputs=[out_img, out_txt])

            with gr.Tab("Data & Preprocessing"):
                gr.Markdown(DATA_AND_PREPROCESSING_MD)

            with gr.Tab("Model & Training"):
                gr.Markdown(MODEL_AND_TRAINING_MD)

            with gr.Tab("Evaluation"):
                gr.Markdown(
                    "## Per-epoch training curve\n"
                    "Rendered server-side from the actual `metrics.json` log "
                    "([file](https://github.com/fjkiani/ai-training/blob/main/domains/geospatial/models/metrics.json)). "
                    "Note the dual y-axes: train loss falls; val IoU rises to ~1.0."
                )
                gr.Image(value=render_training_curve(), label="Training history", show_label=False)
                gr.Markdown(
                    "### What this chart shows\n\n"
                    "- **Train loss → 0.001** within 10 epochs. Synthetic data is easy.\n"
                    "- **Val IoU → 0.9999.** Same caveat: clean synthetic boundaries.\n"
                    "- **The curve is the sanity check.** If train loss didn't drop monotonically here, "
                    "something would be broken in the data loader or loss function. The monotonic drop "
                    "is what we want to see on a synthetic dataset.\n\n"
                    "Real-world Sentinel-2 land-cover IoU lands in the 0.78-0.95 range — not from "
                    "architecture limits but from data difficulty (clouds, atmospheric noise, "
                    "label uncertainty)."
                )

            with gr.Tab("Code Walkthrough"):
                gr.Markdown(CODE_WALKTHROUGH_MD)

            with gr.Tab("Lessons Learned"):
                gr.Markdown(LESSONS_LEARNED_MD)

        gr.Markdown(
            "---\n"
            "Built by [Jedi Labs](https://jedilabs.org). "
            "If your team is wrestling with a GeoTIFF pipeline — "
            "[book a discovery call](https://jedilabs.org/contact)."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
