---
title: AI Geospatial Segmentation
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Geospatial — Land/Water Segmentation

A U-Net (ResNet18 encoder) for binary land/water segmentation. **~100% IoU** on synthetic
coastline tiles in 10 epochs of CPU training.

## What this Space contains

A full multi-tab portfolio for the geospatial domain. Visit the tabs in order:

1. **Try the Demo** — upload a 256×256 satellite-like tile or use one of the included samples
2. **Data & Preprocessing** — synthetic tile generation, CRS alignment, label rasterization
3. **Model & Training** — `segmentation-models-pytorch` U-Net + Albumentations augmentation
4. **Evaluation** — per-epoch IoU curve + sanity-check framing for synthetic data
5. **Code Walkthrough** — the four files that matter
6. **Lessons Learned** — synthetic-vs-real-data delta, expected production IoU ranges

## Headline metrics

| Metric | Value |
|:---|:---:|
| Best validation IoU | **0.9999** |
| Training time (10 epochs, CPU) | <15 minutes |
| Model parameters | ~14M (ResNet18 encoder + U-Net decoder) |
| Inference latency | ~1 second per 256×256 tile |
| Task | Binary land/water segmentation |
| Dataset | Synthetic coastline tiles |

## The honest story

The IoU is a sanity check, not a flex. The real-world IoU range for **Sentinel-2 land/water
segmentation is 0.92–0.95** — the drop from synthetic to real comes from atmospheric noise,
cloud cover, and label uncertainty, not from architecture choice. The same pipeline trained
on Sentinel-2 + OpenStreetMap water polygons would land in that range.

## Related links

- **Portfolio page:** [jedilabs.org/ai-training/geospatial](https://jedilabs.org/ai-training/geospatial)
- **Source code:** [github.com/fjkiani/ai-training/tree/main/domains/geospatial](https://github.com/fjkiani/ai-training/tree/main/domains/geospatial)
- **Blog post:** [jedilabs.org/blog/geospatial-geotiff-to-segmentation](https://jedilabs.org/blog/geospatial-geotiff-to-segmentation)
- **Built by:** [Jedi Labs](https://jedilabs.org)

If your team is wrestling with GeoTIFF pipelines — CRS pain, label rasterization, cloud cover —
[book a discovery call](https://jedilabs.org/contact).
