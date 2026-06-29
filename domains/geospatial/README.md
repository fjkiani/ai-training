# Geospatial Pipeline

GeoTIFF → tiled patches + label masks → land/water segmentation demo

## Overview

A geospatial preprocessing and segmentation pipeline built on rasterio, geopandas, and segmentation-models-pytorch. The pipeline handles GeoTIFF ingestion, reprojection, normalization, tiling, and vector label rasterization. The demo model is a U-Net (ResNet18 encoder) trained on synthetic land/water tiles.

## Pipeline

```
GeoTIFF → load (rasterio) → reproject (EPSG:3857) → per-band normalize [0,1]
→ tile (256×256) → rasterize vector labels (GeoJSON/Shapefile)
→ train/val/test split → .npy image+mask pairs
```

## Demo Model

- **Architecture:** U-Net with ResNet18 encoder (segmentation-models-pytorch)
- **Dataset:** Synthetic land/water tiles (500 tiles, 256×256, 3-band RGB)
- **Task:** Binary segmentation (land vs. water)
- **Training:** 10 epochs, CPU (~10 min)
- **Metrics:** IoU, Dice, pixel accuracy

## Quickstart

```bash
# Install system dependencies
apt-get install -y gdal-bin libgdal-dev

# Install Python dependencies
pip install rasterio geopandas shapely pyproj segmentation-models-pytorch torch gradio

# Train the model (auto-generates synthetic dataset if none exists)
python -m domains.geospatial.train --epochs 10

# Run inference on a single tile
python -c "from domains.geospatial.infer import predict_tile; print(predict_tile(image_array))"

# Launch the Gradio demo
python -m domains.geospatial.demo
```

## Using Real GeoTIFF Data

```python
from domains.geospatial.pipeline import prepare_dataset

# Tile a GeoTIFF with vector labels
manifest = prepare_dataset(
    raster_path="satellite.tif",
    labels_path="labels.geojson",
    output_dir="data/prepared",
    tile_size=256,
)
```

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | GeoTIFF tiling, normalization, label rasterization |
| `model.py` | U-Net (ResNet18) segmentation model |
| `train.py` | Training loop with synthetic dataset generation |
| `infer.py` | Single-tile inference |
| `demo.py` | Gradio web interface |
| `tests/` | Unit tests for pipeline and model |

## Credits

- Tiling and rasterization concepts from [opengeos/geoai](https://github.com/opengeos/geoai) (MIT)
- Model architecture from [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) (MIT)

## License

MIT
