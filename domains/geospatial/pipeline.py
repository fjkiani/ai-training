"""Geospatial preprocessing pipeline.

GeoTIFF -> tiled image patches + label masks ready for ML.

Pipeline steps:
  1. Load raster (rasterio)
  2. Reproject to common CRS (EPSG:3857) if needed
  3. Normalize bands to [0,1]
  4. Tile into fixed-size patches (256x256 or 512x512)
  5. Rasterize vector labels (GeoJSON/Shapefile) into binary masks
  6. Train/val/test split

Built on rasterio + geopandas + shapely, inspired by opengeos/geoai (MIT).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.plot import reshape_as_image

from shared import get_logger, split_indices

log = get_logger("geo.pipeline")

DEFAULT_TILE_SIZE = 256
DEFAULT_CRS = "EPSG:3857"


def load_raster(path: str | Path) -> rasterio.DatasetReader:
    """Open a GeoTIFF for reading."""
    return rasterio.open(str(path))


def normalize_bands(arr: np.ndarray, percentiles: Tuple[int, int] = (2, 98)) -> np.ndarray:
    """Per-band percentile normalization to [0,1]."""
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        # (bands, h, w) -> normalize per band
        for i in range(arr.shape[0]):
            lo, hi = np.percentile(arr[i][arr[i] > 0] if np.any(arr[i] > 0) else arr[i], percentiles)
            if hi > lo:
                arr[i] = np.clip((arr[i] - lo) / (hi - lo), 0, 1)
            else:
                arr[i] = 0.0
    return arr


def tile_raster(
    src: rasterio.DatasetReader,
    tile_size: int = DEFAULT_TILE_SIZE,
    nodata_threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """Generate tile offsets (col_off, row_off) covering the raster.

    Skips tiles that are mostly nodata.
    """
    width, height = src.width, src.height
    tiles = []
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            w = min(tile_size, width - col_off)
            h = min(tile_size, height - row_off)
            if w < tile_size or h < tile_size:
                continue  # skip partial edge tiles for uniformity
            # check nodata fraction
            window = Window(col_off, row_off, w, h)
            data = src.read(1, window=window)
            if src.nodata is not None:
                nodata_frac = np.mean(data == src.nodata)
            else:
                nodata_frac = np.mean(data == 0)
            if nodata_frac < nodata_threshold:
                tiles.append((col_off, row_off))
    log.info(f"Generated {len(tiles)} valid tiles ({tile_size}x{tile_size}) from {width}x{height} raster")
    return tiles


def read_tile(src: rasterio.DatasetReader, col_off: int, row_off: int, tile_size: int = DEFAULT_TILE_SIZE) -> np.ndarray:
    """Read a tile as a normalized (C, H, W) array in [0,1]."""
    window = Window(col_off, row_off, tile_size, tile_size)
    data = src.read(window=window)  # (bands, h, w)
    return normalize_bands(data)


def rasterize_labels(
    src: rasterio.DatasetReader,
    labels_path: str | Path,
    col_off: int,
    row_off: int,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> np.ndarray:
    """Rasterize vector labels into a binary mask for a tile.

    labels_path: GeoJSON or Shapefile with polygon geometries.
    Returns (H, W) uint8 mask.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import box

    gdf = gpd.read_file(str(labels_path))
    if gdf.crs is None:
        gdf = gdf.set_crs(src.crs)
    elif gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    window = Window(col_off, row_off, tile_size, tile_size)
    win_transform = rasterio.windows.transform(window, src.transform)
    tile_bounds = rasterio.windows.bounds(window, src.transform)
    tile_box = box(*tile_bounds)
    clipped = gdf[gdf.geometry.intersects(tile_box)]

    if len(clipped) == 0:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    shapes = [(geom, 1) for geom in clipped.geometry if geom is not None]
    mask = rasterize(
        shapes,
        out_shape=(tile_size, tile_size),
        transform=win_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return mask


def prepare_dataset(
    raster_path: str | Path,
    labels_path: Optional[str | Path],
    output_dir: str | Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    max_tiles: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Full pipeline: tile raster, rasterize labels, split, save .npy pairs.

    Returns a manifest dict with counts and file lists.
    """
    import json
    from shared import ensure_dir

    out = ensure_dir(output_dir)
    (out / "images").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)

    src = load_raster(raster_path)
    tiles = tile_raster(src, tile_size=tile_size)
    if max_tiles:
        tiles = tiles[:max_tiles]

    train_idx, val_idx, test_idx = split_indices(len(tiles), seed=seed)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    manifest = {"tile_size": tile_size, "splits": {}, "num_bands": src.count}
    for split_name, idxs in splits.items():
        files = []
        for i in idxs:
            col_off, row_off = tiles[i]
            img = read_tile(src, col_off, row_off, tile_size)
            np.save(out / "images" / f"{split_name}_{i:05d}.npy", img)
            if labels_path:
                mask = rasterize_labels(src, labels_path, col_off, row_off, tile_size)
                np.save(out / "masks" / f"{split_name}_{i:05d}.npy", mask)
            files.append(i)
        manifest["splits"][split_name] = {"count": len(idxs), "indices": files}

    src.close()
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Dataset prepared at {out}: {manifest['splits']}")
    return manifest
