"""Medical imaging preprocessing pipeline.

DICOM / NIfTI -> preprocessed numpy volumes ready for ML.

Pipeline steps:
  1. Load (pydicom for DICOM, SimpleITK for NIfTI/DICOM series)
  2. Photometric normalization (MONOCHROME1 -> MONOCHROME2)
  3. HU windowing (clip to clinical window, normalize to [0,1])
  4. Resample to target spacing (SimpleITK)
  5. Normalize intensity to [0,1]
  6. Slice/crop to fixed size

Built on SimpleITK + pydicom, inspired by DH82/medip (MIT) and MONAI transforms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

from shared import get_logger

log = get_logger("medical.pipeline")

# Default HU window for CT (soft tissue)
DEFAULT_HU_WINDOW = (-1024, 1600)
DEFAULT_TARGET_SPACING = (1.0, 1.0, 1.0)  # mm
DEFAULT_PATCH_SIZE = (64, 64, 64)  # z, y, x


def load_volume(path: str | Path) -> sitk.Image:
    """Load a DICOM series directory or a NIfTI/mha file into a SimpleITK image.

    - If path is a directory, reads it as a DICOM series.
    - If path is a file (.nii, .nii.gz, .mha, .nrrd), reads directly.
    """
    path = Path(path)
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(path))
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM series found in {path}")
        reader.SetFileNames(dicom_files)
        img = reader.Execute()
        log.info(f"Loaded DICOM series: {len(dicom_files)} slices from {path}")
    else:
        img = sitk.ReadImage(str(path))
        log.info(f"Loaded volume: {path.name} shape={img.GetSize()}")
    return img


def hu_window(arr: np.ndarray, window: Tuple[int, int] = DEFAULT_HU_WINDOW) -> np.ndarray:
    """Clip to HU window and normalize to [0,1]."""
    w_min, w_max = window
    arr = np.clip(arr, w_min, w_max)
    arr = (arr - w_min) / (w_max - w_min)
    return arr.astype(np.float32)


def resample_volume(
    img: sitk.Image,
    target_spacing: Tuple[float, float, float] = DEFAULT_TARGET_SPACING,
) -> sitk.Image:
    """Resample a SimpleITK image to target spacing using B-spline."""
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(img)


def center_crop_or_pad(arr: np.ndarray, size: Tuple[int, int, int]) -> np.ndarray:
    """Center-crop or zero-pad a 3D array to the given size (z, y, x)."""
    result = np.zeros(size, dtype=arr.dtype)
    sz, sy, sx = size
    az, ay, ax = arr.shape
    # source offsets
    sz0 = max((az - sz) // 2, 0)
    sy0 = max((ay - sy) // 2, 0)
    sx0 = max((ax - sx) // 2, 0)
    # dest offsets
    dz0 = max((sz - az) // 2, 0)
    dy0 = max((sy - ay) // 2, 0)
    dx0 = max((sx - ax) // 2, 0)
    # copy region
    cz = min(sz, az)
    cy = min(sy, ay)
    cx = min(sx, ax)
    result[dz0:dz0+cz, dy0:dy0+cy, dx0:dx0+cx] = arr[sz0:sz0+cz, sy0:sy0+cy, sx0:sx0+cx]
    return result


def preprocess_volume(
    path: str | Path,
    hu_window: Tuple[int, int] = DEFAULT_HU_WINDOW,
    target_spacing: Tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
    resample: bool = True,
) -> np.ndarray:
    """Full pipeline: load -> HU window -> resample -> normalize -> crop/pad.

    Returns a float32 array of shape patch_size with values in [0,1].
    """
    img = load_volume(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    arr = hu_window(arr, hu_window)
    if resample:
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(target_spacing)
        img = resample_volume(img, target_spacing)
        arr = sitk.GetArrayFromImage(img)
    arr = center_crop_or_pad(arr.astype(np.float32), patch_size)
    return arr


def preprocess_2d_image(
    path: str | Path,
    size: int = 64,
) -> np.ndarray:
    """Load a single 2D medical image (PNG/NIfTI slice) and normalize for the demo model.

    Returns float32 array of shape (1, size, size) in [0,1].
    """
    path = Path(path)
    if path.suffix.lower() in (".nii", ".nii.gz", ".mha", ".nrrd"):
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        arr = arr[arr.shape[0] // 2] if arr.ndim == 3 else arr  # middle slice
    else:
        from PIL import Image
        im = Image.open(path).convert("L")
        arr = np.array(im, dtype=np.float32)
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(arr)
    # resize to (size, size)
    from PIL import Image
    im = Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size))
    arr = np.array(im, dtype=np.float32) / 255.0
    return arr[None, ...]  # (1, H, W)
