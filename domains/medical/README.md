# Medical Imaging Pipeline

DICOM/NIfTI → preprocessed volumes → MedNIST classification demo

## Overview

A medical imaging preprocessing and classification pipeline built on SimpleITK, pydicom, and MONAI. The pipeline handles DICOM/NIfTI ingestion, HU windowing, resampling, and normalization. The demo model is a U-Net-style classifier trained on MedNIST (6 radiograph classes).

## Pipeline

```
DICOM/NIfTI → pixel decode (pydicom) → photometric normalization
→ HU windowing (-1024 to 1600) → resample (SimpleITK, 1mm isotropic)
→ normalize [0,1] → center crop/pad (64×64×64) → preprocessed .npy
```

## Demo Model

- **Architecture:** U-Net-style encoder + classification head (SimpleUNetClassifier)
- **Dataset:** MONAI MedNIST (47,164 hand radiographs, 6 classes)
- **Classes:** AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT
- **Training:** 2,000-image subset, 5 epochs, CPU (~3 min)
- **Result:** 99.33% validation accuracy

## Quickstart

```bash
# Install dependencies
pip install SimpleITK pydicom monai torch torchvision gradio

# Train the model
python -m domains.medical.train --epochs 5 --subset 2000

# Run inference on a single image
python -c "from domains.medical.infer import predict; print(predict('path/to/image.png'))"

# Launch the Gradio demo
python -m domains.medical.demo
```

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | DICOM/NIfTI preprocessing (load, HU window, resample, normalize) |
| `model.py` | U-Net classifier architecture + checkpoint loading |
| `train.py` | Training loop on MedNIST with metrics logging |
| `infer.py` | Single-image inference |
| `demo.py` | Gradio web interface |
| `tests/` | Unit tests for pipeline and model |

## Credits

- Preprocessing approach inspired by [DH82/medip](https://github.com/DH82/medip) (MIT)
- Dataset and transforms from [MONAI](https://github.com/Project-MONAI/MONAI) (Apache-2.0)
- MedNIST dataset by Project MONAI

## License

MIT
