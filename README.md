# AI Training Pipelines

A collection of domain-specific AI data preprocessing pipelines, each with a trained demo model and interactive Gradio interface.

## Domains

| Domain | Input | Output | Model | Demo |
|--------|-------|--------|-------|------|
| **Medical Imaging** | DICOM/NIfTI | Preprocessed volumes + classification | U-Net classifier (MedNIST, 99.3% acc) | Upload radiograph → predicted class |
| **Geospatial** | GeoTIFF | Tiled patches + segmentation masks | U-Net ResNet18 (land/water segmentation) | Upload satellite tile → land/water mask |
| **Audio** | WAV/MP3 | MFCC/mel-spec features + classification | Random Forest (ESC-50, 50 classes) | Upload audio → waveform + spectrogram + class |
| **Video** | MP4 | Scene list + keyframes + CLIP tags | CLIP zero-shot (no training) | Upload video → scenes + keyframes + tags |

## Quickstart

```bash
# Clone
git clone https://github.com/fjkiani/ai-training.git
cd ai-training

# Install system dependencies
apt-get install -y gdal-bin libgdal-dev ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Train each domain (or use pre-trained checkpoints)
python -m domains.medical.train --epochs 5 --subset 2000
python -m domains.geospatial.train --epochs 10
python -m domains.audio.train --download-esc50
# Video uses zero-shot CLIP — no training needed

# Launch any demo
python -m domains.medical.demo
python -m domains.geospatial.demo
python -m domains.audio.demo
python -m domains.video.demo
```

## Structure

```
ai-training/
├── shared/              # Common utilities (I/O, logging, splitting, viz)
├── domains/
│   ├── medical/         # DICOM/NIfTI pipeline + MedNIST classifier
│   ├── geospatial/      # GeoTIFF pipeline + land/water segmentation
│   ├── audio/           # Audio feature pipeline + ESC-50 classifier
│   └── video/           # Scene detection + CLIP zero-shot tagging
├── demos/               # HuggingFace Spaces deployment guide
└── requirements.txt
```

Each domain folder contains:
- `pipeline.py` — preprocessing (raw input → ML-ready output)
- `model.py` — model architecture
- `train.py` — training script
- `infer.py` — inference
- `demo.py` — Gradio web interface
- `tests/` — unit tests
- `README.md` — domain-specific documentation

## Testing

```bash
# Run all tests
python -m pytest domains/ -v

# Run a specific domain
python -m pytest domains/medical/ -v
```

## Deploying Demos to HuggingFace Spaces

See [`demos/huggingface_spaces.md`](demos/huggingface_spaces.md) for instructions on deploying each Gradio demo to HuggingFace Spaces (free tier).

## License

MIT
