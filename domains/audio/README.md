# Audio Pipeline

WAV/MP3 → feature extraction (MFCC, mel-spectrogram, chroma) → environmental sound classification demo

## Overview

An audio preprocessing and classification pipeline built on librosa and scikit-learn. The pipeline handles audio loading, resampling, feature extraction (MFCC, mel-spectrogram, chroma, spectral contrast, tonnetz), and aggregation to fixed-length vectors. The demo model is a Random Forest classifier trained on ESC-50 (50 environmental sound classes).

## Pipeline

```
WAV/MP3 → load (librosa, 22050 Hz, mono) → extract features:
  MFCC (40), mel-spectrogram (128), chroma (12), spectral contrast (7), tonnetz (6)
→ aggregate (mean + std per coefficient) → 130-dim feature vector
→ z-score normalize → train/val/test split
```

## Demo Model

- **Architecture:** Random Forest (200 trees, balanced class weights)
- **Dataset:** ESC-50 (2,000 environmental sound clips, 50 classes, ~600MB)
- **Classes:** airplane, breathing, cat, dog, rain, siren, thunderstorm, wind, etc.
- **Training:** Seconds on CPU (RF)
- **Metrics:** Accuracy, per-class F1, confusion matrix

## Quickstart

```bash
# Install dependencies
pip install librosa soundfile scikit-learn gradio matplotlib

# Download ESC-50 and train
python -m domains.audio.train --download-esc50

# Run inference on a single audio file
python -c "from domains.audio.infer import predict; print(predict('path/to/audio.wav'))"

# Launch the Gradio demo
python -m domains.audio.demo
```

## Using Your Own Audio Data

Organize audio files by class (one subfolder per class):

```
my_audio/
├── class_a/
│   ├── sample1.wav
│   └── sample2.wav
├── class_b/
│   └── sample3.wav
```

```python
from domains.audio.train import train
train(audio_dir="my_audio/")
```

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | Audio loading, feature extraction, dataset preparation |
| `model.py` | Random Forest classifier + save/load |
| `train.py` | ESC-50 download + training loop |
| `infer.py` | Single-file inference |
| `demo.py` | Gradio web interface (waveform + spectrogram + prediction) |
| `tests/` | Unit tests for feature extraction and model |

## Credits

- Feature extraction approach inspired by [danilodsp/AFX](https://github.com/danilodsp/AFX) (MIT)
- Dataset prep concepts from [MaxHilsdorf/SLAPP](https://github.com/MaxHilsdorf/single_label_audio_processing_pipeline) (MIT)
- ESC-50 dataset by [Karol Piczak](https://github.com/karoldvl/ESC-50) (CC)

## License

MIT
