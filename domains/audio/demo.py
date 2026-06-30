"""Gradio Blocks demo for the audio classification pipeline.

Multi-tab portfolio surface for the ESC-50 Random Forest:
  1. Try the Demo — waveform + mel-spec + prediction
  2. Data & Preprocessing — ESC-50 + the 130-dim feature vector
  3. Model & Training — Random Forest hyperparameters and rationale
  4. Evaluation — per-class F1 chart (server-rendered from metrics.json)
  5. Code Walkthrough — the actual functions that ran
  6. Lessons Learned — RF vs CNN tradeoff, production framing

Run locally:
    python -m domains.audio.demo
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
import librosa
import librosa.display
from PIL import Image

from .infer import predict
from .pipeline import load_audio, get_mel_spectrogram_image

HERE = Path(__file__).parent
CKPT = HERE / "models" / "audio_rf.pkl"
LE = HERE / "models" / "label_encoder.pkl"
METRICS = HERE / "models" / "metrics.json"
SAMPLES_DIR = HERE / "samples"


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def run_inference(audio_path):
    if audio_path is None:
        return None, None, "Please upload an audio file."
    if not CKPT.exists():
        return None, None, f"No trained model found at {CKPT}. Run training first: python -m domains.audio.train --download-esc50"

    result = predict(audio_path)
    y, sr = load_audio(audio_path)

    fig1, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("")
    plt.tight_layout()

    mel = get_mel_spectrogram_image(audio_path)
    fig2, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title("Mel-Spectrogram")
    fig2.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    summary = f"Predicted: {result['predicted_class']} ({result['confidence']:.1%})\n\nTop-5:\n"
    for name, prob in result["top5"]:
        summary += f"  {name}: {prob:.1%}\n"
    return _fig_to_pil(fig1), _fig_to_pil(fig2), summary


def render_per_class_f1():
    """Server-rendered per-class F1 chart from metrics.json."""
    if not METRICS.exists():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No metrics.json found", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _fig_to_pil(fig)
    m = json.loads(METRICS.read_text())
    report = m.get("report", {})
    class_names = m.get("class_names", [])

    # Pull per-class F1 by integer class id (0..49)
    f1_scores = []
    labels = []
    for i, cn in enumerate(class_names):
        key = str(i)
        if key in report:
            f1_scores.append(report[key]["f1-score"])
            labels.append(cn[:18])  # truncate long labels

    # Sort by F1 ascending so worst classes are visible at top
    order = np.argsort(f1_scores)
    f1_sorted = [f1_scores[i] for i in order]
    labels_sorted = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 12))
    bars = ax.barh(range(len(labels_sorted)), f1_sorted,
                    color=["#0279EE" if v >= 0.5 else "#FF9400" for v in f1_sorted])
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=9)
    ax.set_xlabel("F1 score", fontsize=12)
    ax.set_xlim(0, 1.0)

    macro_f1 = m.get("report", {}).get("macro avg", {}).get("f1-score", 0)
    weighted_f1 = m.get("report", {}).get("weighted avg", {}).get("f1-score", 0)
    ax.axvline(macro_f1, color="#75A025", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Macro F1: {macro_f1:.3f}")
    ax.axvline(weighted_f1, color="#FD9BED", linestyle=":", linewidth=1.5, alpha=0.7,
                label=f"Weighted F1: {weighted_f1:.3f}")
    ax.set_title("Per-class F1 on ESC-50 (sorted, best at top)\nBlue ≥ 0.5 · Orange < 0.5",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return _fig_to_pil(fig)


def _get_examples():
    if not SAMPLES_DIR.exists():
        return None
    samples = sorted(SAMPLES_DIR.glob("*.wav"))
    return [[str(s)] for s in samples] if samples else None


DATA_AND_PREPROCESSING_MD = """
## Data & Preprocessing

### What we feed the model

**Dataset:** [ESC-50](https://github.com/karolpiczak/ESC-50) — 2000 environmental sound clips,
5 seconds each, 50 classes (40 clips per class). Balanced. Curated. Honest.

Classes span animals (dog, cat, rooster), human sounds (breathing, sneezing, clapping),
interior sounds (door, vacuum, glass break), exterior sounds (rain, helicopter, engine),
and natural sounds (sea waves, thunderstorm, wind).

Train/val/test split: 1700/300/300 (85/15, then val split off of train). Deterministic seed.

### The 130-dimensional feature vector

Five complementary signal-processing views:

```python
def extract_features(y, sr=22050):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).mean(axis=1)              # 40 dims
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)                # 12 dims
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)        # 7 dims
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1)                   # 6 dims
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=65).mean(axis=1)     # 65 dims
    return np.concatenate([mfcc, chroma, contrast, tonnetz, mel])                # 130 dims
```

| Feature | Dim | What it captures |
|:---|:---:|:---|
| MFCC | 40 | Spectral envelope — universal audio feature |
| Chroma | 12 | Pitch class energy — useful for tonal sounds |
| Spectral contrast | 7 | Peak-vs-valley energy in frequency bands |
| Tonnetz | 6 | Harmonic relations — useful for music |
| Mel-spec stats | 65 | Per-mel-band mean energy |
| **Total** | **130** | |

### The tradeoff

Each `.mean(axis=1)` **collapses the time dimension**. We're throwing away "*when* in the 5 seconds
did the sound happen". That's fine for stationary environmental sounds (rain, wind, sea waves) and
catastrophic for sounds that depend on temporal structure (single bark vs continuous bark, speech).

### Sample rate

22050 Hz, mono. Standard for environmental sound work. ESC-50 ships at 44100 Hz; we downsample on
load via `librosa.load(path, sr=22050, mono=True)`.
"""

MODEL_AND_TRAINING_MD = """
## Model & Training

### Architecture

`sklearn.ensemble.RandomForestClassifier` with 200 trees.

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)  # X_train: (1700, 130), y_train: (1700,)
```

That's the whole model. No GPU. No architecture search. No learning-rate schedule.

### Why Random Forest (not 1D CNN)

| Property | Random Forest | 1D CNN |
|:---|:---|:---|
| Trains in | <1 minute | 10–60 minutes |
| Hardware | CPU | GPU recommended |
| Interpretability | `feature_importances_` per class | Activations + saliency maps |
| Hyperparams | 1 that matters (`n_estimators`) | LR, batch, depth, regularization, schedule |
| Data needed | 1000+ samples | 10,000+ samples |
| Ceiling on ESC-50 | ~65% test acc | ~85-95% test acc (state-of-the-art) |

For a 2000-sample dataset where iteration speed matters more than headline accuracy, RF wins.
For a production system needing >75% accuracy, you'd switch to a 1D CNN (or a pretrained
audio transformer like AST or PANNs).

### Training setup

| Hyperparameter | Value |
|:---|:---|
| n_estimators | 200 |
| max_depth | None (unlimited) |
| min_samples_split | 2 (default) |
| Random seed | 42 |
| Train/val/test | 1700/300/300 |

### Output

- `audio_rf.pkl` — Pickled RandomForestClassifier (~5 MB)
- `label_encoder.pkl` — sklearn LabelEncoder mapping integer IDs ↔ class names
- `metrics.json` — `val_acc`, `test_acc`, plus per-class `precision/recall/f1-score/support`
"""

CODE_WALKTHROUGH_MD = """
## Code Walkthrough

### The four files that matter

| File | Role |
|:---|:---|
| `pipeline.py` | Feature extraction — `extract_features(y, sr)` |
| `model.py` | Random Forest wrapper + class constants |
| `train.py` | Training loop + per-class metrics → `metrics.json` |
| `infer.py` | `predict(audio_path)` returns `{predicted_class, confidence, top5}` |

### Inference path (what runs in this demo)

```python
from domains.audio.infer import predict

result = predict("clip.wav")
# result = {
#   "predicted_class": "dog",
#   "confidence": 0.42,
#   "top5": [("dog", 0.42), ("cat", 0.18), ...]
# }
```

### Repository layout

```
domains/audio/
├── pipeline.py          # Feature extraction
├── model.py             # RF wrapper
├── train.py             # Training loop
├── infer.py             # predict() function
├── demo.py              # ← This Gradio app
├── samples/             # 8 sample WAVs across diverse classes
└── models/
    ├── audio_rf.pkl     # Trained RF (~5 MB)
    ├── label_encoder.pkl
    └── metrics.json     # val/test accuracy + per-class metrics
```

### Full source

[github.com/fjkiani/ai-training/tree/main/domains/audio](https://github.com/fjkiani/ai-training/tree/main/domains/audio)
"""

LESSONS_LEARNED_MD = """
## Lessons Learned

### What we'd keep

- **Random Forest as the baseline** for any new audio classification problem under 5000 samples.
  30× chance on 50 classes is shippable for a lot of real use cases.
- **Feature engineering over feature learning** at this data scale. Spend more time on `librosa`
  calls than on hyperparameter search.
- **Per-class F1 as the primary metric.** Headline accuracy hides class imbalance and per-class
  failures — the Evaluation tab shows why.

### What we'd change for a paying customer

- **Use case dictates model.**
  - **Call center QA:** Random Forest is enough. Classes are coarse (silence, single voice, two
    voices, raised voice, hold music) and data is plentiful.
  - **Industrial anomaly detection:** Switch to 1D CNN on spectrogram patches. Anomalies are
    subtle and temporal.
  - **Wake-word / keyword spotting:** Use a pre-trained transformer like Wav2Vec2 or AST.
    Feature engineering is wasted effort at that point.
- **Add temporal features.** When `.mean(axis=1)` is the bottleneck, add `.std(axis=1)`,
  `.max(axis=1)`, or short-time frame statistics. Or stop collapsing entirely and feed
  spectrograms to a CNN.
- **Calibrate the class probabilities.** Random Forest votes aren't well-calibrated; use
  `CalibratedClassifierCV` if downstream code uses confidence thresholds.

### The single biggest gotcha

**Class imbalance pretending to be balance.** ESC-50 is officially balanced (40 clips/class), but
some classes are acoustically near-identical (rain vs sea waves) and others are acoustically
unique (helicopter). The model finds the unique classes trivial and the similar ones impossible.
Headline accuracy obscures this; per-class F1 reveals it. See the Evaluation tab.

### Expected real-world ranges

| Task | Achievable accuracy |
|:---|:---:|
| ESC-50 with RF (this demo) | 60–65% |
| ESC-50 with 1D CNN | 75–82% |
| ESC-50 with AST / BEATs | 85–95% |
| 5-class call center events | 90–97% |
| Industrial anomaly (binary) | 92–99% |

The right model is the simplest one that hits the accuracy bar your business case requires.

### Blog post

Full long-form writeup including the 1D-CNN comparison:
[jedilabs.org/blog/audio-features-to-rf-classifier](https://jedilabs.org/blog/audio-features-to-rf-classifier)
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Audio — ESC-50 Environmental Sound Classifier",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Audio — Environmental Sound Classification\n"
            "A Random Forest classifier on 130-dimensional `librosa` features across 50 ESC-50 classes. "
            "**60.3% test accuracy** vs. **2% random chance** — a 30× lift, trained in seconds on CPU.\n\n"
            "**Portfolio:** [jedilabs.org/ai-training/audio](https://jedilabs.org/ai-training/audio) · "
            "**Code:** [github.com/fjkiani/ai-training](https://github.com/fjkiani/ai-training/tree/main/domains/audio)"
        )

        with gr.Tabs():
            with gr.Tab("Try the Demo"):
                with gr.Row():
                    with gr.Column():
                        inp = gr.Audio(type="filepath", label="Upload an audio file (WAV/MP3)")
                        btn = gr.Button("Classify", variant="primary")
                        examples = _get_examples()
                        if examples:
                            gr.Examples(examples=examples, inputs=inp, label="Sample clips")
                    with gr.Column():
                        out_wave = gr.Image(label="Waveform")
                        out_mel = gr.Image(label="Mel-Spectrogram")
                        out_text = gr.Textbox(label="Prediction", lines=8)
                btn.click(run_inference, inputs=inp, outputs=[out_wave, out_mel, out_text])

            with gr.Tab("Data & Preprocessing"):
                gr.Markdown(DATA_AND_PREPROCESSING_MD)

            with gr.Tab("Model & Training"):
                gr.Markdown(MODEL_AND_TRAINING_MD)

            with gr.Tab("Evaluation"):
                gr.Markdown(
                    "## Per-class F1 score\n"
                    "Rendered server-side from the actual `metrics.json` log "
                    "([file](https://github.com/fjkiani/ai-training/blob/main/domains/audio/models/metrics.json)). "
                    "Sorted ascending so the **worst-performing classes are at the top**."
                )
                gr.Image(value=render_per_class_f1(), label="Per-class F1", show_label=False)
                gr.Markdown(
                    "### What this chart shows\n\n"
                    "- **Test accuracy: 60.3%** (vs 2% random chance — a 30× lift)\n"
                    "- **Macro F1: 0.567**, **Weighted F1: 0.590** — the small gap means the model "
                    "isn't lopsided across classes.\n"
                    "- **The orange bars (F1 < 0.5)** are the classes the model struggles with — "
                    "mostly sounds with overlapping spectral content (e.g., several animal vocalizations) "
                    "or sounds whose distinguishing feature is temporal (where `.mean(axis=1)` discards "
                    "the signal).\n"
                    "- **The blue bars (F1 ≥ 0.5)** are the classes that work well — typically the ones "
                    "with strong stationary spectral signatures (rain, sea waves, wind, helicopter).\n\n"
                    "Per-class F1 is the metric that matters. Headline accuracy hides this whole story."
                )

            with gr.Tab("Code Walkthrough"):
                gr.Markdown(CODE_WALKTHROUGH_MD)

            with gr.Tab("Lessons Learned"):
                gr.Markdown(LESSONS_LEARNED_MD)

        gr.Markdown(
            "---\n"
            "Built by [Jedi Labs](https://jedilabs.org). "
            "If your team has an audio classification problem — "
            "[book a discovery call](https://jedilabs.org/contact)."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
