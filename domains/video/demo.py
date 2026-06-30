"""Gradio Blocks demo for the video analysis pipeline.

Multi-tab portfolio surface for the zero-shot video tagger:
  1. Try the Demo — scene detection + keyframe tags
  2. Data & Preprocessing — scene detection, keyframe sampling, dedup
  3. Model & Inference — CLIP ViT-Base-Patch32 (no training)
  4. Evaluation — confidence distribution chart on the demo clip
  5. Code Walkthrough — the actual functions that ran
  6. Lessons Learned — zero-shot framing and production design

Run locally:
    python -m domains.video.demo
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

from .infer import analyze_video

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "outputs"
SAMPLES_DIR = HERE / "samples"


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def run_inference(video_path):
    if video_path is None:
        return None, "Please upload a video file."
    try:
        manifest = analyze_video(video_path, OUTPUT_DIR, classify=True)
    except Exception as e:
        return None, f"Error: {e}"

    n_scenes = manifest["n_scenes"]
    n_keyframes = manifest["n_keyframes"]
    classifications = manifest.get("classifications", [])

    if classifications:
        n_show = min(len(classifications), 8)
        ncols = 4
        nrows = int(np.ceil(n_show / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = np.atleast_1d(axes).ravel()
        for i in range(n_show):
            c = classifications[i]
            img = Image.open(c["path"])
            axes[i].imshow(img)
            top_label, top_score = c["labels"][0]
            axes[i].set_title(f"{top_label}\n({top_score:.1%})", fontsize=9)
            axes[i].axis("off")
        for i in range(n_show, len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No keyframes extracted", ha="center", va="center")
        ax.axis("off")

    summary = f"Scenes detected: {n_scenes}\nKeyframes extracted: {n_keyframes}\n\n"
    if classifications:
        summary += "Top tags per keyframe:\n"
        for i, c in enumerate(classifications[:10]):
            top_label, top_score = c["labels"][0]
            summary += f"  Scene {i+1}: {top_label} ({top_score:.1%})\n"
    return _fig_to_pil(fig), summary


def render_demo_confidence_chart():
    """Server-rendered chart: top-3 CLIP confidences on the included demo_scenes.mp4.

    Runs analyze_video on cold-start, plots per-scene top-3 tag confidences as a grouped bar chart.
    Falls back to a static "no demo available" image if the sample file is missing or analysis fails.
    """
    demo_video = SAMPLES_DIR / "demo_scenes.mp4"
    if not demo_video.exists():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No demo video found", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _fig_to_pil(fig)

    try:
        manifest = analyze_video(str(demo_video), OUTPUT_DIR, classify=True)
        classifications = manifest.get("classifications", [])
        if not classifications:
            raise RuntimeError("No classifications returned")

        n_scenes = len(classifications)
        scene_labels = [f"Scene {i+1}" for i in range(n_scenes)]
        # Top-3 per scene
        top3_data = []
        for c in classifications:
            top3 = c["labels"][:3]
            top3_data.append(top3)

        # Grouped bar chart
        fig, ax = plt.subplots(figsize=(11, 6))
        bar_width = 0.25
        x = np.arange(n_scenes)
        colors = ["#0279EE", "#FF9400", "#75A025"]
        for rank in range(3):
            heights = [td[rank][1] if rank < len(td) else 0 for td in top3_data]
            labels_at_rank = [td[rank][0] if rank < len(td) else "" for td in top3_data]
            offset = (rank - 1) * bar_width
            bars = ax.bar(x + offset, heights, bar_width,
                          label=f"Rank {rank+1}", color=colors[rank], alpha=0.85)
            # Annotate bars with the label
            for bar, lbl in zip(bars, labels_at_rank):
                if bar.get_height() > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            lbl, ha="center", va="bottom", fontsize=8, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels(scene_labels, fontsize=11)
        ax.set_ylabel("CLIP confidence", fontsize=12)
        ax.set_ylim(0, 1.10)
        ax.set_title("Zero-shot CLIP top-3 confidences — demo_scenes.mp4", fontsize=13)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        return _fig_to_pil(fig)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, f"Chart unavailable on cold start.\nUpload a video on the 'Try the Demo' tab.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        return _fig_to_pil(fig)


def _get_examples():
    if not SAMPLES_DIR.exists():
        return None
    samples = sorted(SAMPLES_DIR.glob("*.mp4"))
    return [[str(s)] for s in samples] if samples else None


DATA_AND_PREPROCESSING_MD = """
## Data & Preprocessing

### There is no training data

Let that sink in. The model is shipped pretrained (`openai/clip-vit-base-patch32` from HuggingFace),
and our pipeline calls it. The "data step" is just the input video.

For the live demo, we ship one sample MP4 (`demo_scenes.mp4`, 55 KB, 4 scenes). It exists so the
Space has something to show on a cold visit; in production, the user's video is the data.

### Where the engineering actually goes

Three preprocessing steps that have to be right for the CLIP tagger to give useful answers:

#### Step 1: Scene detection

PySceneDetect's `AdaptiveDetector` watches frame-to-frame luminance differences and flags
transitions. Used instead of fixed-interval sampling because most videos don't have evenly
distributed content.

```python
from scenedetect import SceneManager, AdaptiveDetector, open_video

video = open_video("input.mp4")
scene_manager = SceneManager()
scene_manager.add_detector(AdaptiveDetector())
scene_manager.detect_scenes(video)
scene_list = scene_manager.get_scene_list()
# scene_list: [(start_time, end_time), ...]
```

#### Step 2: Keyframe extraction

For each scene, grab the midpoint frame with OpenCV. The midpoint heuristic avoids transition
artifacts (motion blur at scene boundaries) and is faster than averaging frames.

```python
import cv2
cap = cv2.VideoCapture("input.mp4")
for start, end in scene_list:
    mid_ms = (start.get_seconds() + end.get_seconds()) / 2 * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, mid_ms)
    ret, frame = cap.read()
```

#### Step 3: Deduplication

Two scenes can contain visually identical content (a returning logo card, a static text frame,
a recurring interviewee). We hash each keyframe with `imagehash.phash()` and drop near-duplicates
by Hamming distance.

```python
import imagehash
from PIL import Image

threshold = 8  # Hamming distance, empirical
hashes, unique_frames = [], []
for frame in keyframes:
    h = imagehash.phash(Image.fromarray(frame))
    if not any(abs(h - existing) < threshold for existing in hashes):
        hashes.append(h)
        unique_frames.append(frame)
```

The threshold of 8 is empirical — close enough to "near-duplicate" without false positives on
legitimately similar scenes. Tune it for your content.
"""

MODEL_AND_INFERENCE_MD = """
## Model & Inference (no training)

### CLIP ViT-Base-Patch32

The "training" was done by OpenAI on 400 million image-text pairs. We're consumers.

```python
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

### Zero-shot scoring

```python
# Define candidate tags as text prompts
labels = ["landscape", "text", "sky", "person", "indoor scene",
          "vehicle", "animal", "food", "abstract"]
prompts = [f"a photo of {label}" for label in labels]

inputs = processor(text=prompts, images=keyframes, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)  # (n_frames, n_labels)
```

### Prompt template matters

The `"a photo of {label}"` template is doing real work. CLIP was trained on caption-image pairs,
so prompts that read like captions outperform bare class labels by ~5-10% accuracy in published
benchmarks. This is a small but real effect.

### Footprint

| Asset | Size |
|:---|---:|
| CLIP ViT-B/32 weights | ~150 MB |
| Cold start (load weights) | ~5 s |
| Inference per frame | ~0.1 s |
| End-to-end on a 4-scene clip | ~6 s |

### What CLIP knows (and doesn't)

| Concept | CLIP performance |
|:---|:---|
| General scenes (landscape, indoor, urban) | Strong |
| Common objects (car, dog, building) | Strong |
| Coarse activities (riding, eating, sleeping) | Reasonable |
| Fine-grained species (Bernese vs Saint Bernard) | Poor |
| Brand-specific (Coke vs Pepsi cans) | Inconsistent |
| OCR (text in image) | Poor — use a real OCR model |
"""

CODE_WALKTHROUGH_MD = """
## Code Walkthrough

### The four files that matter

| File | Role |
|:---|:---|
| `pipeline.py` | Scene detection + keyframe extraction + dedup |
| `model.py` | CLIP wrapper + default prompt list |
| `infer.py` | `analyze_video(path, output_dir, classify=True)` orchestrates everything |
| `demo.py` | ← This Gradio app |

### Inference path (what runs in this demo)

```python
from domains.video.infer import analyze_video

manifest = analyze_video("clip.mp4", output_dir="outputs/", classify=True)
# manifest = {
#   "n_scenes": 4,
#   "n_keyframes": 4,
#   "classifications": [
#       {"path": "outputs/scene_001.jpg",
#        "labels": [("landscape", 0.408), ("text", 0.192), ...]},
#       ...
#   ]
# }
```

### Repository layout

```
domains/video/
├── pipeline.py          # Scene detect + keyframe + dedup
├── model.py             # CLIP wrapper
├── infer.py             # analyze_video() orchestrator
├── demo.py              # ← This Gradio app
├── samples/             # 1 sample MP4 (demo_scenes.mp4)
└── outputs/             # Per-run keyframe + manifest writes
```

No `models/` directory — no model is trained or stored here. CLIP weights are downloaded
on first use to the HuggingFace cache.

### Full source

[github.com/fjkiani/ai-training/tree/main/domains/video](https://github.com/fjkiani/ai-training/tree/main/domains/video)
"""

LESSONS_LEARNED_MD = """
## Lessons Learned

### What we'd keep

- **Zero-shot CLIP as the first-pass classifier.** No labeled data needed.
  Customer onboarding is instant.
- **Prompt templates over bare labels.** `"a photo of X"` reliably beats `"X"` — small effort,
  measurable lift.
- **Confidence as a ranking, not a probability.** CLIP's softmax-over-prompts doesn't produce
  true probabilities. Treat the numbers as relative within a video, not as universal confidence
  scores.

### What we'd change for a paying customer

- **Content moderation:** Use zero-shot CLIP for fast first-pass filtering ("safe" vs "potentially
  flagged"), then route flagged content to a fine-tuned model for the second pass.
- **Brand safety:** Build a prompt library specific to the brand's risk vocabulary. CLIP handles
  it; we wrap it in a confidence-threshold workflow + human review for low-confidence outputs.
- **Archive search:** Index every scene's CLIP embedding (not just the tag), enable semantic search.
  Cosine similarity on 512-dim CLIP vectors is fast and gives you "find me scenes that look like
  this" out of the box.
- **Fine-grained tagging:** Switch to DINOv2 embeddings + a small linear probe trained on your
  labeled examples. CLIP is too general for "is this a Bernese Mountain Dog".

### The single biggest gotcha

**Confidence calibration.** A CLIP score of 0.97 on one scene and 0.31 on another doesn't mean
the model is "more sure" about the first; it means the first scene matches one prompt much better
relative to the alternatives in your label list. Add an unrelated prompt to the list and the
0.97 number shifts.

The fix: define a confidence threshold *empirically* on a small validation set per customer,
not based on what "looks high".

### Where zero-shot CLIP breaks

- **Domain-specific concepts** (medical pathology, satellite imagery, fashion). Fine-tune on your
  domain's image-text pairs or switch to a domain-specific encoder.
- **Adversarial prompts.** Adding a clever competing prompt can pull confidence from a different
  concept. Prompt engineering is real engineering work here.
- **Long-form video.** Per-scene tagging doesn't capture story-level structure. For that, you
  want temporal models (TimeSformer, Video-LLaVA).

### Blog post

Full long-form writeup including the moderation workflow:
[jedilabs.org/blog/video-zero-shot-tagging-with-clip](https://jedilabs.org/blog/video-zero-shot-tagging-with-clip)
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Video — Scene Detection + CLIP Tagging",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Video — Scene Detection + Zero-Shot Tagging\n"
            "PySceneDetect → keyframe extraction → CLIP ViT-Base-Patch32 zero-shot classification. "
            "**No training. No labeled data.** End-to-end inference on a 4-scene clip in ~6 seconds.\n\n"
            "**Portfolio:** [jedilabs.org/ai-training/video](https://jedilabs.org/ai-training/video) · "
            "**Code:** [github.com/fjkiani/ai-training](https://github.com/fjkiani/ai-training/tree/main/domains/video)"
        )

        with gr.Tabs():
            with gr.Tab("Try the Demo"):
                with gr.Row():
                    with gr.Column():
                        inp = gr.Video(label="Upload a video (MP4)")
                        btn = gr.Button("Analyze", variant="primary")
                        examples = _get_examples()
                        if examples:
                            gr.Examples(examples=examples, inputs=inp, label="Sample clips")
                    with gr.Column():
                        out_img = gr.Image(label="Keyframes + Tags")
                        out_text = gr.Textbox(label="Summary", lines=8)
                btn.click(run_inference, inputs=inp, outputs=[out_img, out_text])

            with gr.Tab("Data & Preprocessing"):
                gr.Markdown(DATA_AND_PREPROCESSING_MD)

            with gr.Tab("Model & Inference"):
                gr.Markdown(MODEL_AND_INFERENCE_MD)

            with gr.Tab("Evaluation"):
                gr.Markdown(
                    "## Zero-shot confidence — demo_scenes.mp4\n"
                    "Rendered server-side on cold start by running the same `analyze_video()` "
                    "pipeline this demo uses, against the included sample MP4."
                )
                gr.Image(value=render_demo_confidence_chart(), label="Top-3 CLIP confidences",
                         show_label=False)
                gr.Markdown(
                    "### What this chart shows\n\n"
                    "- **Each cluster of 3 bars is one scene from the sample clip.**\n"
                    "- **Rank 1 (blue)**, **Rank 2 (orange)**, **Rank 3 (green)** are the top-3 "
                    "CLIP labels for that scene's midpoint keyframe.\n"
                    "- **Wide gap between Rank 1 and Rank 2** = CLIP is confident.\n"
                    "- **Narrow gap** = CLIP is uncertain; flag for human review in production.\n\n"
                    "### Why this matters for production\n\n"
                    "There is no `metrics.json` for this pipeline. We didn't train a model. The way "
                    "to evaluate zero-shot CLIP is to feed it known content and verify the rankings "
                    "make sense — and to set a confidence threshold below which results get routed "
                    "to a human reviewer. The numbers above are your starting point for that "
                    "threshold."
                )

            with gr.Tab("Code Walkthrough"):
                gr.Markdown(CODE_WALKTHROUGH_MD)

            with gr.Tab("Lessons Learned"):
                gr.Markdown(LESSONS_LEARNED_MD)

        gr.Markdown(
            "---\n"
            "Built by [Jedi Labs](https://jedilabs.org). "
            "If your team has a video understanding problem — "
            "[book a discovery call](https://jedilabs.org/contact)."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
