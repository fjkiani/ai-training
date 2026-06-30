---
title: AI Zero-Shot Video Tagger
emoji: 🎬
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

# Video — Scene Detection + Zero-Shot Tagging

PySceneDetect → keyframe extraction → CLIP ViT-Base-Patch32 zero-shot classification.
**No training. No labeled data.** End-to-end inference on a 4-scene clip in ~6 seconds.

## What this Space contains

A full multi-tab portfolio for the video domain. Visit the tabs in order:

1. **Try the Demo** — upload a video or use the included sample MP4
2. **Data & Preprocessing** — scene detection, midpoint keyframe extraction, dedup via pHash
3. **Model & Inference** — CLIP ViT-B/32 architecture + prompt template rationale
4. **Evaluation** — server-rendered top-3 confidence chart on the included demo clip
5. **Code Walkthrough** — the four files that matter
6. **Lessons Learned** — production design (moderation, brand-safety, archive search)

## Headline metrics

| Metric | Value |
|:---|:---:|
| Training data needed | **0** |
| Training time | **0** |
| CLIP model | ViT-Base-Patch32 (~150 MB) |
| Cold start (load weights) | ~5 seconds |
| End-to-end inference (4-scene clip) | ~6 seconds |
| Sample MP4 scene count | 4 |
| Demo top-tag confidences | landscape 40.8%, text 31.9%, sky 97.7%, text 55.5% |

## The honest story

There is no `metrics.json`. We didn't train a model. The way to evaluate zero-shot CLIP is to
feed it known content and check the rankings make sense. The Evaluation tab does exactly that —
the included demo MP4 gives a ground-truth-free way to inspect the model's confidence
distribution.

**Confidence is a ranking, not a probability.** A score of 0.97 on one scene doesn't mean the
model is "more sure" in absolute terms — it means that scene matches one prompt much better
relative to the alternatives in the label list. Calibrate thresholds empirically per customer.

## Related links

- **Portfolio page:** [jedilabs.org/ai-training/video](https://jedilabs.org/ai-training/video)
- **Source code:** [github.com/fjkiani/ai-training/tree/main/domains/video](https://github.com/fjkiani/ai-training/tree/main/domains/video)
- **Blog post:** [jedilabs.org/blog/video-zero-shot-tagging-with-clip](https://jedilabs.org/blog/video-zero-shot-tagging-with-clip)
- **CLIP paper:** [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- **Built by:** [Jedi Labs](https://jedilabs.org)

If your team has a video understanding problem — moderation, brand-safety, archive search —
and you're staring down a labeling budget you don't have,
[book a discovery call](https://jedilabs.org/contact).
