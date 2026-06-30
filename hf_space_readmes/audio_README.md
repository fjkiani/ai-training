---
title: AI Environmental Sound Classifier
emoji: 🔊
colorFrom: purple
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# Audio — Environmental Sound Classification

A Random Forest classifier on 130-dimensional `librosa` features across 50 ESC-50 classes.
**60.3% test accuracy** vs. 2% random chance — a 30× lift, trained in seconds on CPU.

## What this Space contains

A full multi-tab portfolio for the audio domain. Visit the tabs in order:

1. **Try the Demo** — upload an audio clip or use one of the included samples
2. **Data & Preprocessing** — ESC-50 dataset + the 130-dim feature vector breakdown
3. **Model & Training** — Random Forest hyperparameters + RF-vs-CNN rationale
4. **Evaluation** — per-class F1 score chart (server-rendered from metrics.json)
5. **Code Walkthrough** — the four files that matter
6. **Lessons Learned** — temporal-feature tradeoffs, production framing per use case

## Headline metrics

| Metric | Value |
|:---|:---:|
| Test accuracy | **60.3%** |
| Validation accuracy | 60.0% |
| Random chance baseline | 2.0% (50 classes) |
| Lift over chance | **30×** |
| Macro F1 | 0.567 |
| Weighted F1 | 0.590 |
| Training time | <1 minute (CPU) |
| Model size | ~5 MB |
| Feature dimension | 130 |

## The honest story

A Random Forest on hand-engineered features is the *correct* baseline at this data scale (2000
samples). State-of-the-art on ESC-50 with audio transformers (AST, BEATs) is in the 85-95% range —
that's the ceiling if you have GPU + labeled data + research patience.

Where this model breaks: sounds whose distinguishing feature is **temporal** (single bark vs.
continuous bark) — because `.mean(axis=1)` over the spectrogram collapses time. The Evaluation
tab makes this visible per class.

## Related links

- **Portfolio page:** [jedilabs.org/ai-training/audio](https://jedilabs.org/ai-training/audio)
- **Source code:** [github.com/fjkiani/ai-training/tree/main/domains/audio](https://github.com/fjkiani/ai-training/tree/main/domains/audio)
- **Blog post:** [jedilabs.org/blog/audio-features-to-rf-classifier](https://jedilabs.org/blog/audio-features-to-rf-classifier)
- **Built by:** [Jedi Labs](https://jedilabs.org)

If your team has an audio classification problem — call center QA, field ops, security,
wildlife monitoring — [book a discovery call](https://jedilabs.org/contact).
