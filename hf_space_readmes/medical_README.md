---
title: AI Medical Imaging Classifier
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Medical Imaging — MedNIST Classifier

A 6-class radiograph classifier built with MONAI + PyTorch. **99.3% peak validation accuracy** on
MedNIST in 5 epochs of CPU training.

## What this Space contains

A full multi-tab portfolio for the medical imaging domain. Visit the tabs in order:

1. **Try the Demo** — upload a radiograph or use one of the included samples
2. **Data & Preprocessing** — DICOM/NIfTI handling via MONAI transforms
3. **Model & Training** — small CNN architecture, hyperparameters, training rationale
4. **Evaluation** — per-epoch training curve with the full epoch-4 wobble story
5. **Code Walkthrough** — the four files that matter
6. **Lessons Learned** — design choices, gotchas, real-customer differences

## Headline metrics

| Metric | Value |
|:---|:---:|
| Best validation accuracy | **99.3%** |
| Last-epoch validation accuracy | 91.0% |
| Training time (5 epochs, CPU) | <10 minutes |
| Model parameters | ~24K |
| Inference latency | ~1 second per image |
| Classes | 6 (AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT) |
| Dataset | MedNIST (58,954 images) |

## The honest story

Validation accuracy peaks at **99.3% (epoch 2)** and drops to **75.7% (epoch 4)** before
recovering. Classic small-dataset oscillation. We save the best-checkpoint weights, not the
last-epoch weights — that's why the deployed model is 99.3%, not 91%. The Evaluation tab
visualizes the full curve.

## Related links

- **Portfolio page:** [jedilabs.org/ai-training/medical](https://jedilabs.org/ai-training/medical)
- **Source code:** [github.com/fjkiani/ai-training/tree/main/domains/medical](https://github.com/fjkiani/ai-training/tree/main/domains/medical)
- **Blog post:** [jedilabs.org/blog/medical-dicom-to-classification](https://jedilabs.org/blog/medical-dicom-to-classification)
- **Built by:** [Jedi Labs](https://jedilabs.org)

If your team is staring down a DICOM pipeline,
[book a discovery call](https://jedilabs.org/contact).
