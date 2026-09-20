# Pandemonium Rukopys

Ukrainian handwritten text recognition on [RUKOPYS](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys), an AI course project at UCU by team Pandemonium.

## Status

Proposal stage. The dataset is downloaded; its local audit, model implementation, and experiments are pending. No trained model or measured results are available yet.

## Problem and approach

Turn a page image into content regions, region types, and Ukrainian handwritten text:

`page -> DETR detector/type classifier -> handwritten line crops -> TrOCR -> ordered text`

Compare a human-label-only baseline (A), existing silver-label training (B), and filtered self-training (C), keeping the detector fixed for recognition comparisons. Test CV improvements separately. Formula/table transcription is an extension.

Use about 10% of official training pages for grouped validation. Preserve the entire official test set for final evaluation; after settings are fixed, final comparison models may be refitted on all human training pages.

## Data and evaluation

The dataset declares CC BY 4.0; see [data/README.md](data/README.md) for attribution, privacy handling, and split policy.

Planned metrics: CER/WER, detection mAP, type macro-F1 with coverage, handwritten page CER, per-source results, and runtime. Full-task Kaggle scores require the official scoring protocol and must remain separate from handwritten-only metrics.

## Setup and results

No training or evaluation commands exist yet.

## Contribution statement

- **Oleh Hutsuliak** CV: image audit, detection/types, crops, augmentation, and CV metrics.
- **Oleksandr Stadnik** NLP/HTR: text audit, recognition/decoding, pseudo-label filtering, and CER/WER.
- **Shared equally throughout:** split design, integration, experiments, analysis, documentation, and presentations.
- **Agreement:** both agreed.

## AI usage

Codex assisted with writing. We guided it and reviewed all the work.
