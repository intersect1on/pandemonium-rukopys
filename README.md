# Pandemonium Rukopys

Ukrainian handwritten text recognition on [RUKOPYS](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys), an AI course project at UCU by team Pandemonium.

## Status

Data preparation stage. The downloaded snapshot has been inventoried and audited; see [data/eda.md](data/eda.md) for findings. Validation splitting and model training are next. No trained model or model-performance results are available yet.

## Problem and approach

Turn a page image into content regions, region types, and Ukrainian handwritten text:

`page -> DETR detector/type classifier -> handwritten line crops -> TrOCR -> ordered text`

Compare a human-label-only baseline (A), existing silver-label training (B), and filtered self-training (C), keeping the detector fixed for recognition comparisons. Test CV improvements separately. Formula/table transcription is an extension.

Use about 10% of official training pages for grouped validation. Preserve the entire official test set for final evaluation; after settings are fixed, final comparison models may be refitted on all human training pages.

## Data and evaluation

The dataset declares CC BY 4.0; see [data/README.md](data/README.md) for attribution, privacy handling, and split policy.

Planned metrics: CER/WER, detection mAP, type macro-F1 with coverage, handwritten page CER, per-source results, and runtime. Full-task Kaggle scores require the official scoring protocol and must remain separate from handwritten-only metrics.

## Setup and results

The data audit can be reproduced with `uv run python scripts/eda.py PATH_TO_SNAPSHOT` after running `uv sync`. See [data/README.md](data/README.md) for setup, download instructions, and the pinned revision. Training and model-evaluation commands are not implemented yet.

## Contribution statement

- **Oleh Hutsuliak** CV: image audit, detection/types, crops, augmentation, and CV metrics.
- **Oleksandr Stadnik** NLP/HTR: text audit, recognition/decoding, pseudo-label filtering, and CER/WER.
- **Shared equally throughout:** split design, integration, experiments, analysis, documentation, and presentations.
- **Agreement:** both agreed.

## AI usage

Codex assisted with writing and the EDA script. We guided and reviewed the earlier work, and ran the audit on the local snapshot.
