# Pandemonium Rukopys

Ukrainian handwritten text recognition on [RUKOPYS](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys), an AI course project at UCU by team Pandemonium.

## Status

Midterm stage. The downloaded snapshot has been audited; see [data/eda.md](data/eda.md). Recognition has a supervised TrOCR baseline on ground-truth line crops. The 10-epoch detector baseline is trained; validation exposed duplicate boxes and weak localization. A separate fine-tuning experiment adds backbone adaptation, larger images, and auxiliary losses. Modern detector variants are now being compared under equal time budgets.

## Problem and approach

Turn a page image into content regions, region types, and Ukrainian handwritten text:

`page -> DETR detector/type classifier -> handwritten line crops -> TrOCR -> ordered text`

Compare a human-label-only baseline (A), existing silver-label training (B), and filtered self-training (C), keeping the detector fixed for recognition comparisons. Test CV improvements separately. Formula/table transcription is an extension.

Use about 10% of official training pages for grouped validation. Preserve the entire official test set for final evaluation; after settings are fixed, final comparison models may be refitted on all human training pages.

## Data and evaluation

The dataset declares CC BY 4.0; see [data/README.md](data/README.md) for attribution, privacy handling, and split policy.

Planned metrics: CER/WER, detection mAP, type macro-F1 with coverage, handwritten page CER, per-source results, and runtime. Full-task Kaggle scores require the official scoring protocol and must remain separate from handwritten-only metrics.

## Setup and results

The data audit can be reproduced with `uv run python scripts/eda.py PATH_TO_SNAPSHOT` after running `uv sync`. See [data/README.md](data/README.md) for setup, download instructions, and the pinned revision.

The recognition notebooks need the pinned snapshot (local `data/rukopys/` or Hugging Face cache) and an NVIDIA GPU with about 7 GB free. Run them in order from the repository root:

```sh
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/01_htr_pilot.ipynb
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/02_htr_baseline_train.ipynb
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/03_htr_eval.ipynb
```

`01_htr_pilot` writes the split ([data/splits/](data/splits/README.md)) and line crops (`data/htr_crops/`, not in Git), and compares checkpoints, tokenisation, and decoding. `02_htr_baseline_train` trains baseline A (about 1.5 hours on an RTX 5060 Laptop GPU) and saves weights to `models/trocr-uk-baseline/` (not in Git). `03_htr_eval` writes validation predictions and metrics to `results/htr/`. Shared helpers are in `src/htr.py`. They also find the relocated local snapshot under `data/rukopys/`.

The CV benchmark compares six pretrained detectors for 50 minutes each on the M3 Pro. **RF-DETR Small leads mAP (0.4251)** and is selected for longer refinement; **YOLO26 Small leads handwritten crop coverage (90.1%)** and remains an HTR comparison. See the [comparison table and next-step rationale](results/cv/benchmark/comparison.md).

```sh
uv run python -u scripts/benchmark_detectors.py --minutes 50 --resume-existing
uv run python scripts/benchmark_detectors.py --report
```

`--device gpu` selects CUDA → MPS → CPU. The budget is cumulative per candidate; loading and final evaluation are extra. `--resume-existing` preserves saved progress. For one candidate, add `--models rf-small`; for inference only, add `--evaluate-only`. Settings and scores are in `results/cv/benchmark/runs.json`. Earlier pilots are archived locally. Reproducing the original DETR reference uses `uv run python -u scripts/train_detector.py --device gpu`.

Weights, data and predicted crops stay outside Git. Best Transformer weights are in `models/benchmark-<model>/best/`; YOLO uses `weights/common_best.pt`. HTR-compatible crops are in `data/detector_crops/<model>/`. Evaluate the same recognizer on RF-DETR Small and YOLO crops, reporting CER/WER together with coverage and unmatched predictions. Official test remains reserved.

## Contribution statement

- **Oleh Hutsuliak** CV: image audit, detection/types, crops, augmentation, and CV metrics.
- **Oleksandr Stadnik** NLP/HTR: text audit, recognition/decoding, pseudo-label filtering, and CER/WER.
- **Shared equally throughout:** split design, integration, experiments, analysis, documentation, and presentations.
- **Agreement:** both agreed.

## AI usage

Codex assisted with writing and the EDA script. We guided and reviewed the earlier work, and ran the audit on the local snapshot.

Claude assisted with the recognition helpers and notebooks. We guided and reviewed the work and checked its results against the data.

Codex assisted with fine-tuning setup and timed detector benchmarks. We checked box coordinates, validation predictions, training/resume behavior.