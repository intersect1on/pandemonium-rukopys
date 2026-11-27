# Pandemonium Rukopys

Ukrainian handwritten text recognition on [RUKOPYS](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys), an AI course project at UCU by team Pandemonium.

## Status

Midterm stage. The downloaded snapshot has been audited; see [data/eda.md](data/eda.md). Recognition has a supervised TrOCR baseline on ground-truth line crops. The 10-epoch detector baseline is trained; validation exposed duplicate boxes and weak localization. Six modern detector benchmarks are complete. RF-DETR Small refinement reached validation mAP 0.5105; early stopping retained epoch 12 after 17 epochs.

Final stage, silver fine-tuning (27 Nov): B and C are trained and scored on human validation. Silver-only training helps the sources it covers but forgets dictation. An exploratory run with human replay (M) lowers validation line CER from 0.1150 to 0.1046. The test comparison is still to come. [Silver results](results/htr/silver-finetune.md)

## Problem and approach

Turn a page image into content regions, region types, and Ukrainian handwritten text:

`page -> DETR detector/type classifier -> handwritten line crops -> TrOCR -> ordered text`

Recognition baseline A uses human-annotated handwritten line crops. B and C each fine-tune from A on v3.0 silver, using supplied transcripts for B and filtered teacher transcripts for C. The detector stays fixed for the recognition comparison. Detection covers seven region types and the recognizer transcribes handwriting. [Silver pool record](data/silver.md)

The v2.2 human split contains 1,197 training pages and 133 validation pages, stratified by source with seed 42. The corrected 382-page v3.0 test set is reserved for final evaluation.

## Data and evaluation

The dataset declares CC BY 4.0; see [data/README.md](data/README.md) for attribution, privacy handling, and split policy.

Planned metrics: CER/WER, detection mAP, type macro-F1 with coverage, handwritten page CER, per-source results, and runtime. Full-task Kaggle scores require the official scoring protocol and must remain separate from handwritten-only metrics.

## Setup and results

Run `uv sync` and `uv run python scripts/sync_rukopys.py` from the repository root. [Dataset setup and pinned revisions](data/README.md)

The recognition notebooks need the pinned snapshot (local `data/rukopys/` or Hugging Face cache) and an NVIDIA GPU with about 7 GB free. Run them in order from the repository root:

```sh
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/01_htr_pilot.ipynb
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/02_htr_baseline_train.ipynb
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/03_htr_eval.ipynb
```

`01_htr_pilot` writes the split ([data/splits/](data/splits/README.md)) and line crops (`data/htr_crops/`, not in Git), and compares checkpoints, tokenisation, and decoding. `02_htr_baseline_train` trains baseline A (about 1.5 hours on an RTX 5060 Laptop GPU) and saves weights to `models/trocr-uk-baseline/` (not in Git). `03_htr_eval` writes validation predictions and metrics to `results/htr/`. Shared helpers are in `src/htr.py`. They also find the relocated local snapshot under `data/rukopys/`.

The CV benchmark compares six pretrained detectors for 50 minutes each on the M3 Pro. **RF-DETR Small leads mAP (0.4251)** and is selected for longer refinement; **YOLO26 Small leads handwritten crop coverage (90.1%)** and remains an HTR comparison. The [50-minute comparison](results/cv/benchmark/comparison.md) records model selection. Further RF-DETR Small refinement reached mAP **0.5105** and handwritten line coverage **89.7%**; the [refinement report](results/cv/rf-small-refined.md) includes metrics and failure examples.

```sh
uv run python -u scripts/benchmark_detectors.py --minutes 50 --resume-existing
uv run python scripts/benchmark_detectors.py --report
```

CV notebooks cover [model selection](notebooks/04_detector_comparison.ipynb), [data checks](notebooks/05_cv_data.ipynb), [refinement](notebooks/06_rf_small_refine.ipynb) and [evaluation](notebooks/07_cv_eval.ipynb).

The benchmark budget is cumulative per candidate. `--resume-existing` resumes saved progress, `--models rf-small` selects one candidate, and `--evaluate-only` evaluates saved checkpoints. Run settings and scores are recorded in `results/cv/benchmark/runs.json`.

[08_htr_detected_eval](notebooks/08_htr_detected_eval.ipynb) runs the same TrOCR checkpoint on the refined RF-DETR crops (`data/detector_crops/rf-small-refined/`) and compares it with annotated crops on the same matched lines. On the 1,862 matched validation lines CER is 0.1119 versus 0.1102 for annotated crops (WER 0.3056 versus 0.3071); 214 of 2,076 eligible lines are missed. Results are in `results/htr/detected_*`, with page-bootstrap 95% intervals per source.

[09_htr_page_pipeline](notebooks/09_htr_page_pipeline.ipynb) runs the full chain on the 133 validation pages (RF-DETR Small at score 0.55, TrOCR on handwritten boxes, lines joined top to bottom; helpers in `src/pipeline.py`). Page CER, where missed lines are deletions and extra predictions are insertions, is 0.1849 (95% interval 0.1515–0.2207) against 0.1119 on matched lines. It is not comparable to matched-line CER or to Kaggle scores. Results are in `results/htr/page_*`.

**Silver fine-tuning** (`scripts/silver_htr.py`, run end to end by `scripts/run_silver_htr.sh`, about 5 hours) continues A for 2,800 steps with a fresh optimizer on the reviewed 1,050-page silver pool (`data/silver_pool/`). B uses the supplied transcripts. C uses A's own transcripts filtered by confidence. H is a step-matched human-only control and M mixes silver with human replay. On the 2,076 human validation lines (beam 4), line CER is A 0.1150, B 0.1217, C 0.1241, H 0.1131 and M **0.1046**. M's gain over A is −0.0104 (paired page-bootstrap 95% interval −0.0145 to −0.0065). B gains on archive, school and university but loses on dictation, which silver lacks. C is worse than A everywhere. With the midterm detector, validation page CER is A 0.1849, B 0.1807, C 0.1910 and M **0.1763**. `scripts/silver_report.py` writes the full tables to `results/htr/silver/README.md`. The [silver results](results/htr/silver-finetune.md) give findings and caveats, including that M was designed after B and C were scored.

Weights, data and predicted crops stay outside Git.

## Contribution statement

- **Oleh Hutsuliak** CV: image audit, detection/types, crops, augmentation, and CV metrics.
- **Oleksandr Stadnik** NLP/HTR: text audit, recognition/decoding, pseudo-label filtering, and CER/WER.
- **Shared equally throughout:** split design, integration, experiments, analysis, documentation, and presentations.
- **Agreement:** both agreed.

## AI usage

Codex assisted with writing and the EDA script. We guided and reviewed the earlier work, and ran the audit on the local snapshot.

Claude assisted with the recognition helpers and notebooks. We guided and reviewed the work and checked its results against the data.

Codex assisted with fine-tuning setup and timed detector benchmarks. We checked box coordinates, validation predictions, training/resume behavior.
