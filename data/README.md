# RUKOPYS

Ukrainian document-recognition data from [UCU's RUKOPYS dataset](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys).

| Use | Pinned revision |
| --- | --- |
| Human training and the original 133-page validation set | v2.2 · `78697007cfa5f77aae5d3b9fcedfac3169f84fb6` |
| Silver fine-tuning and final 382-page test | v3.0 · `50b4cdc0e861ae2adf481def238068563f405ff8` |

Keep both revisions. The human split and its labels define the baseline comparison. Score the saved baseline and the frozen final model on the same v3.0 test set. [Release notes](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys#changelog)

Each split contains `images/` and `metadata.jsonl`, one record per page. Regions have pixel boxes `[x1, y1, x2, y2]`, type, transcript, language, and legibility. Types: handwritten, printed, formula (LaTeX), table (pipe-separated values), annotation, image, and graph (last two expect empty text). See [EDA](eda.md) for counts, examples, and issues.

## Setup, download, and run

Requires [uv](https://docs.astral.sh/uv/). From the repository root, install the locked project environment:

```sh
uv sync
```

Sync the pinned splits from the repository root. The script reuses the existing `data/rukopys` cache when present and otherwise uses the standard Hugging Face cache. It reads an optional `HF_TOKEN` from the environment or your local `.env`.

```sh
uv run python scripts/sync_rukopys.py
```

The existing [EDA](eda.md) describes the original v2.2 snapshot. To rerun a dataset-wide audit, supply a full snapshot with all three splits. The sync command above fetches only the splits needed by the project at each revision.

```sh
uv run python scripts/eda.py PATH_TO_SNAPSHOT
```

The script writes `data/eda.md` and three figures in `data/images/`. It checks metadata, image headers, and exact duplicates; it does not fully decode every image. Raw data stay outside Git.

See the [silver pool record](silver.md) for selection, review status and the public page IDs.

## Use and handling

- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), including commercial reuse. Credit the authors/UCU, link the dataset/license, and identify changes.
- **Privacy:** the provider reports removing personal information.
- **Collection/secrets:** no scraping; use the provided download.
- **Risk:** transcription can omit or invent content.
