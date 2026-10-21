# Splits

`train_val.csv` assigns every official `train` page to `train` or `val`. It has one row per page: `file_name`, `source`, `source_detail`, `annotation_source`, `split`.

- About 10% of pages per source go to validation (133 of 1,330), sampled with seed 42.
- All regions of a page stay in that page's split. The metadata has no writer or document IDs, and train has no exact duplicate images, so no further grouping is possible.
- Train dictation pages all copy the same 2024 dictation text, so validation dictation is partly memorisable. Report results per source.
- The official `test` split is not part of this file and is not used for model or threshold selection.

The file is written by `notebooks/01_htr_pilot.ipynb`. It is provisional until both modules agree on it. If it changes, rerun both HTR notebooks.
