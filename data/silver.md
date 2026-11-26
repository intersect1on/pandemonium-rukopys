# Silver fine-tuning pool

The shared pool contains 1,050 pages from RUKOPYS v3.0, pinned to `50b4cdc0e861ae2adf481def238068563f405ff8`. The [page manifest](splits/silver_pool.csv) lists 350 archive, 350 school and 350 university IDs. The reviewed export contains 14,918 eligible Ukrainian handwriting crops for TrOCR. RF-DETR uses the full-page layout from the same pages.

Seed 42 drives candidate selection. The builder checks image geometry and crop eligibility. It screens for near-duplicate images against human train/validation, v3.0 test and selected silver pages, and for repeated text against human train/validation and selected silver pages. [RUKOPYS release notes](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys#changelog)

Three Luna reviewers inspected all 1,200 original page layouts and 338 replacement candidates. The review included 4,143 transcript region spot checks. The university shortlist had 354 approved pages, so the final selection uses 350 pages from each source. It keeps 930 originals and uses 120 replacements. All selected pages retain their supplied v3.0 annotations.

After [syncing the pinned splits](README.md), run `uv run python scripts/prepare_silver_pool.py --manifest data/splits/silver_pool.csv`. This writes the boxes, transcripts and crop images under the ignored `data/silver_pool/` folder. The shared CSV contains IDs and source names only.
