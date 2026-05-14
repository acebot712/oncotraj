# OncoTraj

A benchmark for longitudinal resistance prediction in EGFR+ NSCLC on first-line osimertinib.

**Status:** v0.1.0 scaffold — schema and code surface only. No data, no trained models yet.

## Scope (v1, locked)

- **Cohort:** EGFR-mutant advanced NSCLC, first-line osimertinib.
- **Tasks:** (A) 90-day binary resistance, (B) time-to-resistance regression, (C) per-mechanism classification (6-class).
- **Baselines:** LogReg, RandomForest, XGBoost, LSTM, small Transformer (~10M).
- **Compute target:** single A100, ≤24h per model.
- **Splits:** patient-level 70/15/15 + site-out (TRACERx) held-out.

## Quickstart

```bash
uv sync
uv run pytest
uv run ruff check
```

## Layout

```
src/oncotraj/      # library code (schemas, parsers, splits, metrics, baselines)
tests/             # pytest suite
notebooks/         # exploratory analyses
scripts/           # one-off CLI utilities
data/raw/          # gitignored — source datasets land here
data/processed/    # harmonized Parquet outputs per DATASET_SPEC.md
```

## License

Code: MIT (see [`LICENSE`](LICENSE)). Data redistribution follows upstream terms (AACR GENIE, TRACERx, MSK-CHORD, etc.).

## Citation

Citation stub — to be filled at preprint.
