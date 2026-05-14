# OncoTraj v1 Leaderboard

Auto-generated from `src/oncotraj/eval`. Submissions are evaluated against
the locked test split via `oncotraj-eval --predictions <csv> --split test`
and dropped as JSON into `eval_reports/`. Run `oncotraj-eval --refresh`
(or call `oncotraj.eval.update_leaderboard(...)` programmatically) to
regenerate the table below.

Default sort: **Task A ROC-AUC on test**, descending. Secondary sort:
**Task C macro F1**.

<!-- LEADERBOARD START -->
<!-- Auto-generated; do not edit between markers. -->

### Split: `test`

| Submission | n | Task A AUC | Task A Brier | Task A ECE | Task B MAE (d) | Task B C-index | Task C macro F1 | Task C accuracy | Generated |
|---|---|---|---|---|---|---|---|---|---|
| `random_forest_taskA` | 24 | 0.941 | 0.084 | 0.099 | — | — | — | — | 2026-05-14T11:17:38 |
| `xgboost_taskA` | 24 | 0.933 | 0.102 | 0.130 | — | — | — | — | 2026-05-14T11:17:40 |
| `logistic_taskA` | 24 | 0.924 | 0.086 | 0.185 | — | — | — | — | 2026-05-14T11:17:38 |
| `majority_taskA` | 24 | 0.500 | 0.207 | 0.017 | — | — | — | — | 2026-05-14T11:17:38 |

_No submissions on the `val` split yet._

<!-- LEADERBOARD END -->

## How to submit

1. Train a model on the OncoTraj v1 cohort (see `data/processed/oncotraj_v0/`).
2. Write predictions to a CSV conforming to the schema in
   `src/oncotraj/eval/report.py` (`SCHEMA_DOC`).
3. Run `oncotraj-eval --predictions yours.csv --split test \
       --submission-id <your-name> --output eval_reports/<your-name>.json`.
4. Rerun with `--leaderboard leaderboard.md` to update this file.
