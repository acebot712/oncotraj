"""Locked baselines per PAPER_OUTLINE.md §6.1.

- LogisticRegression: static features only.
- RandomForest: static + last-observed serial (snapshot).
- XGBoost: static + last-observed serial.
- LSTM: full serial trajectory.
- SmallTransformer: full serial trajectory with learned time embeddings (~10M params).

All baselines are post-hoc calibrated with isotonic regression on validation
before test-set evaluation.
"""
