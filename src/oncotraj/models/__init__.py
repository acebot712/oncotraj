"""Classical-ML baselines for the v1 OncoTraj tasks (PAPER_OUTLINE.md §6.1).

Five locked baselines per the outline; this module implements the four
non-sequence ones — LSTM and Transformer are out of scope here.

Unified interface (see `base.py`):
- `model.fit(X, y)`
- `model.predict(X)`
- `model.predict_proba(X)` (classifiers only; regressors raise NotImplementedError)
- `model.save(path)` / `Model.load(path)` via joblib

Available models:
- `MajorityClassModel` — predicts the modal class (or median) seen in train.
- `LogisticRegressionModel` — sklearn LogisticRegression + StandardScaler.
- `RandomForestModel` — sklearn RandomForest{Classifier,Regressor}.
- `XGBoostModel` — xgboost.XGB{Classifier,Regressor}.
"""

from .base import MODEL_REGISTRY, OncoTrajModel, get_model
from .logistic import LogisticRegressionModel
from .majority import MajorityClassModel
from .random_forest import RandomForestModel
from .xgb import XGBoostModel

__all__ = [
    "MODEL_REGISTRY",
    "LogisticRegressionModel",
    "MajorityClassModel",
    "OncoTrajModel",
    "RandomForestModel",
    "XGBoostModel",
    "get_model",
]
