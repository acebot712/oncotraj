"""Per-task evaluation metrics.

All functions accept plain numpy/pandas inputs (no torch dependency) so the
evaluation harness is decoupled from any specific model framework. Optional
arguments are skipped cleanly if missing — e.g. AUC is reported only when
probabilities are present; Harrell's C-index only when censoring is provided.

Locked metrics per PAPER_OUTLINE.md §8:
- Task A: ROC-AUC, Brier score, Expected Calibration Error, reliability bins.
- Task B: MAE on uncensored events, Harrell's C-index on the full cohort.
- Task C: macro and weighted F1, per-class F1, OvR ROC-AUC, confusion matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Equal-width reliability binning.

    Returns a dict with keys `bin_centers`, `frequencies`, `mean_pred`, `counts`.
    Bins with zero patients are dropped so downstream plotting/JSON output is
    compact.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)

    bin_centers: list[float] = []
    frequencies: list[float] = []
    mean_pred: list[float] = []
    counts: list[int] = []
    for b in range(n_bins):
        mask = indices == b
        n = int(mask.sum())
        if n == 0:
            continue
        bin_centers.append(float(0.5 * (edges[b] + edges[b + 1])))
        frequencies.append(float(y_true[mask].mean()))
        mean_pred.append(float(y_prob[mask].mean()))
        counts.append(n)
    return {
        "bin_centers": bin_centers,
        "frequencies": frequencies,
        "mean_pred": mean_pred,
        "counts": counts,
        "n_bins": n_bins,
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width ECE.

    ECE = sum_b (n_b / N) * |freq_b - mean_pred_b|.
    """
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins)
    total = sum(bins["counts"]) or 1
    return float(
        sum(
            (n / total) * abs(freq - pred)
            for n, freq, pred in zip(
                bins["counts"], bins["frequencies"], bins["mean_pred"], strict=False
            )
        )
    )


# ---------------------------------------------------------------------------
# Task A
# ---------------------------------------------------------------------------


def task_a_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
    n_bins: int = 10,
) -> dict:
    """Binary task: 90-day resistance event (v1 simplification: ever-progresses).

    AUC, Brier, and calibration are reported only when probabilities are
    provided. Accuracy and F1 are always reported.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metrics: dict = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None  # single-class slice
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
        metrics["ece"] = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        metrics["reliability"] = reliability_bins(y_true, y_prob, n_bins=n_bins)
    return metrics


# ---------------------------------------------------------------------------
# Task B
# ---------------------------------------------------------------------------


def harrell_c_index(times: np.ndarray, preds: np.ndarray, events: np.ndarray) -> float:
    """Harrell's C-index. Higher is better.

    Pairs are concordant when, between two patients with observed event
    times, the one with the shorter time has a higher predicted hazard
    (i.e. shorter `pred`). Pairs where neither event is observed, or the
    earlier of the two is censored, are uncomparable and dropped.
    """
    times = np.asarray(times, dtype=float)
    preds = np.asarray(preds, dtype=float)
    events = np.asarray(events).astype(int)
    n = len(times)
    if n < 2:
        return float("nan")
    concordant = 0
    tied = 0
    comparable = 0
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j or times[j] < times[i]:
                continue
            # `i` had event at t_i; `j` is either at risk later (event later
            # or censored later). j's time must be > t_i for the pair to be
            # comparable.
            if times[j] <= times[i] and events[j] == 0:
                continue
            comparable += 1
            if preds[i] < preds[j]:
                concordant += 1
            elif preds[i] == preds[j]:
                tied += 1
    if comparable == 0:
        return float("nan")
    return (concordant + 0.5 * tied) / comparable


def task_b_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    event: Sequence[int] | None = None,
) -> dict:
    """Time-to-resistance regression.

    - MAE on the subset with event observed.
    - RMSE on the same subset.
    - Harrell's C-index on the full cohort (requires event indicator).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics: dict = {"n": len(y_true)}
    ev = np.ones_like(y_true, dtype=int) if event is None else np.asarray(event).astype(int)
    obs_mask = ev == 1
    if obs_mask.any():
        metrics["mae"] = float(mean_absolute_error(y_true[obs_mask], y_pred[obs_mask]))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true[obs_mask], y_pred[obs_mask])))
        metrics["n_observed"] = int(obs_mask.sum())
    metrics["c_index"] = harrell_c_index(y_true, y_pred, ev)
    metrics["censoring_rate"] = float((ev == 0).mean())
    return metrics


# ---------------------------------------------------------------------------
# Task C
# ---------------------------------------------------------------------------


def task_c_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_prob: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Multi-class mechanism classification.

    `y_prob` should be a 2D array of shape (n, K) aligned to `class_names`
    when present; OvR ROC-AUC is reported when probs are given.
    """
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    classes = list(class_names) if class_names else sorted(set(y_true_arr) | set(y_pred_arr))

    metrics: dict = {
        "n": len(y_true_arr),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", labels=classes, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", labels=classes, zero_division=0)
        ),
        "per_class_f1": {
            cls: float(
                f1_score(y_true_arr, y_pred_arr, labels=[cls], average="macro", zero_division=0)
            )
            for cls in classes
        },
        "confusion_matrix": {
            "labels": classes,
            "matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=classes).tolist(),
        },
        "class_distribution": {cls: int((y_true_arr == cls).sum()) for cls in classes},
    }

    if y_prob is not None and len(classes) >= 2:
        try:
            arr = np.asarray(y_prob, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == len(classes):
                # OvR AUC averaged across classes.
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(
                        pd.get_dummies(y_true_arr)
                        .reindex(columns=classes, fill_value=0)
                        .to_numpy(),
                        arr,
                        average="macro",
                        multi_class="ovr",
                    )
                )
        except ValueError:
            metrics["roc_auc_ovr"] = None
    return metrics
