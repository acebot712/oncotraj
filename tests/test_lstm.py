"""LSTM baseline tests.

Most tests run on CPU with tiny epochs/batches. The grid-search integration
test runs a 2-config sweep against the synthetic parquet build to keep CI
under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from oncotraj.models.lstm import (
    LSTMModel,
    PerSampleFeatureDim,
    build_sequences,
    pad_collate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_dataset  # noqa: E402
import train_lstm  # noqa: E402


def _toy_variants() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "gene_symbol": "EGFR",
                "alteration_type": "SNV",
                "vaf": 0.10,
                "sample_date": "2022-01-01",
                "protein_change_hgvs": "p.Leu858Arg",
            },
            {
                "patient_id": "p1",
                "gene_symbol": "EGFR",
                "alteration_type": "SNV",
                "vaf": 0.40,
                "sample_date": "2023-01-01",
                "protein_change_hgvs": "p.Cys797Ser",
            },
            {
                "patient_id": "p2",
                "gene_symbol": "EGFR",
                "alteration_type": "SNV",
                "vaf": 0.20,
                "sample_date": "2022-06-01",
                "protein_change_hgvs": "p.Leu858Arg",
            },
            {
                "patient_id": "p3",
                "gene_symbol": "MET",
                "alteration_type": "CNV_amplification",
                "vaf": -1.0,
                "sample_date": "2023-06-01",
                "protein_change_hgvs": None,
            },
        ]
    )


def test_build_sequences_shape():
    seqs = build_sequences(_toy_variants(), ["p1", "p2", "p3", "p_missing"])
    # p1 has 2 timepoints, p2/p3 have 1, p_missing has 0 -> 1-length zero seq.
    assert seqs["p1"].shape == (2, PerSampleFeatureDim)
    assert seqs["p2"].shape == (1, PerSampleFeatureDim)
    assert seqs["p3"].shape == (1, PerSampleFeatureDim)
    assert seqs["p_missing"].shape == (1, PerSampleFeatureDim)
    # baseline_flag is 1.0 at first timestep, 0 thereafter.
    assert seqs["p1"][0, -1] == 1.0
    assert seqs["p1"][1, -1] == 0.0


def test_build_sequences_c797s_flag_emergent():
    """Per-sample c797s_present should be 0 at baseline and 1 at PD."""
    seqs = build_sequences(_toy_variants(), ["p1"])
    assert seqs["p1"][0, 7] == 0.0  # baseline: only L858R
    assert seqs["p1"][1, 7] == 1.0  # PD: C797S


def test_pad_collate_sorts_by_length():
    batch = [
        {
            "pid": "a",
            "seq": torch.zeros(1, PerSampleFeatureDim),
            "length": 1,
            "reg": torch.tensor(1.0),
            "reg_mask": torch.tensor(1.0),
            "cls": torch.tensor(0),
            "cls_mask": torch.tensor(1.0),
        },
        {
            "pid": "b",
            "seq": torch.zeros(3, PerSampleFeatureDim),
            "length": 3,
            "reg": torch.tensor(2.0),
            "reg_mask": torch.tensor(1.0),
            "cls": torch.tensor(1),
            "cls_mask": torch.tensor(1.0),
        },
        {
            "pid": "c",
            "seq": torch.zeros(2, PerSampleFeatureDim),
            "length": 2,
            "reg": torch.tensor(3.0),
            "reg_mask": torch.tensor(1.0),
            "cls": torch.tensor(0),
            "cls_mask": torch.tensor(1.0),
        },
    ]
    out = pad_collate(batch)
    # Sorted descending by length: b, c, a.
    assert out["pids"] == ["b", "c", "a"]
    assert out["lengths"].tolist() == [3, 2, 1]
    assert out["seq"].shape == (3, 3, PerSampleFeatureDim)


def test_lstm_fit_predict_classifier_tiny():
    """Fit on a 6-patient toy cohort, verify shapes; not testing accuracy."""
    pids = [f"p{i}" for i in range(6)]
    variants = pd.concat(
        [
            _toy_variants().assign(patient_id=lambda d: pid)  # noqa: B023
            for pid in pids
        ],
        ignore_index=True,
    )
    X = pd.DataFrame(index=pids)
    y_cls = pd.Series(["EGFR_C797S", "other_or_unknown"] * 3, index=pids, name="task_c")

    model = LSTMModel(
        is_classifier=True,
        hidden_size=8,
        num_layers=1,
        epochs=3,
        patience=10,
        device=torch.device("cpu"),
        variants_df=variants,
    )
    model.fit(X, y_cls)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert preds.shape == (6,)
    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lstm_fit_predict_regressor_tiny():
    pids = [f"p{i}" for i in range(6)]
    variants = pd.concat(
        [
            _toy_variants().assign(patient_id=lambda d: pid)  # noqa: B023
            for pid in pids
        ],
        ignore_index=True,
    )
    X = pd.DataFrame(index=pids)
    y_reg = pd.Series(np.arange(6, dtype=float) * 30, index=pids, name="task_b")

    model = LSTMModel(
        is_classifier=False,
        hidden_size=8,
        num_layers=1,
        epochs=3,
        patience=10,
        device=torch.device("cpu"),
        variants_df=variants,
    )
    model.fit(X, y_reg)
    preds = model.predict(X)
    assert preds.shape == (6,)
    with pytest.raises(NotImplementedError):
        model.predict_proba(X)


def test_lstm_joint_multitask_uses_both_heads():
    pids = [f"p{i}" for i in range(8)]
    variants = pd.concat(
        [
            _toy_variants().assign(patient_id=lambda d: pid)  # noqa: B023
            for pid in pids
        ],
        ignore_index=True,
    )
    y_reg = pd.Series([100.0, 200.0, 150.0, 80.0], index=pids[:4], name="task_b")
    y_cls = pd.Series(
        ["EGFR_C797S", "other_or_unknown", "EGFR_C797S", "other_or_unknown"],
        index=pids[4:],
        name="task_c",
    )

    model = LSTMModel(
        hidden_size=8,
        num_layers=1,
        epochs=5,
        patience=20,
        device=torch.device("cpu"),
    )
    model.fit_multitask(
        patient_ids=pids,
        variants=variants,
        y_reg=y_reg,
        y_cls=y_cls,
    )
    X = pd.DataFrame(index=pids)
    model.is_classifier = False
    reg_preds = model.predict(X)
    model.is_classifier = True
    cls_preds = model.predict(X)
    assert reg_preds.shape == (8,)
    assert cls_preds.shape == (8,)


def test_lstm_grid_search_end_to_end(tmp_path, monkeypatch):
    """Mini 2-config grid against a synthetic-only parquet."""
    from oncotraj.data.splits import make_splits_from_parquet

    output = tmp_path / "v0"
    build_dataset.build_dataset(
        output_dir=output,
        cohort="egfr_nsclc",
        use_synthetic=True,
        raw_root=tmp_path / "raw",
        papers_root=None,
    )
    splits_path = tmp_path / "splits.json"
    make_splits_from_parquet(parquet_dir=output, output_path=splits_path, seed=42)

    monkeypatch.setattr(train_lstm, "REPO_ROOT", tmp_path)

    result = train_lstm.grid_search(
        parquet_dir=output,
        splits_path=splits_path,
        output_dir=tmp_path / "models",
        hidden_grid=[8, 16],
        lr_grid=[1e-3],
        epochs=3,
    )
    assert result["n_grid"] == 2
    assert result["best"]["config"] is not None
    assert (tmp_path / "models" / "taskB_lstm.joblib").exists()
    assert (tmp_path / "models" / "taskC_lstm.joblib").exists()
