"""Hand-crafted feature engineering for the v1 baselines.

Per the user spec for paper #6 (classical ML baselines), features are:
- `latest_vaf`: max VAF across this patient's most recent ctDNA sample.
- `vaf_slope`: per-day change in max EGFR-VAF from earliest to latest sample.
  A simple two-point slope; full linear regression is overkill for the
  patient counts we have today.
- `variant_burden`: total count of variant rows for the patient.
- `time_on_therapy_days`: (treatment.end_date or last_followup_date) -
  treatment.start_date. Negative or missing values clamp to 0.
- One-hot indicators for `source_dataset` and `egfr_variant_class`.

The output is a DataFrame indexed by `patient_id`. Target builders
(`build_target`) produce a `pd.Series` aligned to this index.

Limitations the user already knows about:
- VAF features are weak for synthetic and FLAURA rows (synthetic GENIE/CHORD
  doesn't emit serial draws; FLAURA emits at most baseline + discontinuation).
  When real ctDNA dynamics land via TRACERx or registries, these features
  become useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BuiltTables:
    """The four-table view a feature builder operates on."""

    patients: pd.DataFrame
    variants: pd.DataFrame
    treatments: pd.DataFrame
    outcomes: pd.DataFrame


def _latest_vaf_per_patient(variants: pd.DataFrame) -> pd.Series:
    """Max VAF (excluding sentinels) at each patient's most-recent sample."""
    if variants.empty:
        return pd.Series(dtype=float, name="latest_vaf")
    df = variants.loc[variants["vaf"] >= 0.0, ["patient_id", "sample_date", "vaf"]].copy()
    if df.empty:
        return pd.Series(dtype=float, name="latest_vaf")
    df["sample_date"] = pd.to_datetime(df["sample_date"])
    df = df.sort_values(["patient_id", "sample_date"])
    last = df.groupby("patient_id").tail(1).groupby("patient_id")["vaf"].max()
    last.name = "latest_vaf"
    return last


def _vaf_slope_per_patient(variants: pd.DataFrame) -> pd.Series:
    """(latest_vaf - baseline_vaf) / days_between, in 1/day units.

    Computed only on EGFR sensitizing-allele VAFs (gene == EGFR, alteration SNV).
    Zero for patients with only one sample.
    """
    if variants.empty:
        return pd.Series(dtype=float, name="vaf_slope")
    egfr = variants.loc[
        (variants["gene_symbol"] == "EGFR")
        & (variants["alteration_type"] == "SNV")
        & (variants["vaf"] >= 0.0),
        ["patient_id", "sample_date", "vaf"],
    ].copy()
    if egfr.empty:
        return pd.Series(dtype=float, name="vaf_slope")
    egfr["sample_date"] = pd.to_datetime(egfr["sample_date"])
    egfr = egfr.sort_values(["patient_id", "sample_date"])
    grouped = egfr.groupby("patient_id")
    first = grouped.head(1).set_index("patient_id")
    last = grouped.tail(1).set_index("patient_id")
    common = first.index.intersection(last.index)
    if common.empty:
        return pd.Series(dtype=float, name="vaf_slope")
    days = (last.loc[common, "sample_date"] - first.loc[common, "sample_date"]).dt.days
    delta = last.loc[common, "vaf"] - first.loc[common, "vaf"]
    slope = (delta / days.replace(0, np.nan)).fillna(0.0)
    slope.name = "vaf_slope"
    return slope


def _variant_burden(variants: pd.DataFrame) -> pd.Series:
    if variants.empty:
        return pd.Series(dtype=int, name="variant_burden")
    s = variants.groupby("patient_id").size().astype(int)
    s.name = "variant_burden"
    return s


def _time_on_therapy(treatments: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    """Days of osimertinib exposure: end_date - start_date, or last_followup - start."""
    if treatments.empty or patients.empty:
        return pd.Series(dtype=float, name="time_on_therapy_days")
    osi = treatments.loc[
        treatments["is_osimertinib"], ["patient_id", "start_date", "end_date"]
    ].copy()
    if osi.empty:
        return pd.Series(dtype=float, name="time_on_therapy_days")
    osi["start_date"] = pd.to_datetime(osi["start_date"])
    osi["end_date"] = pd.to_datetime(osi["end_date"])
    lfu = patients.set_index("patient_id")["last_followup_date"].pipe(pd.to_datetime)
    osi = osi.set_index("patient_id")
    effective_end = osi["end_date"].fillna(lfu.reindex(osi.index))
    days = (effective_end - osi["start_date"]).dt.days.clip(lower=0).fillna(0.0)
    days.name = "time_on_therapy_days"
    return days.groupby(level=0).max()


def build_features(tables: BuiltTables) -> pd.DataFrame:
    """Assemble the v1 baseline feature matrix indexed by patient_id."""
    patients = tables.patients.set_index("patient_id")
    parts = [
        _latest_vaf_per_patient(tables.variants),
        _vaf_slope_per_patient(tables.variants),
        _variant_burden(tables.variants),
        _time_on_therapy(tables.treatments, tables.patients),
    ]
    feats = pd.concat(parts, axis=1)
    feats = feats.reindex(patients.index)
    # Categorical one-hots.
    one_hots = pd.get_dummies(
        patients[["source_dataset", "egfr_variant_class"]],
        prefix=["src", "egfr"],
        dtype=float,
    )
    feats = pd.concat([feats, one_hots], axis=1)
    # Fill NaNs with sentinel zeros for numeric features; one-hots are already 0/1.
    feats = feats.fillna(0.0)
    feats.index.name = "patient_id"
    return feats


# ---------------------------------------------------------------------------
# Target builders
# ---------------------------------------------------------------------------


def build_target_a(tables: BuiltTables) -> pd.Series:
    """Task A v1 simplification: 1 iff the patient has any progression_recist
    outcome. The locked Task A is `progression within 90 days of prediction
    time t`, but we lack absolute dates for FLAURA; this binary stand-in
    keeps the baseline-training pipeline runnable today.
    """
    prog = tables.outcomes.loc[
        tables.outcomes["event_type"] == "progression_recist", "patient_id"
    ].unique()
    target = pd.Series(0, index=tables.patients["patient_id"].unique(), dtype=int, name="task_a")
    target.loc[target.index.isin(prog)] = 1
    return target


def build_target_b(tables: BuiltTables) -> pd.Series:
    """Task B: days from osimertinib start to first progression. NaN if no PD."""
    treatments = tables.treatments.loc[tables.treatments["is_osimertinib"]].copy()
    treatments["start_date"] = pd.to_datetime(treatments["start_date"])
    start = treatments.groupby("patient_id")["start_date"].min()

    prog = tables.outcomes.loc[
        tables.outcomes["event_type"] == "progression_recist", ["patient_id", "event_date"]
    ].copy()
    prog["event_date"] = pd.to_datetime(prog["event_date"])
    first_prog = prog.groupby("patient_id")["event_date"].min()

    common = start.index.intersection(first_prog.index)
    days = (first_prog.loc[common] - start.loc[common]).dt.days.astype(float)
    days.name = "task_b"
    return days


def build_target_c(tables: BuiltTables) -> pd.Series:
    """Task C: dominant mechanism class at first progression."""
    prog = tables.outcomes.loc[
        tables.outcomes["event_type"] == "progression_recist",
        ["patient_id", "event_date", "resistance_mechanism_class"],
    ].copy()
    prog["event_date"] = pd.to_datetime(prog["event_date"])
    prog = prog.sort_values(["patient_id", "event_date"])
    first = prog.groupby("patient_id").head(1).set_index("patient_id")["resistance_mechanism_class"]
    first = first.dropna()
    first.name = "task_c"
    return first


TASK_BUILDERS = {
    "A": ("classification", build_target_a),
    "B": ("regression", build_target_b),
    "C": ("classification", build_target_c),
}
