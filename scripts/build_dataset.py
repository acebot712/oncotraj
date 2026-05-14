"""Build the harmonised OncoTraj dataset from one or more source caches.

Usage:
    python scripts/build_dataset.py --cohort egfr_nsclc \
        --output data/processed/oncotraj_v0.parquet \
        --use-synthetic

The --output argument is treated as a directory; the four canonical tables
(patients/variants/treatments/outcomes.parquet) are written inside it,
along with `_meta.json` recording row counts and the schema version.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from oncotraj import SCHEMA_VERSION
from oncotraj.parsers import cbioportal, genie, synthetic
from oncotraj.parsers import papers as papers_pkg
from oncotraj.parsers.common import ParsedTables, validate_tables


def _build_synthetic(raw_root: Path) -> ParsedTables:
    """Materialise GENIE + cBioPortal-shaped synthetic fixtures and parse them."""
    genie_dir = raw_root / "genie"
    cbio_dir = raw_root / "cbioportal"
    synthetic.write_synthetic_genie(genie_dir)
    synthetic.write_synthetic_cbioportal(cbio_dir)
    return genie.parse(genie_dir).concat(cbioportal.parse(cbio_dir))


def _build_real(raw_root: Path, fetch: bool) -> ParsedTables:  # pragma: no cover - network
    genie_dir = raw_root / "genie"
    cbio_dir = raw_root / "cbioportal"
    if fetch:
        genie.fetch_genie(genie_dir)
        cbioportal.fetch_cbioportal(cbio_dir)
    return genie.parse(genie_dir).concat(cbioportal.parse(cbio_dir))


def _apply_cohort(tables: ParsedTables, cohort: str) -> ParsedTables:
    if cohort != "egfr_nsclc":
        raise ValueError(f"Unsupported cohort: {cohort!r}. v0 supports only 'egfr_nsclc'.")
    included = tables.patients.loc[tables.patients["included_in_v1_cohort"], "patient_id"].tolist()
    in_cohort = set(included)
    return ParsedTables(
        patients=tables.patients.loc[tables.patients["patient_id"].isin(in_cohort)].reset_index(
            drop=True
        ),
        variants=tables.variants.loc[tables.variants["patient_id"].isin(in_cohort)].reset_index(
            drop=True
        ),
        treatments=tables.treatments.loc[
            tables.treatments["patient_id"].isin(in_cohort)
        ].reset_index(drop=True),
        outcomes=tables.outcomes.loc[tables.outcomes["patient_id"].isin(in_cohort)].reset_index(
            drop=True
        ),
    )


def _write_outputs(tables: ParsedTables, output_dir: Path, sources: list[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables.patients.to_parquet(output_dir / "patients.parquet", index=False)
    tables.variants.to_parquet(output_dir / "variants.parquet", index=False)
    tables.treatments.to_parquet(output_dir / "treatments.parquet", index=False)
    tables.outcomes.to_parquet(output_dir / "outcomes.parquet", index=False)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "sources": sources,
        "row_counts": {
            "patients": len(tables.patients),
            "variants": len(tables.variants),
            "treatments": len(tables.treatments),
            "outcomes": len(tables.outcomes),
        },
    }
    (output_dir / "_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def build_dataset(
    output_dir: Path,
    cohort: str = "egfr_nsclc",
    use_synthetic: bool = False,
    fetch: bool = False,
    raw_root: Path | None = None,
    papers_root: Path | None = None,
) -> dict:
    raw_root = raw_root or (Path("data") / "raw" / ("synthetic" if use_synthetic else "real"))

    tables = _build_synthetic(raw_root) if use_synthetic else _build_real(raw_root, fetch)

    paper_study_ids: list[str] = []
    if papers_root is not None and Path(papers_root).exists():
        paper_tables = papers_pkg.load_all(Path(papers_root))
        if len(paper_tables):
            tables = tables.concat(paper_tables)
            paper_study_ids = sorted(
                {
                    pid.split(":", 1)[1].split(".", 1)[0]
                    for pid in paper_tables.patients["patient_id"]
                    if ":" in pid and "." in pid.split(":", 1)[1]
                }
            )

    cohort_tables = _apply_cohort(tables, cohort)

    # Cast date columns so pyarrow writes them as DATE32, not object.
    for df in (
        cohort_tables.patients,
        cohort_tables.variants,
        cohort_tables.treatments,
        cohort_tables.outcomes,
    ):
        for col in df.columns:
            if (
                df[col].dtype == object
                and not df[col].empty
                and isinstance(
                    df[col].dropna().iloc[0] if not df[col].dropna().empty else None, date
                )
            ):
                df[col] = pd.to_datetime(df[col]).dt.date

    validate_tables(cohort_tables)
    sources = ["GENIE_BPC", "MSK_CHORD"]
    if use_synthetic:
        sources.append("__synthetic__")
    sources.extend(f"paper:{sid}" for sid in paper_study_ids)
    return _write_outputs(cohort_tables, output_dir, sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="egfr_nsclc")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Bypass real-data fetch; materialise synthetic fixtures and parse them.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Force network fetch of GENIE + cBioPortal sources before parsing.",
    )
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=None,
        help="Root dir containing per-study supplement folders. Each subdir name "
        "must match a registered adapter's study_id (see parsers/papers/README.md).",
    )
    args = parser.parse_args()

    meta = build_dataset(
        output_dir=args.output,
        cohort=args.cohort,
        use_synthetic=args.use_synthetic,
        fetch=args.fetch,
        raw_root=args.raw_root,
        papers_root=args.papers_dir,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
