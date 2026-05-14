"""Per-source parsers. One module per `SourceDataset` enum value.

Each parser exposes `parse(input_dir, output_dir) -> None` and emits the four
Parquet files plus `_meta.json` per DATASET_SPEC.md.
"""
