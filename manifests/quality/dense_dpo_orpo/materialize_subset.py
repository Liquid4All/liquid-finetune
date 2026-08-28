#!/usr/bin/env python3
"""Materialize immutable 10k/1k ORPO-DPO preference subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset, load_dataset

from leap_finetune.data_loading.validate_dataset_format import get_row_filter

DATASET = "mlabonne/orpo-dpo-mix-40k"
SUBSET = "default"
REVISION = "0f72511202b8f093e9be60e1683d84b046062e36"
TRAIN_ROWS = 10_000
EVAL_ROWS = 1_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = load_dataset(
        DATASET,
        SUBSET,
        revision=REVISION,
        split="train",
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    is_valid = get_row_filter("dpo", model_family="lfm25")
    selected = []
    for source_row_id, row in enumerate(source):
        if not is_valid(row):
            continue
        selected.append({**row, "source_row_id": source_row_id})
        if len(selected) == TRAIN_ROWS + EVAL_ROWS:
            break
    if len(selected) != TRAIN_ROWS + EVAL_ROWS:
        raise RuntimeError(
            f"Expected {TRAIN_ROWS + EVAL_ROWS} valid rows, found {len(selected)}"
        )
    splits = {
        "train": Dataset.from_list(selected[:TRAIN_ROWS]),
        "eval": Dataset.from_list(selected[TRAIN_ROWS:]),
    }
    records = {}
    for split, dataset in splits.items():
        artifact = args.output_dir / f"{split}_{len(dataset)}.parquet"
        dataset.to_parquet(artifact)
        records[split] = {
            "artifact": str(artifact.resolve()),
            "rows": len(dataset),
            "source_row_ids": list(dataset["source_row_id"]),
            "sha256": _sha256(artifact),
        }

    manifest = {
        "dataset": DATASET,
        "subset": SUBSET,
        "revision": REVISION,
        "selection": "first 11000 rows passing leap's LFM2.5 DPO filter",
        **records,
    }
    (args.output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
