#!/usr/bin/env python3
"""Materialize immutable 10k/1k ORPO-DPO preference subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

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
        split=f"train[:{TRAIN_ROWS + EVAL_ROWS}]",
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    source = source.add_column("source_row_id", list(range(len(source))))
    splits = {
        "train": source.select(range(TRAIN_ROWS)),
        "eval": source.select(range(TRAIN_ROWS, TRAIN_ROWS + EVAL_ROWS)),
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
        "selection": "published train rows 0:10000 and 10000:11000",
        **records,
    }
    (args.output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
