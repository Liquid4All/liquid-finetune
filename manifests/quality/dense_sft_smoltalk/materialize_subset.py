#!/usr/bin/env python3
"""Materialize the immutable 10k/1k SmolTalk quality subsets."""

from __future__ import annotations

import argparse
import hashlib
from functools import partial
import json
from pathlib import Path

from datasets import load_dataset

DATASET = "HuggingFaceTB/smoltalk"
SUBSET = "smol-constraints"
REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_int(value: str, *, minimum: int) -> int:
    parsed = int(value)
    if parsed < minimum:
        raise argparse.ArgumentTypeError(f"expected an integer >= {minimum}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    nonnegative = partial(_bounded_int, minimum=0)
    positive = partial(_bounded_int, minimum=1)
    parser.add_argument("--train-start", type=nonnegative, default=0)
    parser.add_argument("--train-count", type=positive, default=10000)
    parser.add_argument("--test-start", type=nonnegative, default=0)
    parser.add_argument("--test-count", type=positive, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    selections = (
        ("train", args.train_start, args.train_count),
        ("test", args.test_start, args.test_count),
    )
    for split, start, count in selections:
        stop = start + count
        dataset = load_dataset(
            DATASET,
            SUBSET,
            revision=REVISION,
            split=f"{split}[{start}:{stop}]",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        source_row_ids = list(range(start, start + len(dataset)))
        dataset = dataset.add_column("source_row_id", source_row_ids)
        suffix = str(count) if start == 0 else f"{start}_{stop}"
        artifact = args.output_dir / f"{split}_{suffix}.parquet"
        dataset.to_parquet(artifact)
        records[split] = {
            "artifact": str(artifact.resolve()),
            "rows": len(dataset),
            "source_row_ids": source_row_ids,
            "sha256": _sha256(artifact),
        }

    manifest = {
        "dataset": DATASET,
        "subset": SUBSET,
        "revision": REVISION,
        "selection": "contiguous rows of each published split",
        **records,
    }
    (args.output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
