#!/usr/bin/env python3
"""Materialize the immutable SmolTalk quality train/test subsets."""

from __future__ import annotations

import argparse
import hashlib
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    for split, count in (("train", 1000), ("test", 100)):
        dataset = load_dataset(
            DATASET,
            SUBSET,
            revision=REVISION,
            split=f"{split}[:{count}]",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        dataset = dataset.add_column("source_row_id", list(range(len(dataset))))
        artifact = args.output_dir / f"{split}_{count}.parquet"
        dataset.to_parquet(artifact)
        records[split] = {
            "artifact": str(artifact.resolve()),
            "rows": len(dataset),
            "source_row_ids": list(range(len(dataset))),
            "sha256": _sha256(artifact),
        }

    manifest = {
        "dataset": DATASET,
        "subset": SUBSET,
        "revision": REVISION,
        "selection": "first rows of each published split",
        **records,
    }
    (args.output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
