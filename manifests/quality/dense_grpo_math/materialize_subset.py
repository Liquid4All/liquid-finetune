#!/usr/bin/env python3
"""Materialize immutable 10k/1k numeric OpenR1-Math GRPO subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import Dataset, load_dataset

DATASET = "open-r1/OpenR1-Math-220k"
SUBSET = "default"
REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"
TRAIN_ROWS = 10_000
EVAL_ROWS = 1_000
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_answer(value) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().rstrip(".").replace(",", "").replace("$", "")
    normalized = normalized.replace(" ", "")
    return normalized if _NUMBER.fullmatch(normalized) else None


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
        streaming=True,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    selected = []
    source_row_ids = []
    for source_row_id, row in enumerate(source):
        answer = _numeric_answer(row.get("answer"))
        problem = row.get("problem")
        if answer is None or not isinstance(problem, str) or not problem.strip():
            continue
        selected.append(
            {"prompt": problem, "solution": answer, "source_row_id": source_row_id}
        )
        source_row_ids.append(source_row_id)
        if len(selected) == TRAIN_ROWS + EVAL_ROWS:
            break

    if len(selected) != TRAIN_ROWS + EVAL_ROWS:
        raise RuntimeError(
            f"Expected {TRAIN_ROWS + EVAL_ROWS} numeric rows, found {len(selected)}"
        )

    records = {}
    splits = {
        "train": selected[:TRAIN_ROWS],
        "eval": selected[TRAIN_ROWS:],
    }
    for split, rows in splits.items():
        artifact = args.output_dir / f"{split}_{len(rows)}.parquet"
        Dataset.from_list(rows).to_parquet(artifact)
        records[split] = {
            "artifact": str(artifact.resolve()),
            "rows": len(rows),
            "source_row_ids": [row["source_row_id"] for row in rows],
            "sha256": _sha256(artifact),
        }

    manifest = {
        "dataset": DATASET,
        "subset": SUBSET,
        "revision": REVISION,
        "selection": "first 11000 rows with scalar numeric answers",
        **records,
    }
    (args.output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
