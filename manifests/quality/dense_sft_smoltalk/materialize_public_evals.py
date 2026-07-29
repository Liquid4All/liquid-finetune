#!/usr/bin/env python3
"""Materialize pinned 100-row GSM8K and balanced MMLU eval subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

from datasets import Dataset, load_dataset

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
LETTERS = "ABCD"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gsm8k_rows(cache_dir: Path | None) -> list[dict]:
    source = load_dataset(
        "openai/gsm8k",
        "main",
        revision=GSM8K_REVISION,
        split="test[:100]",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Solve the problem. Show concise reasoning and end with "
                        f"`#### <number>`.\n\n{row['question']}"
                    ),
                },
                {"role": "assistant", "content": row["answer"]},
            ],
            "source_row_id": index,
        }
        for index, row in enumerate(source)
    ]


def mmlu_rows(cache_dir: Path | None) -> list[dict]:
    source = load_dataset(
        "cais/mmlu",
        "all",
        revision=MMLU_REVISION,
        split="test",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    by_subject: dict[str, deque] = defaultdict(deque)
    for index, row in enumerate(source):
        by_subject[row["subject"]].append((index, row))

    selected = []
    subjects = sorted(by_subject)
    while len(selected) < 100:
        made_progress = False
        for subject in subjects:
            if by_subject[subject] and len(selected) < 100:
                selected.append(by_subject[subject].popleft())
                made_progress = True
        if not made_progress:
            break

    rows = []
    for source_row_id, row in selected:
        choices = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"])
        )
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Answer the multiple-choice question with only the "
                            f"letter of the correct option.\n\n{row['question']}\n{choices}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": LETTERS[int(row["answer"])],
                    },
                ],
                "source_row_id": source_row_id,
                "subject": row["subject"],
            }
        )
    return rows


def write_subset(output_dir: Path, name: str, rows: list[dict]) -> dict:
    artifact = output_dir / f"{name}_100.parquet"
    Dataset.from_list(rows).to_parquet(artifact)
    return {
        "artifact": str(artifact.resolve()),
        "rows": len(rows),
        "source_row_ids": [row["source_row_id"] for row in rows],
        "sha256": sha256(artifact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "gsm8k": {
            "dataset": "openai/gsm8k",
            "subset": "main",
            "split": "test",
            "revision": GSM8K_REVISION,
            "selection": "first 100 published rows",
            **write_subset(args.output_dir, "gsm8k", gsm8k_rows(args.cache_dir)),
        },
        "mmlu": {
            "dataset": "cais/mmlu",
            "subset": "all",
            "split": "test",
            "revision": MMLU_REVISION,
            "selection": "subject-sorted round robin until 100 rows",
            **write_subset(args.output_dir, "mmlu_balanced", mmlu_rows(args.cache_dir)),
        },
    }
    (args.output_dir / "public_eval_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
