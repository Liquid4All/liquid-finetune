"""VLM rows from a local .jsonl: content parts may reach the pipeline as JSON strings."""

import json

import pytest
from datasets import load_dataset
from PIL import Image

from liquid_finetune.data_loading.validate_dataset_format import (
    get_row_filter,
    normalize_columns,
)

pytestmark = pytest.mark.configs


@pytest.fixture
def jsonl_row(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(tmp_path / "a.png")},
                    {"type": "text", "text": "What is this?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "An image."}]},
        ]
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def test_rows_read_the_way_training_reads_them_are_accepted(jsonl_row):
    # Training hands the Arrow table straight to the pipeline (no per-row decoding).
    raw = load_dataset("json", data_files=str(jsonl_row), split="train")
    row = raw.data.table.slice(0, 1).to_pylist()[0]
    normalized = normalize_columns("vlm_sft")(row)
    assert all(
        isinstance(part, dict)
        for message in normalized["messages"]
        for part in message["content"]
    )
    assert get_row_filter("vlm_sft")(normalized)


def test_json_string_parts_are_parsed(jsonl_row):
    row = json.loads(jsonl_row.read_text())
    row["messages"][0]["content"] = [
        json.dumps(part) for part in row["messages"][0]["content"]
    ]
    normalized = normalize_columns("vlm_sft")(row)
    assert normalized["messages"][0]["content"][0]["type"] == "image"
    assert get_row_filter("vlm_sft")(normalized)


def test_parts_that_are_not_json_objects_are_still_rejected(jsonl_row):
    row = json.loads(jsonl_row.read_text())
    row["messages"][0]["content"] = ["plain text", "[1, 2]"]
    normalized = normalize_columns("vlm_sft")(row)
    assert normalized["messages"][0]["content"] == ["plain text", "[1, 2]"]
    assert not get_row_filter("vlm_sft")(normalized)
