import json
import pathlib
from types import SimpleNamespace

import pyarrow as pa

from leap_finetune.data_processing.fingerprint import preprocessing_fingerprint_data
from leap_finetune.data_processing.importing import (
    load_callable,
    resolve_callable_reference,
)
from leap_finetune.data_processing.pipeline import apply_preprocessing
from leap_finetune.data_loading.ray_data_utils import _write_preprocessing_report


class _Dataset:
    """Small eager stand-in exercising the Ray Dataset transform contract."""

    def __init__(self, rows):
        self.rows = rows

    def count(self):
        return len(self.rows)

    def schema(self):
        return pa.Table.from_pylist(self.rows).schema

    def map(self, function, **_kwargs):
        return _Dataset([function(row) for row in self.rows])

    def filter(self, function, **_kwargs):
        return _Dataset([row for row in self.rows if function(row)])

    def flat_map(self, function, **_kwargs):
        return _Dataset([output for row in self.rows for output in function(row)])

    def map_batches(self, function, **_kwargs):
        output = function(pa.Table.from_pylist(self.rows))
        return _Dataset(output.to_pylist())


def _write_recipe(path: pathlib.Path) -> None:
    path.write_text(
        """
import pyarrow as pa


def add(row, amount, context):
    return {**row, "value": row["value"] + amount, "split": context.split}


def keep(row, minimum):
    return row["value"] >= minimum


def duplicate(row, copies, context):
    return [
        {**row, "copy": index, "seed": context.seed}
        for index in range(copies)
    ]


def batch_scale(batch, factor):
    values = [value.as_py() * factor for value in batch["value"]]
    return batch.set_column(
        batch.schema.get_field_index("value"),
        "value",
        pa.array(values),
    )


def fit_max(dataset, context):
    return {"maximum": max(row["value"] for row in dataset.rows), "split": context.split}


def divide_by_max(row, artifact):
    return {**row, "value": row["value"] / artifact["maximum"], "fit_split": artifact["split"]}
"""
    )


def test_file_callable_resolution_and_loading(tmp_path):
    recipe = tmp_path / "recipe.py"
    _write_recipe(recipe)

    reference = resolve_callable_reference("recipe.py:add", tmp_path)
    function = load_callable(reference)

    assert reference == f"{recipe}:add"
    assert (
        function({"value": 1}, amount=2, context=SimpleNamespace(split="test"))["value"]
        == 3
    )


def test_file_callable_reloads_when_source_changes(tmp_path):
    recipe = tmp_path / "reloadable.py"
    recipe.write_text("def value(row): return 1\n")
    reference = f"{recipe}:value"
    first = load_callable(reference)
    recipe.write_text("def value(row): return 2\n")
    second = load_callable(reference)

    assert first({}) == 1
    assert second({}) == 2


def test_ordered_operations_run_at_selected_hook(tmp_path):
    recipe = tmp_path / "recipe.py"
    _write_recipe(recipe)

    def reference(name):
        return f"{recipe}:{name}"

    operations = [
        {
            "op": "map",
            "callable": reference("add"),
            "kwargs": {"amount": 2},
            "stage": "raw_pre_normalize",
        },
        {
            "op": "filter",
            "callable": reference("keep"),
            "kwargs": {"minimum": 4},
            "stage": "raw_pre_normalize",
        },
        {
            "op": "flat_map",
            "callable": reference("duplicate"),
            "kwargs": {"copies": 2},
            "stage": "raw_pre_normalize",
        },
        {
            "op": "map_batches",
            "callable": reference("batch_scale"),
            "kwargs": {"factor": 10},
            "stage": "raw_pre_normalize",
            "batch_size": 512,
        },
        {
            "op": "map",
            "callable": reference("add"),
            "kwargs": {"amount": 100},
            "stage": "post_validate_pre_tokenize",
        },
    ]

    result = apply_preprocessing(
        _Dataset([{"value": 1}, {"value": 2}]),
        operations,
        stage="raw_pre_normalize",
        split="train",
        seed=17,
        report=(report := []),
    )

    assert result.rows == [
        {"value": 40, "split": "train", "copy": 0, "seed": 17},
        {"value": 40, "split": "train", "copy": 1, "seed": 17},
    ]
    assert [
        (item["op"], item["input_rows"], item["output_rows"]) for item in report
    ] == [
        ("map", 2, 2),
        ("filter", 2, 1),
        ("flat_map", 1, 2),
        ("map_batches", 2, 2),
    ]
    assert report[-1]["row_delta"] == 0
    assert "value" in report[-1]["output_schema"]


def test_two_phase_fit_artifact_is_injected_into_transform(tmp_path):
    recipe = tmp_path / "recipe.py"
    _write_recipe(recipe)
    result = apply_preprocessing(
        _Dataset([{"value": 2}, {"value": 4}]),
        [
            {
                "op": "map",
                "fit_callable": f"{recipe}:fit_max",
                "callable": f"{recipe}:divide_by_max",
            }
        ],
        stage="raw_pre_normalize",
        split="train",
        seed=9,
    )

    assert result.rows == [
        {"value": 0.5, "fit_split": "train"},
        {"value": 1.0, "fit_split": "train"},
    ]


def test_fingerprint_tracks_callable_source_and_graph(tmp_path):
    recipe = tmp_path / "recipe.py"
    _write_recipe(recipe)
    operation = {
        "op": "map",
        "callable": f"{recipe}:add",
        "kwargs": {"amount": 1},
    }

    first = preprocessing_fingerprint_data([operation])
    recipe.write_text(recipe.read_text() + "\n# version two\n")
    second = preprocessing_fingerprint_data([operation])
    changed_kwargs = preprocessing_fingerprint_data(
        [{**operation, "kwargs": {"amount": 2}}]
    )

    assert first[0]["callable_source_sha256"] != second[0]["callable_source_sha256"]
    assert first != changed_kwargs


def test_preprocessing_report_records_cache_and_outputs(tmp_path):
    report_path = tmp_path / "reports" / "preprocessing.json"
    loader = SimpleNamespace(
        preprocessing_report_path=str(report_path),
        dataset_path="data.jsonl",
        subset=None,
        split="train",
        val_dataset_path=None,
        val_subset=None,
        val_split=None,
        preprocessing=[],
    )

    _write_preprocessing_report(
        loader,
        [{"op": "filter", "input_rows": 2, "output_rows": 1}],
        cache_status="miss",
        shuffle_seed=42,
        train_ds=_Dataset([{"value": 1}]),
        eval_ds=None,
    )

    report = json.loads(report_path.read_text())
    assert report["cache_status"] == "miss"
    assert report["seed"] == 42
    assert report["outputs"]["train_rows"] == 1
    assert report["operations"][0]["output_rows"] == 1
