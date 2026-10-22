import pyarrow as pa
from types import SimpleNamespace

import datasets.arrow_dataset

from leap_finetune.data_loading.ray_data_utils import ray_dataset_to_hf


def test_ray_dataset_to_hf_skips_table_fingerprinting(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("table was hashed for a fingerprint")

    monkeypatch.setattr(datasets.arrow_dataset, "generate_fingerprint", fail)
    batches = [
        pa.table({"query": [{"text": "a"}], "id": ["1"]}),
        pa.table({"query": [{"text": "b"}], "id": ["2"]}),
    ]
    shard = SimpleNamespace(iter_batches=lambda batch_format: iter(batches))

    dataset = ray_dataset_to_hf(shard)

    assert dataset["id"] == ["1", "2"]
    assert dataset[1]["query"] == {"text": "b"}
