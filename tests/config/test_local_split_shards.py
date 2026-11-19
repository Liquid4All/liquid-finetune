"""Local parquet directories whose shards are named by split."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from liquid_finetune.data_loading.local_data import _load
from liquid_finetune.data_loading.validate_dataset_format import (
    find_local_files,
    split_shards,
)

pytestmark = pytest.mark.configs


def _write(path, rows):
    pq.write_table(pa.table({"text": rows}), path)


@pytest.fixture
def split_dir(tmp_path):
    _write(tmp_path / "train-00000-of-00002.parquet", ["t0", "t1"])
    _write(tmp_path / "train-00001-of-00002.parquet", ["t2"])
    _write(tmp_path / "validation-00000-of-00001.parquet", ["v0"])
    return tmp_path


def _names(files):
    return [file.name for file in files]


def test_split_shards_selects_requested_split(split_dir):
    files = find_local_files(str(split_dir), "*.parquet")
    assert _names(split_shards(files, "train")) == [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ]
    assert _names(split_shards(files, "validation")) == [
        "validation-00000-of-00001.parquet"
    ]
    assert _names(split_shards(files, "validation[:10]")) == [
        "validation-00000-of-00001.parquet"
    ]


def test_split_shards_leaves_plain_shard_directories_alone(tmp_path):
    _write(tmp_path / "part-0.parquet", ["a"])
    _write(tmp_path / "part-1.parquet", ["b"])
    files = find_local_files(str(tmp_path), "*.parquet")
    assert split_shards(files, "train") is None


def test_split_shards_falls_back_when_split_has_no_named_files(tmp_path):
    _write(tmp_path / "validation-00000.parquet", ["v0"])
    files = find_local_files(str(tmp_path), "*.parquet")
    assert split_shards(files, "train") is None


class _Loader:
    limit = None


def test_local_loader_reads_only_the_requested_split(split_dir):
    train = _load(_Loader(), str(split_dir), None, "train")
    validation = _load(_Loader(), str(split_dir), None, "validation")
    assert sorted(train["text"]) == ["t0", "t1", "t2"]
    assert validation["text"] == ["v0"]
