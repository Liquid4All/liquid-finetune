import math
import pathlib

import pytest
from pylate import models
import yaml
from sentence_transformers import SentenceTransformer

from conftest import requires_gpu, requires_multi_gpu, run_e2e_training

pytestmark = pytest.mark.retrieval
FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _local_retrieval_config(kind, tmp_path):
    """Use a synchronous copy; the shipped configs are SLURM launch configs."""
    config = yaml.safe_load((FIXTURES / f"e2e_{kind}.yaml").read_text())
    dataset_path = pathlib.Path(config["dataset"]["path"])
    if not dataset_path.is_absolute():
        config["dataset"]["path"] = str((FIXTURES / dataset_path).resolve())
    config.pop("slurm", None)
    config_path = tmp_path / f"e2e_{kind}_local.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _assert_retrieval_improved(result):
    assert result is not None
    metrics = result.metrics or {}
    assert math.isfinite(metrics["eval_loss"])
    deltas = {
        key: value
        for key, value in metrics.items()
        if key.startswith("retrieval/delta/")
    }
    assert deltas, f"No baseline-to-final retrieval metrics: {metrics}"
    ndcg_deltas = {key: value for key, value in deltas.items() if "ndcg@10" in key}
    assert ndcg_deltas, f"No NDCG@10 improvement metric: {deltas}"
    assert max(ndcg_deltas.values()) > 0, f"NDCG@10 did not improve: {ndcg_deltas}"


def _assert_checkpoint_reloads(kind, output_dir):
    checkpoints = [path.parent for path in output_dir.rglob("modules.json")]
    assert checkpoints, f"No retrieval checkpoint found under {output_dir}"
    checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
    assert (checkpoint / "modeling_lfm2_bidirectional.py").is_file()

    model_cls = SentenceTransformer if kind == "embedding" else models.ColBERT
    model = model_cls(
        str(checkpoint),
        device="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )
    assert len(model) == 2


@pytest.mark.parametrize("kind", ["embedding", "colbert"])
@requires_gpu
def test_single_gpu_retrieval_training_improves(kind, e2e_output_dir, tmp_path):
    config_path = _local_retrieval_config(kind, tmp_path)
    result = run_e2e_training(str(config_path), e2e_output_dir)
    _assert_retrieval_improved(result)
    _assert_checkpoint_reloads(kind, e2e_output_dir)


@pytest.mark.parametrize("kind", ["embedding", "colbert"])
@requires_multi_gpu
def test_multi_gpu_retrieval_training_improves(
    kind, e2e_output_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("LEAP_NUM_WORKERS", "2")
    config_path = _local_retrieval_config(kind, tmp_path)
    result = run_e2e_training(str(config_path), e2e_output_dir)
    _assert_retrieval_improved(result)
    _assert_checkpoint_reloads(kind, e2e_output_dir)
