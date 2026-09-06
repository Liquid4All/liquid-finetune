import math

import pytest
from pylate import models
from sentence_transformers import SentenceTransformer

from conftest import requires_gpu, requires_multi_gpu, run_e2e_training

pytestmark = pytest.mark.retrieval
FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


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
def test_single_gpu_retrieval_training_improves(kind, e2e_output_dir):
    result = run_e2e_training(str(FIXTURES / f"e2e_{kind}.yaml"), e2e_output_dir)
    _assert_retrieval_improved(result)
    _assert_checkpoint_reloads(kind, e2e_output_dir)


@pytest.mark.parametrize("kind", ["embedding", "colbert"])
@requires_multi_gpu
def test_multi_gpu_retrieval_training_improves(kind, e2e_output_dir, monkeypatch):
    monkeypatch.setenv("LIQUID_NUM_WORKERS", "2")
    result = run_e2e_training(str(FIXTURES / f"e2e_{kind}.yaml"), e2e_output_dir)
    _assert_retrieval_improved(result)
    _assert_checkpoint_reloads(kind, e2e_output_dir)
