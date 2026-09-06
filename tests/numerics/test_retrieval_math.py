from types import SimpleNamespace

from datasets import Dataset

from liquid_finetune.checkpointing.callback import LiquidCheckpointCallback
from liquid_finetune.training.retrieval_utils import (
    align_retrieval_train_shard,
    build_ir_evaluation_data,
)


def test_retrieval_worker_aligns_distributed_shards(monkeypatch):
    import liquid_finetune.training.retrieval_utils as retrieval_utils

    dataset = Dataset.from_list(
        [{"query": str(index), "positive": "p"} for index in range(9)]
    )
    monkeypatch.setattr(retrieval_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(retrieval_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(retrieval_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(retrieval_utils.torch.cuda, "is_available", lambda: False)

    def set_global_minimum(count, op):
        assert op is retrieval_utils.dist.ReduceOp.MIN
        count.fill_(7)

    monkeypatch.setattr(retrieval_utils.dist, "all_reduce", set_global_minimum)
    aligned = align_retrieval_train_shard(
        dataset,
        per_device_batch_size=4,
        gather_across_devices=True,
    )
    assert len(aligned) == 4


def test_ir_evaluation_data_builds_deduplicated_corpus_and_qrels():
    dataset = Dataset.from_list(
        [
            {"query": "q1", "positive": "p", "negative": "n"},
            {"query": "q2", "positive": "p", "negative": "n2"},
        ]
    )
    queries, corpus, relevant_docs = build_ir_evaluation_data(dataset)
    assert queries == {"q0": "q1", "q1": "q2"}
    assert set(corpus.values()) == {"p", "n", "n2"}
    assert relevant_docs["q0"] == relevant_docs["q1"]


def test_retrieval_improvement_metrics_reports_delta():
    state = SimpleNamespace(
        log_history=[
            {"eval_retrieval_cosine_accuracy": 0.25},
            {"eval_loss": 1.0},
            {"eval_retrieval_cosine_accuracy": 0.75},
        ]
    )
    metrics = LiquidCheckpointCallback._retrieval_improvement_metrics(state)
    assert metrics["retrieval/baseline/retrieval_cosine_accuracy"] == 0.25
    assert metrics["retrieval/final/retrieval_cosine_accuracy"] == 0.75
    assert metrics["retrieval/delta/retrieval_cosine_accuracy"] == 0.5
