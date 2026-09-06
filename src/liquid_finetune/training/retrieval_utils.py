from __future__ import annotations

from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainingArguments
import torch
import torch.distributed as dist

from liquid_finetune.training.utils.config_filter import (
    BASE_RUNTIME_EXCLUDED_KEYS,
    DISTRIBUTED_RUNTIME_EXCLUDED_KEYS,
    MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS,
    MODEL_RUNTIME_EXCLUDED_KEYS,
    filter_runtime_config_kwargs,
)

RETRIEVAL_RUNTIME_EXCLUDED_KEYS = (
    BASE_RUNTIME_EXCLUDED_KEYS
    | MODEL_RUNTIME_EXCLUDED_KEYS
    | DISTRIBUTED_RUNTIME_EXCLUDED_KEYS
    | MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS
    | {
        "loss",
        "mini_batch_size",
        "temperature",
        "gather_across_devices",
    }
)


def build_ir_evaluation_data(dataset: Dataset):
    """Build query/corpus/qrels dictionaries from retrieval pairs or triplets."""
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    document_ids: dict[str, str] = {}

    def add_document(text: str) -> str:
        if text not in document_ids:
            document_id = f"d{len(document_ids)}"
            document_ids[text] = document_id
            corpus[document_id] = text
        return document_ids[text]

    for index, row in enumerate(dataset):
        query_id = f"q{index}"
        queries[query_id] = row["query"]
        relevant_docs[query_id] = {add_document(row["positive"])}
        if row.get("negative") is not None:
            add_document(row["negative"])

    return queries, corpus, relevant_docs


def canonical_retrieval_dataset(dataset: Dataset | None) -> Dataset | None:
    """Keep only positional text columns consumed by Sentence Transformers."""
    if dataset is None:
        return None
    columns = ["query", "positive"]
    if "negative" in dataset.column_names:
        columns.append("negative")
    return dataset.select_columns(columns)


def register_remote_model_for_checkpointing(model) -> None:
    """Include trusted remote model code in Sentence Transformers checkpoints."""
    transformer = model[0]
    auto_model = transformer.auto_model
    auto_map = getattr(auto_model.config, "auto_map", {})
    if "AutoModel" in auto_map:
        auto_model.register_for_auto_class("AutoModel")


def align_retrieval_train_shard(
    dataset: Dataset,
    *,
    per_device_batch_size: int,
    gather_across_devices: bool,
) -> Dataset:
    """Give every rank the same number of complete contrastive batches."""
    if (
        not gather_across_devices
        or not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return dataset

    if per_device_batch_size < 1:
        raise ValueError("per_device_train_batch_size must be positive")

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    minimum_count = torch.tensor(len(dataset), dtype=torch.int64, device=device)
    dist.all_reduce(minimum_count, op=dist.ReduceOp.MIN)
    smallest_shard = int(minimum_count.item())
    usable_count = smallest_shard - (smallest_shard % per_device_batch_size)
    if usable_count < per_device_batch_size:
        raise ValueError(
            "Retrieval training needs at least one complete batch on every worker; "
            f"smallest shard has {smallest_shard} rows and batch size is "
            f"{per_device_batch_size}."
        )
    if usable_count == len(dataset):
        return dataset
    return dataset.select(range(usable_count))


def build_retrieval_training_args(
    train_config: dict,
    *,
    tracker: str,
    job_name: str,
) -> SentenceTransformerTrainingArguments:
    filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=RETRIEVAL_RUNTIME_EXCLUDED_KEYS,
        config_cls=SentenceTransformerTrainingArguments,
    )
    filtered.setdefault(
        "per_device_eval_batch_size",
        filtered.get("per_device_train_batch_size", 1),
    )
    return SentenceTransformerTrainingArguments(
        report_to=tracker,
        run_name=job_name,
        **filtered,
    )
