from datasets import Dataset, load_dataset
from rich.console import Console
from trl.data_utils import pack_dataset

from liquid_finetune.checkpointing.model_info import get_model_family
from liquid_finetune.data_loading.tokenize_data import tokenize_dpo, tokenize_sft
from liquid_finetune.data_loading.validate_dataset_format import (
    get_row_filter,
    get_source_type,
    normalize_columns,
)
from liquid_finetune.data_loading.validate_tool_format import get_tool_normalizer

console = Console()


def _load(loader, path: str, subset: str | None, split: str) -> Dataset:
    source = get_source_type(path)
    if source == "directory" or source == "huggingface":
        if loader.limit is not None and "[" not in split and "+" not in split:
            split = f"{split}[:{loader.limit}]"
        return load_dataset(path, subset, split=split)
    builder = {
        "parquet": "parquet",
        "json": "json",
        "csv": "csv",
        "arrow": "arrow",
    }.get(source)
    if builder is None and source in {"s3", "gcs", "azure", "cloud"}:
        lower = path.lower()
        builder = "parquet" if lower.endswith((".parquet", ".pq")) else "json"
    if builder is None:
        raise ValueError(
            f"Local single-GPU training does not support dataset source: {path}"
        )
    if loader.limit is not None and "[" not in split and "+" not in split:
        split = f"{split}[:{loader.limit}]"
    return load_dataset(builder, data_files=path, split=split)


def _prepare(loader, dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        normalize_columns(loader.dataset_type, image_root=loader.image_root)
    )
    model_family = "lfm2"
    if loader.dataset_type in ("sft", "dpo"):
        model_family = (
            get_model_family(loader.model_name) if loader.model_name else model_family
        )
        dataset = dataset.map(get_tool_normalizer(model_family))
    return dataset.filter(
        get_row_filter(loader.dataset_type, model_family=model_family)
    )


def _split(loader, shuffle: bool, seed: int):
    train = _prepare(
        loader, _load(loader, loader.get_train_path(), loader.subset, loader.split)
    )

    if shuffle:
        train = train.shuffle(seed=seed)

    evaluation = None
    if loader.val_dataset_path is not None or loader.val_split is not None:
        path = loader.get_eval_path()
        if path is not None:
            evaluation = _prepare(
                loader,
                _load(
                    loader,
                    path,
                    loader.val_subset
                    if loader.val_split is not None
                    else loader.subset,
                    loader.val_split or "train",
                ),
            )
    elif loader.test_size is not None:
        eval_count = max(1, int(len(train) * loader.test_size))
        train, evaluation = (
            train.select(range(len(train) - eval_count)),
            train.select(range(len(train) - eval_count, len(train))),
        )
    if len(train) == 0:
        raise ValueError("Dataset is empty after validation")
    if evaluation is not None and len(evaluation) == 0:
        raise ValueError("Validation dataset is empty after validation")
    return train, evaluation


def _tokenize_sft(dataset, tokenizer, config, packing: bool):
    max_length = config.get("max_length", 2048)
    num_proc = config.get("dataset_num_proc")
    dataset = dataset.map(
        tokenize_sft,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "assistant_only_loss": bool(config.get("assistant_only_loss", False)),
            "completion_only_loss": bool(config.get("completion_only_loss", False)),
            "truncate": not config.get("drop_overlength", False),
        },
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    if config.get("drop_overlength", False):
        dataset = dataset.filter(
            lambda row: row["length"] <= max_length, num_proc=num_proc
        )
    if packing:
        columns = [
            name
            for name in ("input_ids", "assistant_masks", "completion_mask")
            if name in dataset.column_names
        ]
        dataset = pack_dataset(
            dataset.select_columns(columns), seq_length=max_length, strategy="bfd"
        )
        dataset = dataset.map(
            lambda row: {"length": len(row["input_ids"])}, num_proc=num_proc
        )
    return dataset


def _tokenize_dpo(dataset, tokenizer, config):
    return dataset.map(
        tokenize_dpo,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_prompt_length": config.get("max_prompt_length"),
            "max_completion_length": config.get("max_completion_length"),
        },
        remove_columns=dataset.column_names,
        num_proc=config.get("dataset_num_proc"),
    )


def create_local_datasets(
    loader, tokenizer=None, training_config=None, shuffle_seed: int = 42
):
    """Build the same validated/tokenized datasets as Ray, using HF Arrow locally."""
    loader.quick_validate()
    config = training_config or {}
    train, evaluation = _split(
        loader, config.get("shuffle_dataset", True), shuffle_seed
    )
    if tokenizer is not None and loader.dataset_type == "sft":
        train = _tokenize_sft(
            train, tokenizer, config, bool(config.get("packing", False))
        )
        if evaluation is not None:
            evaluation = _tokenize_sft(evaluation, tokenizer, config, False)
    elif tokenizer is not None and loader.dataset_type == "dpo":
        train = _tokenize_dpo(train, tokenizer, config)
        if evaluation is not None:
            evaluation = _tokenize_dpo(evaluation, tokenizer, config)
    console.print(
        f"[green]✓ Local dataset ready:[/green] {len(train):,} train"
        + (f" / {len(evaluation):,} eval" if evaluation is not None else "")
    )
    return train, evaluation
