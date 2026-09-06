import hashlib
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import ray
import ray.data
import pyarrow as pa
from datasets import Dataset
from rich.console import Console

from liquid_finetune import TOKENIZATION_CACHE_DIR

from .dataset_loader import DatasetLoader
from .tokenize_data import tokenize_and_pack_sft, tokenize_dpo_dataset
from .validate_dataset_format import get_row_filter, normalize_columns
from .validate_tool_format import get_tool_normalizer

logger = logging.getLogger(__name__)


_CACHE_SUCCESS_MARKER = "_SUCCESS"
_TOKENIZATION_CACHE_FORMAT_VERSION = 4


def _shuffle_raw_dataset(
    ds: ray.data.Dataset, shuffle_seed: int, training_config: dict
) -> ray.data.Dataset:
    if not training_config.get("shuffle_dataset", True):
        return ds
    return ds.random_shuffle(seed=shuffle_seed)


def create_ray_datasets(
    loader: DatasetLoader,
    shuffle_seed: int = 42,
    tokenizer=None,
    training_config: dict | None = None,
) -> tuple[ray.data.Dataset, ray.data.Dataset | None]:
    """
    Create validated, shuffled, split Ray Datasets from a DatasetLoader.

    Pipeline: quick_validate → load → normalize → filter → shuffle → split → [tokenize/pack]

    When tokenizer is provided, tokenization and optional packing happen
    centrally before sharding, producing equal-length shards (±1 row).
    Tokenized results are cached as Parquet for subsequent runs.
    """
    console = Console()
    use_pretokenize = tokenizer is not None and training_config is not None
    can_cache = use_pretokenize and loader.cache_dataset
    fingerprint, cache_key = _build_tokenization_cache_key_if_needed(
        loader,
        shuffle_seed=shuffle_seed,
        tokenizer=tokenizer,
        training_config=training_config,
        enabled=can_cache,
    )

    # === Tokenization Cache ===
    if fingerprint is not None:
        cached = _load_tokenization_cache(fingerprint)
        if cached is not None:
            train_ds, eval_ds = cached
            _print_cache_hit(console, fingerprint, train_ds, eval_ds)
            return train_ds, eval_ds
        logger.info(
            "Tokenization cache miss (%s), will tokenize and cache", fingerprint
        )

    # === Load / Normalize / Filter / Split ===
    train_ds, eval_ds = _load_and_prepare_datasets(
        loader, shuffle_seed, console, training_config or {}
    )

    # === Tokenize / Pack ===
    if use_pretokenize:
        train_ds, eval_ds = _tokenize_datasets(
            train_ds,
            eval_ds,
            loader=loader,
            tokenizer=tokenizer,
            training_config=training_config,
            console=console,
        )

    # === Save Tokenization Cache ===
    if fingerprint is not None:
        try:
            _save_tokenization_cache(fingerprint, train_ds, eval_ds, cache_key)
            console.print(f"[dim]Cached tokenized data ({fingerprint})[/dim]")
        except Exception:
            logger.warning(
                "Failed to write tokenization cache, continuing without cache"
            )

    return train_ds, eval_ds


def ray_dataset_to_hf(ray_ds) -> Dataset | None:
    """
    Convert a Ray dataset shard to a HuggingFace Dataset.

    This materializes only the Ray shard assigned to the current worker. Keep
    large full-dataset materialization in Ray/cache paths instead of local temp
    directories.
    """
    if ray_ds is None:
        return None

    tables = []
    if hasattr(ray_ds, "to_arrow_refs"):
        arrow_refs = ray_ds.to_arrow_refs()
        if arrow_refs:
            tables = ray.get(arrow_refs)
    elif hasattr(ray_ds, "iter_batches"):
        for batch in ray_ds.iter_batches(batch_format="pyarrow"):
            if isinstance(batch, pa.Table):
                tables.append(batch)
            elif isinstance(batch, pa.RecordBatch):
                tables.append(pa.Table.from_batches([batch]))
            else:
                raise TypeError(
                    "Expected pyarrow batches from Ray dataset shard, "
                    f"got {type(batch).__name__}"
                )
    else:
        raise TypeError(f"Unsupported Ray dataset shard type: {type(ray_ds).__name__}")

    if not tables:
        return Dataset.from_dict({})
    return Dataset(pa.concat_tables(tables))


# === Dataset Pipeline ===


def _load_and_prepare_datasets(
    loader: DatasetLoader,
    shuffle_seed: int,
    console: Console,
    training_config: dict,
) -> tuple[ray.data.Dataset, ray.data.Dataset | None]:
    """Load raw data, normalize/filter rows, shuffle, and build train/eval splits."""

    if not _should_skip_quick_validate():
        loader.quick_validate()
    if loader.dataset_type in ("vlm_sft", "vlm_dpo"):
        console.print(
            "[dim]Filtering VLM samples (validating images across workers)...[/dim]"
        )

    if loader.val_dataset_path is not None or loader.val_split is not None:
        return _load_explicit_train_eval_datasets(
            loader, shuffle_seed, console, training_config
        )

    ds = _shuffle_raw_dataset(
        _normalize_and_filter_dataset(
            loader,
            loader.to_ray_dataset(
                dataset_path=loader.get_train_path(),
                subset=loader.subset,
                split=loader.split,
            ),
        ),
        shuffle_seed,
        training_config,
    )
    total_count = ds.count()

    if total_count == 0:
        raise ValueError(
            f"Dataset is empty after validation. Check if your data matches "
            f"the expected format for dataset_type='{loader.dataset_type}'"
        )

    if loader.test_size is None:
        train_ds = ds
        eval_ds = None
        console.print(f"[green]✓ Dataset ready:[/green] {total_count:,} train samples")
        return train_ds, eval_ds

    eval_count = max(1, int(total_count * loader.test_size))
    train_count = total_count - eval_count

    if train_count < 1:
        raise ValueError(
            f"Not enough samples ({total_count}) for train/eval split "
            f"with test_size={loader.test_size}"
        )

    train_ds, eval_ds = ds.split_at_indices([train_count])
    console.print(
        f"[green]✓ Dataset ready:[/green] {total_count:,} samples "
        f"(train: {train_count:,}, eval: {eval_count:,})"
    )
    return train_ds, eval_ds


def _normalize_and_filter_dataset(
    loader: DatasetLoader,
    ds: ray.data.Dataset,
) -> ray.data.Dataset:
    """Normalize and filter a raw Ray dataset."""
    # Normalize column names/formats before filtering
    # (handles JSON string conversations, column renames, image_root prefix)
    normalizer = normalize_columns(loader.dataset_type, image_root=loader.image_root)
    ds = ds.map(normalizer)

    model_family = "lfm2"
    if loader.dataset_type in ("sft", "dpo"):
        from liquid_finetune.checkpointing.model_info import get_model_family

        model_family = (
            get_model_family(loader.model_name) if loader.model_name else "lfm2"
        )
        ds = ds.map(get_tool_normalizer(model_family))

    # Filter invalid rows using Ray's native filter (pure Python, Ray handles Arrow)
    row_filter = get_row_filter(loader.dataset_type, model_family=model_family)
    return ds.filter(row_filter)


def _load_explicit_train_eval_datasets(
    loader: DatasetLoader,
    shuffle_seed: int,
    console: Console,
    training_config: dict,
) -> tuple[ray.data.Dataset, ray.data.Dataset | None]:
    """Load train and optional eval datasets from explicit source configuration."""
    train_ds = _shuffle_raw_dataset(
        _normalize_and_filter_dataset(
            loader,
            loader.to_ray_dataset(
                dataset_path=loader.get_train_path(),
                subset=loader.subset,
                split=loader.split,
            ),
        ),
        shuffle_seed,
        training_config,
    )
    train_count = train_ds.count()
    if train_count == 0:
        raise ValueError(
            f"Training dataset is empty after validation for path '{loader.get_train_path()}'"
        )

    eval_path = loader.get_eval_path()
    eval_ds = None
    eval_count = 0
    if eval_path is not None:
        eval_ds = _normalize_and_filter_dataset(
            loader,
            loader.to_ray_dataset(
                dataset_path=eval_path,
                subset=loader.val_subset
                if loader.val_split is not None
                else loader.subset,
                split=loader.val_split or "train",
            ),
        )
        eval_count = eval_ds.count()
        if eval_count == 0:
            raise ValueError(
                f"Validation dataset is empty after validation for path '{eval_path}'"
            )

    total_count = train_count + eval_count
    if eval_ds is None:
        console.print(f"[green]✓ Dataset ready:[/green] {train_count:,} train samples")
    else:
        console.print(
            f"[green]✓ Dataset ready:[/green] {total_count:,} samples "
            f"(train: {train_count:,}, eval: {eval_count:,})"
        )
    return train_ds, eval_ds


def _tokenize_datasets(
    train_ds: ray.data.Dataset,
    eval_ds: ray.data.Dataset | None,
    *,
    loader: DatasetLoader,
    tokenizer,
    training_config: dict,
    console: Console,
) -> tuple[ray.data.Dataset, ray.data.Dataset | None]:
    """Tokenize/pack train and eval Ray datasets after train/eval splitting."""
    dataset_type = loader.dataset_type

    if dataset_type == "sft":
        max_length = training_config.get("max_length", 2048)
        packing = training_config.get("packing", False)
        drop_overlength = training_config.get("drop_overlength", False)
        assistant_only_loss = training_config.get("assistant_only_loss", False)
        completion_only_loss = training_config.get("completion_only_loss", False)

        console.print(
            "[dim]Tokenizing SFT "
            f"(max_length={max_length}, packing={packing}, "
            f"drop_overlength={drop_overlength})...[/dim]"
        )
        train_ds = tokenize_and_pack_sft(
            train_ds,
            tokenizer,
            max_length,
            packing,
            assistant_only_loss=assistant_only_loss,
            completion_only_loss=completion_only_loss,
            drop_overlength=drop_overlength,
        )
        if eval_ds is not None:
            eval_ds = tokenize_and_pack_sft(
                eval_ds,
                tokenizer,
                max_length,
                packing=packing,
                assistant_only_loss=assistant_only_loss,
                completion_only_loss=completion_only_loss,
                drop_overlength=drop_overlength,
            )
        return train_ds, eval_ds

    if dataset_type == "kto":
        # TRL's KTOTrainer tokenizes and derives its KL-pair columns itself
        # (per worker shard), so KTO rows pass through untokenized.
        console.print("[dim]KTO: deferring tokenization to KTOTrainer workers[/dim]")
        return train_ds, eval_ds

    if dataset_type == "dpo":
        max_prompt_length = training_config.get("max_prompt_length")
        max_completion_length = training_config.get("max_completion_length")

        console.print("[dim]Tokenizing DPO...[/dim]")
        train_ds = tokenize_dpo_dataset(
            train_ds, tokenizer, max_prompt_length, max_completion_length
        )
        if eval_ds is not None:
            eval_ds = tokenize_dpo_dataset(
                eval_ds, tokenizer, max_prompt_length, max_completion_length
            )

    return train_ds, eval_ds


# === Tokenization Cache ===


def _build_tokenization_cache_key_if_needed(
    loader: DatasetLoader,
    *,
    shuffle_seed: int,
    tokenizer,
    training_config: dict | None,
    enabled: bool,
) -> tuple[str | None, dict | None]:
    """Build a tokenization cache key only when cache lookup/write is enabled."""
    if not enabled:
        return None, None
    return _build_tokenization_cache_key(
        loader,
        shuffle_seed=shuffle_seed,
        tokenizer_id=tokenizer.name_or_path,
        tokenizer_chat_template=getattr(tokenizer, "chat_template", None),
        training_config=training_config,
    )


def _print_cache_hit(
    console: Console,
    fingerprint: str,
    train_ds: ray.data.Dataset,
    eval_ds: ray.data.Dataset | None,
) -> None:
    train_count = train_ds.count()
    if eval_ds is None:
        console.print(
            f"[green]✓ Cache hit[/green] [dim]({fingerprint})[/dim]: "
            f"{train_count:,} train samples"
        )
        return

    eval_count = eval_ds.count()
    console.print(
        f"[green]✓ Cache hit[/green] [dim]({fingerprint})[/dim]: "
        f"{train_count + eval_count:,} samples "
        f"(train: {train_count:,}, eval: {eval_count:,})"
    )


def _build_tokenization_cache_key(
    loader: DatasetLoader,
    *,
    shuffle_seed: int,
    tokenizer_id: str,
    training_config: dict,
    tokenizer_chat_template=None,
) -> tuple[str, dict]:
    """Build a deterministic cache key from all parameters affecting tokenized output."""
    dataset_type = loader.dataset_type

    key = {
        "cache_format_version": _TOKENIZATION_CACHE_FORMAT_VERSION,
        "dataset_path": loader.dataset_path,
        "subset": loader.subset,
        "split": loader.split,
        "val_dataset_path": loader.val_dataset_path,
        "val_subset": loader.val_subset,
        "val_split": loader.val_split,
        "limit": loader.limit,
        "test_size": loader.test_size,
        "dataset_type": dataset_type,
        "tokenizer": tokenizer_id,
        "shuffle_seed": shuffle_seed,
    }

    if dataset_type == "sft":
        key["max_length"] = training_config.get("max_length", 2048)
        key["packing"] = training_config.get("packing", False)
        key["drop_overlength"] = training_config.get("drop_overlength", False)
        key["assistant_only_loss"] = training_config.get("assistant_only_loss", False)
        key["completion_only_loss"] = training_config.get("completion_only_loss", False)
        key["shuffle_dataset"] = training_config.get("shuffle_dataset", True)
        key["chat_template"] = training_config.get("chat_template")
        key["chat_template_path"] = training_config.get("chat_template_path")
        key["chat_template_path_sha256"] = _hash_text_file(
            training_config.get("chat_template_path")
        )
    elif dataset_type == "dpo":
        key["max_prompt_length"] = training_config.get("max_prompt_length")
        key["max_completion_length"] = training_config.get("max_completion_length")
        key["shuffle_dataset"] = training_config.get("shuffle_dataset", True)
        key["chat_template_sha256"] = _hash_template(tokenizer_chat_template)
    elif dataset_type == "kto":
        # KTO caches raw (untokenized) rows; only load/shuffle params matter.
        key["shuffle_dataset"] = training_config.get("shuffle_dataset", True)

    canonical = json.dumps(key, sort_keys=True)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return fingerprint, key


def _load_tokenization_cache(
    fingerprint: str,
) -> tuple[ray.data.Dataset, ray.data.Dataset | None] | None:
    """Load cached train/eval parquet if a complete cache directory exists."""
    cache_dir = TOKENIZATION_CACHE_DIR / fingerprint
    train_dir = cache_dir / "train"
    eval_dir = cache_dir / "eval"
    metadata_path = cache_dir / "fingerprint.json"
    success_marker = cache_dir / _CACHE_SUCCESS_MARKER

    if (
        not train_dir.exists()
        or not metadata_path.exists()
        or not success_marker.exists()
    ):
        return None

    try:
        metadata = json.loads(metadata_path.read_text())
        has_eval = metadata.get("has_eval", eval_dir.exists())
        if has_eval and not eval_dir.exists():
            logger.warning(
                "Tokenization cache %s is incomplete: missing eval dir %s",
                fingerprint,
                eval_dir,
            )
            return None

        train_ds = ray.data.read_parquet(str(train_dir))
        eval_ds = ray.data.read_parquet(str(eval_dir)) if has_eval else None
        return train_ds, eval_ds
    except Exception as exc:
        logger.warning(
            "Failed to read tokenization cache %s from %s, will re-tokenize: %s",
            fingerprint,
            cache_dir,
            exc,
        )
        return None


def _save_tokenization_cache(
    fingerprint: str,
    train_ds: ray.data.Dataset,
    eval_ds: ray.data.Dataset | None,
    key_dict: dict,
) -> None:
    """Write train/eval datasets as parquet into the cache directory atomically."""
    cache_dir = TOKENIZATION_CACHE_DIR / fingerprint
    success_marker = cache_dir / _CACHE_SUCCESS_MARKER
    if cache_dir.exists() and success_marker.exists():
        return

    temp_cache_dir = TOKENIZATION_CACHE_DIR / (
        f"{fingerprint}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    if temp_cache_dir.exists():
        shutil.rmtree(temp_cache_dir)
    temp_cache_dir.mkdir(parents=True, exist_ok=True)

    train_ds.write_parquet(str(temp_cache_dir / "train"))
    if eval_ds is not None:
        eval_ds.write_parquet(str(temp_cache_dir / "eval"))

    metadata = dict(key_dict)
    metadata["has_eval"] = eval_ds is not None
    (temp_cache_dir / "fingerprint.json").write_text(json.dumps(metadata, indent=2))
    (temp_cache_dir / _CACHE_SUCCESS_MARKER).write_text("")

    if cache_dir.exists():
        if success_marker.exists():
            shutil.rmtree(temp_cache_dir)
            return
        shutil.rmtree(cache_dir)
    Path(temp_cache_dir).rename(cache_dir)


def _hash_text_file(path: str | os.PathLike | None) -> str | None:
    """Return a stable content hash for local text files used in tokenization."""
    if not path:
        return None

    candidate = Path(path).expanduser()
    if not candidate.exists() or not candidate.is_file():
        return None
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def _hash_template(template) -> str | None:
    """Hash tokenizer chat templates without storing full template text in metadata."""
    if template is None:
        return None
    if isinstance(template, str):
        template_text = template
    else:
        template_text = json.dumps(template, sort_keys=True, default=str)
    return hashlib.sha256(template_text.encode()).hexdigest()


# === Environment ===


def _should_skip_quick_validate() -> bool:
    return os.environ.get("LIQUID_SKIP_QUICK_VALIDATE", "").lower() in {
        "1",
        "true",
        "yes",
    }
