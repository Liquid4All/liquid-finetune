import json
import logging
from pathlib import Path
from typing import Any, Callable

import fsspec
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from rich.console import Console

from liquid_finetune.data_loading.validate_tool_format import (
    detect_tool_format,
    get_tool_normalizer,
    has_foreign_tool_markers,
    validate_tool_format,
    validate_tool_calls_dpo,
    validate_tool_calls_in_messages,
)

logger = logging.getLogger(__name__)


# ============================================================================
# QUICK VALIDATION (Pre-Ray, ~10 samples)
# ============================================================================


def is_cloud_path(path: str) -> bool:
    """Check if path is a cloud storage path (S3, GCS, Azure)."""
    cloud_prefixes = ("s3://", "gs://", "az://", "abfs://", "abfss://")
    return path.startswith(cloud_prefixes)


def find_local_files(dataset_path: str, *patterns: str) -> list[Path]:
    """Return sorted local files matching any of the given glob patterns."""
    path = Path(dataset_path).expanduser()
    if not path.is_dir():
        return []

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path.glob(pattern))
    return sorted(matches)


def get_source_type(dataset_path: str) -> str:
    """Determine the source type for display and loading logic."""
    expanded_path = Path(dataset_path).expanduser()
    path_str = str(expanded_path)
    path_lower = path_str.lower()

    if is_cloud_path(dataset_path):
        if path_lower.startswith("s3://"):
            return "s3"
        elif path_lower.startswith("gs://"):
            return "gcs"
        elif path_lower.startswith(("az://", "abfs://", "abfss://")):
            return "azure"
        return "cloud"
    elif expanded_path.exists() or dataset_path.startswith(("./", "/", "~")):
        if path_lower.endswith((".parquet", ".pq")):
            return "parquet"
        elif path_lower.endswith(".arrow"):
            return "arrow"
        elif path_lower.endswith(".csv"):
            return "csv"
        elif path_lower.endswith((".json", ".jsonl", ".json.zst")):
            return "json"
        elif expanded_path.is_dir() and find_local_files(path_str, "*.parquet", "*.pq"):
            return "parquet"
        elif expanded_path.is_dir() and find_local_files(path_str, "*.arrow"):
            return "arrow"
        elif expanded_path.is_dir() and find_local_files(path_str, "*.csv"):
            return "csv"
        elif expanded_path.is_dir() and find_local_files(
            path_str, "*.json", "*.jsonl", "*.json.zst"
        ):
            return "json"
        else:
            return "directory"
    else:
        return "huggingface"


def quick_validate_schema(
    dataset_path: str,
    dataset_type: str,
    subset: str | None = None,
    split: str = "train",
    num_samples: int = 10,
    image_root: str | None = None,
    model_name: str | None = None,
) -> None:
    """
    Fast schema validation on small sample. Fails fast on obvious errors.
    Runs in main process before Ray starts.

    Applies normalization first (column renames, JSON parsing, image_root) so
    raw dataset formats are accepted.
    """
    console = Console()

    source_type = get_source_type(dataset_path)
    console.print(
        f"[dim]Validating {source_type} schema ({num_samples} samples)...[/dim]"
    )

    sample_ds = _load_sample_dataset(dataset_path, subset, split, num_samples)

    if len(sample_ds) == 0:
        raise ValueError(f"Dataset appears to be empty: {dataset_path}")

    model_family = "lfm2"
    if model_name and dataset_type in ("sft", "dpo"):
        from liquid_finetune.checkpointing.model_info import get_model_family

        model_family = get_model_family(model_name)

    # Normalize before validation (handles JSON strings, column renames, image_root)
    normalizer = normalize_columns(dataset_type, image_root=image_root)
    sample_ds = sample_ds.map(normalizer)

    # === Tool-Call Format ===
    if model_name and dataset_type in ("sft", "dpo"):
        samples = [sample_ds[i] for i in range(len(sample_ds))]
        format_info = detect_tool_format(samples)
        if format_info.has_tool_calls:
            issues = validate_tool_format(format_info, model_family)
            for issue in issues:
                if issue.severity == "error":
                    raise ValueError(
                        f"Tool call format error: {issue.message}\n"
                        f"Fix: {issue.fix_hint}"
                    )
                if issue.severity == "warning":
                    console.print(f"[yellow]Tool format:[/yellow] {issue.message}")
                else:
                    console.print(f"[dim]Tool format: {issue.message}[/dim]")

        sample_ds = sample_ds.map(get_tool_normalizer(model_family))

    # Use the same validation as full validation
    validate_dataset_format(sample_ds, dataset_type, model_family=model_family)

    console.print("[green]✓ Schema validated[/green]")


def _load_sample_dataset(
    dataset_path: str,
    subset: str | None,
    split: str,
    num_samples: int,
) -> Dataset:
    """Load a small sample as HF Dataset."""
    try:
        source_type = get_source_type(dataset_path)

        if source_type in ("s3", "gcs", "azure", "cloud"):
            # Cloud storage - use fsspec + pyarrow (no Ray needed)
            fs, path = fsspec.core.url_to_fs(dataset_path)
            path_lower = dataset_path.lower()
            if ".parquet" in path_lower or ".pq" in path_lower:
                with fs.open(path, "rb") as f:
                    pf = pq.ParquetFile(f)
                    batch = next(pf.iter_batches(batch_size=num_samples))
                return Dataset.from_pandas(batch.to_pandas())
            else:
                with fs.open(path, "r") as f:
                    rows = [json.loads(line) for _, line in zip(range(num_samples), f)]
                return Dataset.from_list(rows)

        elif source_type in ("parquet", "json", "csv", "arrow"):
            # Local file or directory
            p = Path(dataset_path).expanduser()
            if source_type == "parquet":
                if p.is_dir():
                    # Directory of parquets: read first shard
                    parquet_files = find_local_files(dataset_path, "*.parquet", "*.pq")
                    if not parquet_files:
                        raise ValueError(
                            f"No parquet files found in directory: {dataset_path}"
                        )
                    pf = pq.ParquetFile(parquet_files[0])
                    batch = next(pf.iter_batches(batch_size=num_samples))
                    return Dataset.from_pandas(batch.to_pandas())
                else:
                    return load_dataset(
                        "parquet",
                        data_files=dataset_path,
                        split=f"{split}[:{num_samples}]",
                    )
            elif source_type == "csv":
                return load_dataset(
                    "csv",
                    data_files=dataset_path,
                    split=f"{split}[:{num_samples}]",
                )
            elif source_type == "arrow":
                return load_dataset(
                    "arrow",
                    data_files=dataset_path,
                    split=f"{split}[:{num_samples}]",
                )
            else:
                return load_dataset(
                    "json",
                    data_files=dataset_path,
                    split=f"{split}[:{num_samples}]",
                )

        elif source_type == "directory":
            return load_dataset(dataset_path, subset, split=f"{split}[:{num_samples}]")

        else:
            # HuggingFace Hub - use streaming then convert to Dataset
            samples = []
            split_parts = [part.strip() for part in split.split("+") if part.strip()]
            if not split_parts:
                raise ValueError("Dataset split cannot be empty")
            for split_part in split_parts:
                ds_stream = load_dataset(
                    dataset_path, subset, split=split_part, streaming=True
                )
                for item in ds_stream:
                    if len(samples) >= num_samples:
                        break
                    samples.append(item)
                if len(samples) >= num_samples:
                    break
            return Dataset.from_list(samples)
    except Exception as e:
        raise ValueError(f"Failed to load dataset samples from '{dataset_path}': {e}")


# ============================================================================
# DISTRIBUTED FILTERING (Ray Data native operations)
# ============================================================================


def _is_valid_kto_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, list) or not value:
        return False
    return all(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and bool(message["role"].strip())
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for message in value
    )


def _has_foreign_kto_markers(value: Any) -> bool:
    if isinstance(value, str):
        contents = [value]
    else:
        contents = [
            message["content"] for message in value if message["role"] == "assistant"
        ]
    return any(
        ("<" in content or "[" in content) and has_foreign_tool_markers(content)
        for content in contents
    )


def get_row_filter(
    dataset_type: str,
    model_family: str = "lfm2",
) -> Callable[[dict], bool]:
    """
    Get a row filter function for ray.data.filter().
    Uses pure Python - Ray handles Arrow/serialization internally.
    """

    def is_valid_sft(row: dict) -> bool:
        """Check if row has valid SFT conversational format."""
        # Find the messages column
        messages = None
        for col in ["messages", "conversation", "conversations", "chat", "dialogue"]:
            if col in row and row[col]:
                messages = row[col]
                break

        if not messages or len(messages) == 0:
            return False

        first = messages[0]
        if not (isinstance(first, dict) and "role" in first and "content" in first):
            return False

        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and ("<" in content or "[" in content):
                if has_foreign_tool_markers(content):
                    return False

        return True

    def is_valid_dpo(row: dict) -> bool:
        """Check if row has valid DPO format."""
        chosen = row.get("chosen")
        rejected = row.get("rejected")

        if not chosen or not rejected:
            return False

        if chosen == rejected:
            return False

        for data in (chosen, rejected):
            if isinstance(data, str):
                if ("<" in data or "[" in data) and has_foreign_tool_markers(data):
                    return False
            elif isinstance(data, list):
                for msg in data:
                    if not isinstance(msg, dict) or msg.get("role") != "assistant":
                        continue
                    content = msg.get("content", "")
                    if isinstance(content, str) and ("<" in content or "[" in content):
                        if has_foreign_tool_markers(content):
                            return False

        return True

    def is_valid_kto(row: dict) -> bool:
        """Check if row has valid KTO (unpaired preference) format."""
        prompt = row.get("prompt")
        completion = row.get("completion")
        label = row.get("label")

        if not _is_valid_kto_value(prompt) or not _is_valid_kto_value(completion):
            return False
        if not isinstance(label, bool):
            return False
        return not _has_foreign_kto_markers(completion)

    def is_valid_vlm_sft(row: dict) -> bool:
        """Check if row has valid VLM SFT format with loadable images.

        Validates every message for structure (role, content list, typed items)
        and every image for loadability.
        """
        messages = row.get("messages")
        if not messages or not isinstance(messages, list) or len(messages) == 0:
            return False

        # Import inside function body for Ray serialization compatibility
        from liquid_finetune.data_loading.image_loader import is_image_loadable

        for message in messages:
            if not isinstance(message, dict):
                return False
            if "role" not in message or "content" not in message:
                return False

            content = message["content"]
            if not isinstance(content, list) or len(content) == 0:
                return False

            for item in content:
                if not isinstance(item, dict) or "type" not in item:
                    return False

                item_type = item["type"]
                if item_type == "image":
                    if not isinstance(item.get("image"), str):
                        return False
                    if not is_image_loadable(item["image"]):
                        return False
                elif item_type == "text":
                    if not isinstance(item.get("text"), str):
                        return False
                else:
                    return False

        return True

    def is_valid_vlm_dpo(row: dict) -> bool:
        """Check VLM DPO rows with image path(s) and preference messages."""
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not prompt or not chosen or not rejected:
            return False
        if chosen == rejected:
            return False
        if not (
            isinstance(prompt, list)
            and isinstance(chosen, list)
            and isinstance(rejected, list)
        ):
            return False

        from liquid_finetune.data_loading.image_loader import is_image_loadable

        image = row.get("image")
        images = row.get("images")
        if isinstance(image, str):
            return is_image_loadable(image)
        if isinstance(images, list) and images:
            return all(
                isinstance(item, str) and is_image_loadable(item) for item in images
            )
        return False

    def is_valid_grpo(row: dict) -> bool:
        """Check if row has a valid GRPO format: non-empty `prompt` (str or messages list)."""
        prompt = row.get("prompt")
        if not prompt:
            return False
        # Accept string prompts (standard format) or messages-list prompts
        # (conversational format).
        if isinstance(prompt, str):
            return bool(prompt.strip())
        if isinstance(prompt, list):
            if len(prompt) == 0:
                return False
            first = prompt[0]
            return isinstance(first, dict) and "role" in first and "content" in first
        return False

    def is_valid_vlm_grpo(row: dict) -> bool:
        """VLM GRPO: `prompt` must be a messages list with loadable image items.

        Mirrors `is_valid_vlm_sft` but reads from `prompt` instead of `messages`
        because GRPO's dataset contract uses the `prompt` column name.
        """
        prompt = row.get("prompt")
        if not prompt or not isinstance(prompt, list) or len(prompt) == 0:
            return False

        # Import inside function body for Ray serialization compatibility
        from liquid_finetune.data_loading.image_loader import is_image_loadable

        for message in prompt:
            if not isinstance(message, dict):
                return False
            if "role" not in message or "content" not in message:
                return False

            content = message["content"]
            if not isinstance(content, list) or len(content) == 0:
                return False

            for item in content:
                if not isinstance(item, dict) or "type" not in item:
                    return False
                item_type = item["type"]
                if item_type == "image":
                    if not isinstance(item.get("image"), str):
                        return False
                    if not is_image_loadable(item["image"]):
                        return False
                elif item_type == "text":
                    if not isinstance(item.get("text"), str):
                        return False
                else:
                    return False
        return True

    def is_valid_retrieval(row: dict) -> bool:
        query = row.get("query")
        positive = row.get("positive")
        if not _is_nonempty_text(query) or not _is_nonempty_text(positive):
            return False
        if query.strip() == positive.strip():
            return False

        negative = row.get("negative")
        if negative is None:
            return True
        return _is_nonempty_text(negative) and negative.strip() != positive.strip()

    if dataset_type == "sft":
        return is_valid_sft
    elif dataset_type == "kto":
        return is_valid_kto
    elif dataset_type == "dpo":
        return is_valid_dpo
    elif dataset_type in ("embedding", "colbert"):
        return is_valid_retrieval
    elif dataset_type == "vlm_sft":
        return is_valid_vlm_sft
    elif dataset_type == "vlm_dpo":
        return is_valid_vlm_dpo
    elif dataset_type == "grpo":
        return is_valid_grpo
    elif dataset_type == "vlm_grpo":
        return is_valid_vlm_grpo
    else:
        return lambda row: True


def normalize_columns(dataset_type: str, image_root: str | None = None):
    """
    Get a row transform function to normalize column names and formats.
    For use with ray.data.map() if needed.
    """

    # Column names that should be renamed to 'messages'
    _CONVERSATION_ALIASES = ["conversation", "conversations", "chat", "dialogue"]

    def normalize_sft(row: dict) -> dict:
        # Rename conversation column to 'messages' if needed
        for col in _CONVERSATION_ALIASES:
            if col in row and "messages" not in row:
                row["messages"] = row.pop(col)
                break
        return row

    def normalize_vlm_sft(row: dict) -> dict:
        import json
        import pathlib

        import numpy as np

        # Parquet deserialization returns list columns as ndarrays.
        for key in ("messages", *_CONVERSATION_ALIASES):
            val = row.get(key)
            if isinstance(val, np.ndarray):
                row[key] = val.tolist()

        # === 1. Find and rename conversation column ===
        for col in _CONVERSATION_ALIASES:
            if col in row and "messages" not in row:
                row["messages"] = row.pop(col)
                break

        if "messages" not in row:
            return row

        # === 2. Parse JSON string if needed ===
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
            row["messages"] = messages

        if not isinstance(messages, list):
            return row

        # === 3. Uniformize content shape ===
        # Wrap string content into list-of-parts so the column isn't mixed
        # list/string, which Arrow can't represent downstream.
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                message["content"] = [{"type": "text", "text": message["content"]}]

        # === 4. Prepend image_root to relative image paths ===
        if image_root:
            root = pathlib.PurePosixPath(image_root)
            for message in messages:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "image"
                        and isinstance(item.get("image"), str)
                        and not pathlib.PurePosixPath(item["image"]).is_absolute()
                    ):
                        item["image"] = str(root / item["image"])

        return row

    def normalize_vlm_dpo(row: dict) -> dict:
        import json
        import pathlib

        import numpy as np

        def as_py(value):
            if isinstance(value, np.ndarray):
                return [as_py(v) for v in value.tolist()]
            if isinstance(value, list):
                return [as_py(v) for v in value]
            if isinstance(value, tuple):
                return [as_py(v) for v in value]
            if isinstance(value, dict):
                return {k: as_py(v) for k, v in value.items()}
            return value

        for key in ("prompt", "chosen", "rejected", "images"):
            val = as_py(row.get(key))
            if isinstance(val, str) and key in {
                "prompt",
                "chosen",
                "rejected",
                "images",
            }:
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            row[key] = val

        if image_root:
            root = pathlib.PurePosixPath(image_root)
            image = row.get("image")
            if (
                isinstance(image, str)
                and not pathlib.PurePosixPath(image).is_absolute()
            ):
                row["image"] = str(root / image)

            images = row.get("images")
            if isinstance(images, list):
                row["images"] = [
                    str(root / item)
                    if isinstance(item, str)
                    and not pathlib.PurePosixPath(item).is_absolute()
                    else item
                    for item in images
                ]

        return row

    def add_dpo_prompt(row: dict) -> dict:
        if "prompt" not in row or not row["prompt"]:
            # Extract prompt from chosen
            chosen = row.get("chosen", [])
            if isinstance(chosen, list):
                for msg in chosen:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        row["prompt"] = msg.get("content", "")
                        break
                else:
                    row["prompt"] = ""
            else:
                row["prompt"] = ""
        return row

    def _split_messages(messages: list) -> tuple[list, str | None]:
        """Split an SFT messages list into GRPO (prompt, solution)."""
        prompt_turns: list = []
        assistant_text: str | None = None
        for msg in messages:
            if not isinstance(msg, dict):
                prompt_turns.append(msg)
                continue
            if msg.get("role") != "assistant":
                prompt_turns.append(msg)
                continue
            content = msg.get("content")
            if isinstance(content, str):
                assistant_text = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        assistant_text = item.get("text", "")
                        break
        return prompt_turns, assistant_text

    def normalize_grpo(row: dict) -> dict:
        """Normalize a text GRPO row.

        Same SFT schema works for ``sft`` → ``dpo`` → ``grpo``: the standard
        ``messages`` column is split into ``prompt`` (non-assistant turns)
        and ``solution`` (last assistant text). A pre-existing ``solution``
        column is preserved. Native ``prompt`` and legacy ``question`` /
        ``query`` / ``input`` aliases remain supported.
        """
        import json

        import numpy as np

        for key in ("prompt", "messages", "conversation", "conversations"):
            val = row.get(key)
            if isinstance(val, np.ndarray):
                row[key] = val.tolist()

        source_alias = None
        if row.get("prompt") is None:
            for alias in (
                "messages",
                "conversation",
                "conversations",
                "chat",
                "dialogue",
            ):
                if row.get(alias) is not None:
                    source_alias = alias
                    break

        if source_alias is not None:
            messages = row.pop(source_alias)
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except (json.JSONDecodeError, TypeError):
                    row["prompt"] = messages
                    return row
            if not isinstance(messages, list) or not messages:
                row["prompt"] = messages
                return row
            prompt_turns, assistant_text = _split_messages(messages)
            row["prompt"] = prompt_turns or messages
            if assistant_text is not None and not row.get("solution"):
                row["solution"] = assistant_text

        if row.get("prompt") is None:
            for alias in ("query", "question", "input"):
                if row.get(alias):
                    row["prompt"] = row[alias]
                    break

        return row

    def normalize_vlm_grpo(row: dict) -> dict:
        """Normalize a VLM GRPO row.

        Same as ``normalize_grpo`` — the VLM SFT ``messages`` format is
        split into ``prompt`` (user/system turns) and ``solution`` (last
        assistant text) so the same dataset file drives ``vlm_sft``,
        ``vlm_dpo``, and ``vlm_grpo``.
        """
        import json
        import pathlib

        import numpy as np

        def as_py(value):
            if isinstance(value, np.ndarray):
                return [as_py(item) for item in value.tolist()]
            if isinstance(value, list):
                return [as_py(item) for item in value]
            if isinstance(value, tuple):
                return [as_py(item) for item in value]
            if isinstance(value, dict):
                return {key: as_py(item) for key, item in value.items()}
            return value

        for key in ("prompt", "messages", "conversation", "conversations"):
            if key in row:
                row[key] = as_py(row[key])

        source_alias = None
        if "prompt" not in row:
            for alias in (
                "messages",
                "conversation",
                "conversations",
                "chat",
                "dialogue",
            ):
                if alias in row:
                    source_alias = alias
                    break

        if source_alias is not None:
            messages = row.pop(source_alias)
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except (json.JSONDecodeError, TypeError):
                    row["prompt"] = messages
                    return row
            if not isinstance(messages, list) or not messages:
                row["prompt"] = messages
                return row
            prompt_turns, assistant_text = _split_messages(messages)
            row["prompt"] = prompt_turns or messages
            if assistant_text is not None and not row.get("solution"):
                row["solution"] = assistant_text

        if "prompt" not in row:
            return row

        prompt = row["prompt"]
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
                row["prompt"] = prompt
            except (json.JSONDecodeError, TypeError):
                return row

        if not isinstance(prompt, list):
            return row

        for message in prompt:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                    if isinstance(content, dict):
                        content = [content]
                except (json.JSONDecodeError, TypeError):
                    content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                continue

            normalized_content = []
            for item in content:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except (json.JSONDecodeError, TypeError):
                        item = {"type": "text", "text": item}
                normalized_content.append(item)
            message["content"] = normalized_content

        if image_root:
            root = pathlib.PurePosixPath(image_root)
            for message in prompt:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "image"
                        and isinstance(item.get("image"), str)
                        and not pathlib.PurePosixPath(item["image"]).is_absolute()
                    ):
                        item["image"] = str(root / item["image"])

        return row

    def normalize_retrieval(row: dict) -> dict:
        import numpy as np

        aliases = {
            "anchor": "query",
            "question": "query",
            "positives": "positive",
            "document": "positive",
            "passage": "positive",
            "negatives": "negative",
            "hard_negative": "negative",
            "hard_negatives": "negative",
        }
        for source, target in aliases.items():
            if target not in row and source in row:
                row[target] = row.pop(source)

        negative = row.get("negative")
        if isinstance(negative, np.ndarray):
            row["negative"] = negative.tolist()
        return row

    if dataset_type == "vlm_sft":
        return normalize_vlm_sft
    elif dataset_type == "vlm_dpo":
        return normalize_vlm_dpo
    elif dataset_type == "sft":
        return normalize_sft
    elif dataset_type == "dpo":
        return add_dpo_prompt
    elif dataset_type in ("embedding", "colbert"):
        return normalize_retrieval
    elif dataset_type == "grpo":
        return normalize_grpo
    elif dataset_type == "vlm_grpo":
        return normalize_vlm_grpo
    else:
        return lambda row: row


def _find_messages_column(batch: pd.DataFrame) -> str | None:
    """Find the conversational column in a batch."""
    for col in ["messages", "conversation", "conversations", "chat", "dialogue"]:
        if col in batch.columns:
            return col
    for col in batch.columns:
        if len(batch) > 0 and _is_valid_conversation(batch[col].iloc[0]):
            return col
    return None


def _is_valid_conversation(content: Any) -> bool:
    """Check if content is valid conversational format.

    Accepts both Python lists and numpy arrays (parquet returns ndarray).
    Validates structure: list of dicts with 'role' and 'content' keys.
    """
    import numpy as np

    # Must be a sequence type (list or numpy array)
    if not isinstance(content, (list, np.ndarray)):
        return False

    # Must have at least one message
    if len(content) == 0:
        return False

    # First message must have required fields
    first = content[0]
    if not isinstance(first, dict):
        return False

    return "role" in first and "content" in first


def _extract_prompt(chosen: Any) -> str:
    """Extract prompt from chosen (conversational format)."""
    if isinstance(chosen, list):
        for msg in chosen:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return ""


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_dataset_format(
    dataset: Dataset,
    dataset_type: str,
    model_family: str = "lfm2",
) -> Dataset:
    """Validate and convert dataset format based on dataset_type"""

    if dataset_type == "sft":
        return validate_sft_format(dataset, model_family=model_family)
    elif dataset_type == "dpo":
        return validate_dpo_format(dataset, model_family=model_family)
    elif dataset_type == "kto":
        return validate_kto_format(dataset)
    elif dataset_type in ("embedding", "colbert"):
        return validate_retrieval_format(dataset)
    elif dataset_type == "vlm_sft":
        return validate_vlm_sft_format(dataset)
    elif dataset_type == "vlm_dpo":
        return validate_vlm_dpo_format(dataset)
    elif dataset_type == "grpo":
        return validate_grpo_format(dataset)
    elif dataset_type == "vlm_grpo":
        return validate_vlm_grpo_format(dataset)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_retrieval_format(dataset: Dataset) -> Dataset:
    """Validate query-positive pairs with an optional negative string."""
    columns = dataset.column_names
    missing = [column for column in ("query", "positive") if column not in columns]
    if missing:
        raise ValueError(
            f"Retrieval dataset is missing required column(s) {missing}. Found: {columns}. "
            "Expected non-empty `query` and `positive` strings and an optional negative string."
        )

    invalid: list[int] = []
    for index, row in enumerate(dataset):
        query = row.get("query")
        positive = row.get("positive")
        negative = row.get("negative")
        if (
            not _is_nonempty_text(query)
            or not _is_nonempty_text(positive)
            or query.strip() == positive.strip()
        ):
            invalid.append(index)
            continue
        if negative is None:
            continue
        if not _is_nonempty_text(negative) or negative.strip() == positive.strip():
            invalid.append(index)
    if invalid:
        raise ValueError(
            f"Retrieval dataset has {len(invalid)} invalid row(s), including {invalid[:10]}. "
            "Queries/positives must be distinct non-empty strings; the negative must be a distinct non-empty string."
        )
    return dataset


def validate_grpo_format(dataset: Dataset) -> Dataset:
    """Validate GRPO dataset: must have a `prompt` column (str or messages list).

    GRPO datasets only require `prompt`. Any additional columns are forwarded
    to reward functions as kwargs by TRL.
    """
    columns = dataset.column_names
    if "prompt" not in columns:
        raise ValueError(
            f"GRPO dataset is missing the `prompt` column. Found columns: {columns}. "
            f"Each row must have a 'prompt' field containing either a plain string or a "
            f"list of messages like [{{'role': 'user', 'content': '...'}}]."
        )

    invalid: list[int] = []
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        if not prompt:
            invalid.append(i)
            continue
        if isinstance(prompt, str):
            if not prompt.strip():
                invalid.append(i)
            continue
        if isinstance(prompt, list):
            if len(prompt) == 0:
                invalid.append(i)
                continue
            first = prompt[0]
            if not (isinstance(first, dict) and "role" in first and "content" in first):
                invalid.append(i)
            continue
        # Neither str nor list
        invalid.append(i)

    if invalid:
        shown = invalid[:5]
        msg = f"Found {len(invalid)} GRPO samples with invalid `prompt` field (indices: {shown}"
        if len(invalid) > 5:
            msg += f"... and {len(invalid) - 5} more"
        msg += "). Each prompt must be a non-empty string or a messages list."
        raise ValueError(msg)

    return dataset


def validate_vlm_grpo_format(dataset: Dataset) -> Dataset:
    """Validate VLM GRPO dataset: `prompt` is a messages list with loadable images.

    Mirrors `validate_vlm_sft_format` structure but reads from the `prompt` column.
    """
    columns = dataset.column_names
    if "prompt" not in columns:
        raise ValueError(f"VLM GRPO dataset missing `prompt` column. Found: {columns}")

    sample_indices = [0, min(5, len(dataset) - 1), min(50, len(dataset) - 1)]

    for idx in sample_indices:
        if idx >= len(dataset):
            continue

        sample = dataset[idx]
        prompt = sample["prompt"]

        if not isinstance(prompt, list):
            raise ValueError(
                f"Sample {idx}: VLM GRPO `prompt` must be a list of messages, got {type(prompt)}"
            )
        if len(prompt) == 0:
            raise ValueError(f"Sample {idx}: `prompt` cannot be empty")

        for msg_idx, message in enumerate(prompt):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: must be dict, got {type(message)}"
                )
            if "role" not in message or "content" not in message:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: missing 'role' or 'content'"
                )

            content = message["content"]
            if not isinstance(content, list) or len(content) == 0:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: `content` must be a non-empty list"
                )

            for ci, item in enumerate(content):
                if not isinstance(item, dict) or "type" not in item:
                    raise ValueError(
                        f"Sample {idx}, message {msg_idx}, content {ci}: must be dict with 'type'"
                    )
                item_type = item["type"]

                if item_type == "text":
                    if not isinstance(item.get("text"), str):
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {ci}: 'text' must be str"
                        )
                elif item_type == "image":
                    image_data = item.get("image")
                    if not isinstance(image_data, str):
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {ci}: 'image' must be a "
                            f"path string (got {type(image_data).__name__}). Use paths, not PIL objects."
                        )
                    from liquid_finetune.data_loading.image_loader import (
                        is_image_loadable,
                    )

                    if not is_image_loadable(image_data):
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {ci}: image not loadable: {image_data}"
                        )
                else:
                    raise ValueError(
                        f"Sample {idx}, message {msg_idx}, content {ci}: unsupported type {item_type!r}"
                    )

    logger.info(f"VLM GRPO dataset validation passed: {len(dataset)} samples")
    return dataset


def validate_sft_format(dataset: Dataset, model_family: str = "lfm2") -> Dataset:
    """Validate and convert SFT dataset to proper format."""
    columns = dataset.column_names

    if any(col in columns for col in ["chosen", "rejected"]):
        raise ValueError("This is a DPO dataset, not SFT. Use dataset_type='dpo'")

    # Find the conversational column
    conv_col = _find_conversational_column(dataset, columns)
    if conv_col is None:
        raise ValueError(
            f"No conversational column found. Expected 'messages' with format: "
            f"[{{'role': 'user', 'content': '...'}}]. Found columns: {columns}"
        )

    # Validate ALL samples
    invalid_indices = []
    for i in range(len(dataset)):
        if not _is_valid_conversation(dataset[i][conv_col]):
            invalid_indices.append(i)

    if invalid_indices:
        # Show first few invalid indices
        shown = invalid_indices[:5]
        msg = f"Found {len(invalid_indices)} invalid samples (indices: {shown}"
        if len(invalid_indices) > 5:
            msg += f"... and {len(invalid_indices) - 5} more"
        msg += "). Each message must have 'role' and 'content' fields."
        raise ValueError(msg)

    # === Tool-Call Format ===
    for i in range(len(dataset)):
        validate_tool_calls_in_messages(dataset[i][conv_col], i, model_family)

    # Rename column if needed
    if conv_col != "messages":
        return dataset.rename_column(conv_col, "messages")

    return dataset


def _find_conversational_column(dataset: Dataset, columns: list) -> str | None:
    """Find the column containing conversational data."""
    # Check known column names first
    for col in ["messages", "conversation", "conversations", "chat", "dialogue"]:
        if col in columns and len(dataset) > 0:
            if _is_valid_conversation(dataset[0][col]):
                return col

    # Fall back to checking all columns
    for col in columns:
        if len(dataset) > 0 and _is_valid_conversation(dataset[0].get(col)):
            return col

    return None


def validate_dpo_format(dataset: Dataset, model_family: str = "lfm2") -> Dataset:
    """Validate and convert DPO dataset to proper format."""
    columns = set(dataset.column_names)

    # Check required columns
    if not {"chosen", "rejected"}.issubset(columns):
        raise ValueError(
            f"DPO needs 'chosen' and 'rejected' columns. Found: {list(columns)}"
        )

    # Validate ALL samples
    invalid_indices = []
    identical_indices = []

    for i in range(len(dataset)):
        chosen = dataset[i]["chosen"]
        rejected = dataset[i]["rejected"]

        # Check non-empty
        if not chosen or not rejected:
            invalid_indices.append(i)
            continue

        # Check chosen != rejected
        if chosen == rejected:
            identical_indices.append(i)

    if invalid_indices:
        shown = invalid_indices[:5]
        raise ValueError(
            f"Found {len(invalid_indices)} samples with empty chosen/rejected "
            f"(indices: {shown}{'...' if len(invalid_indices) > 5 else ''})"
        )

    if identical_indices:
        shown = identical_indices[:5]
        raise ValueError(
            f"Found {len(identical_indices)} samples where chosen == rejected "
            f"(indices: {shown}{'...' if len(identical_indices) > 5 else ''})"
        )

    # === Tool-Call Format ===
    for i in range(len(dataset)):
        validate_tool_calls_dpo(
            dataset[i]["chosen"], dataset[i]["rejected"], i, model_family
        )

    # Add prompt if missing
    if "prompt" in columns:
        return dataset

    return dataset.map(lambda x: {**x, "prompt": _extract_prompt(x["chosen"])})


def validate_kto_format(dataset: Dataset) -> Dataset:
    """Validate KTO (unpaired preference) dataset format.

    Expected columns are prompt/completion/label, where prompt and completion
    are strings or message lists and label marks the completion as desirable
    (True) or undesirable (False). TRL's KTOTrainer applies the chat template
    and tokenizes, so rows stay untokenized here.
    """
    columns = set(dataset.column_names)

    required = {"prompt", "completion", "label"}
    if not required.issubset(columns):
        raise ValueError(
            f"KTO needs {sorted(required)} columns. Found: {list(columns)}"
        )

    invalid_indices = []
    bad_label_indices = []

    for i in range(len(dataset)):
        row = dataset[i]
        if (
            not _is_valid_kto_value(row["prompt"])
            or not _is_valid_kto_value(row["completion"])
            or _has_foreign_kto_markers(row["completion"])
        ):
            invalid_indices.append(i)
        if not isinstance(row["label"], bool):
            bad_label_indices.append(i)

    if invalid_indices:
        shown = invalid_indices[:5]
        raise ValueError(
            f"Found {len(invalid_indices)} samples with invalid prompt/completion "
            f"(indices: {shown}{'...' if len(invalid_indices) > 5 else ''})"
        )

    if bad_label_indices:
        shown = bad_label_indices[:5]
        raise ValueError(
            f"Found {len(bad_label_indices)} samples with non-boolean label "
            f"(indices: {shown}{'...' if len(bad_label_indices) > 5 else ''})"
        )

    return dataset


def validate_vlm_dpo_format(dataset: Dataset) -> Dataset:
    """Validate VLM DPO rows for TRL's vision preference collator.

    Expected columns are prompt/chosen/rejected plus either image or images.
    Image paths stay as strings and are opened lazily by the training collator.
    """
    columns = set(dataset.column_names)
    required = {"prompt", "chosen", "rejected"}
    if not required.issubset(columns) or not ({"image", "images"} & columns):
        expected = sorted(required | {"image"})
        raise ValueError(
            f"VLM DPO needs {expected} columns, or 'images' instead of 'image'. "
            f"Found: {list(columns)}"
        )

    from liquid_finetune.data_loading.image_loader import is_image_loadable

    invalid: list[int] = []
    identical: list[int] = []

    def _images_loadable(row: dict) -> bool:
        image = row.get("image")
        if isinstance(image, str):
            return is_image_loadable(image)

        images = row.get("images")
        if isinstance(images, list) and images:
            return all(
                isinstance(item, str) and is_image_loadable(item) for item in images
            )
        return False

    for i in range(len(dataset)):
        row = dataset[i]
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        if not (
            isinstance(prompt, list)
            and isinstance(chosen, list)
            and isinstance(rejected, list)
            and _images_loadable(row)
        ):
            invalid.append(i)
            continue
        if chosen == rejected:
            identical.append(i)

    if invalid:
        shown = invalid[:5]
        raise ValueError(
            f"Found {len(invalid)} invalid VLM DPO samples "
            f"(indices: {shown}{'...' if len(invalid) > 5 else ''})"
        )
    if identical:
        shown = identical[:5]
        raise ValueError(
            f"Found {len(identical)} VLM DPO samples where chosen == rejected "
            f"(indices: {shown}{'...' if len(identical) > 5 else ''})"
        )

    return dataset


def validate_vlm_sft_format(dataset: Dataset) -> Dataset:
    """
    Comprehensive validation VLM dataset format.

    Expected format:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/path/to/image.jpg"},
                    {"type": "text", "text": "What do you see?"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Response..."}]
            }
        ]
    }
    """
    # Check basic structure
    columns = dataset.column_names
    if "messages" not in columns:
        raise ValueError(f"Dataset missing 'messages' column. Found columns: {columns}")

    # Validate a few samples for detailed structure
    sample_indices = [0, min(5, len(dataset) - 1), min(50, len(dataset) - 1)]

    for idx in sample_indices:
        if idx >= len(dataset):
            continue

        sample = dataset[idx]
        messages = sample["messages"]

        # Check messages is a list
        if not isinstance(messages, list):
            raise ValueError(
                f"Sample {idx}: 'messages' must be a list, got {type(messages)}"
            )

        if len(messages) == 0:
            raise ValueError(f"Sample {idx}: 'messages' cannot be empty")

        # Validate each message in the conversation
        for msg_idx, message in enumerate(messages):
            # Check message structure
            if not isinstance(message, dict):
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: message must be dict, got {type(message)}"
                )

            if "role" not in message:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: missing 'role' field"
                )

            if "content" not in message:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: missing 'content' field"
                )

            # Validate role
            role = message["role"]
            if role not in ["user", "assistant", "system"]:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: invalid role '{role}', expected user/assistant/system"
                )

            # Validate content
            content = message["content"]
            if not isinstance(content, list):
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: 'content' must be list, got {type(content)}"
                )

            if len(content) == 0:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: 'content' cannot be empty"
                )

            # Validate each content item
            for content_idx, content_item in enumerate(content):
                if not isinstance(content_item, dict):
                    raise ValueError(
                        f"Sample {idx}, message {msg_idx}, content {content_idx}: must be dict, got {type(content_item)}"
                    )

                if "type" not in content_item:
                    raise ValueError(
                        f"Sample {idx}, message {msg_idx}, content {content_idx}: missing 'type' field"
                    )

                content_type = content_item["type"]

                if content_type == "text":
                    if "text" not in content_item:
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {content_idx}: text content missing 'text' field"
                        )
                    if not isinstance(content_item["text"], str):
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {content_idx}: 'text' must be string"
                        )

                elif content_type == "image":
                    if "image" not in content_item:
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {content_idx}: image content missing 'image' field"
                        )

                    image_data = content_item["image"]

                    # Check if image is a string path (correct) or PIL object (incorrect)
                    if not isinstance(image_data, str):
                        from PIL import Image

                        if isinstance(image_data, Image.Image):
                            raise ValueError(
                                f"Sample {idx}, message {msg_idx}, content {content_idx}: image must be path string, not PIL Image object. Use image paths for Ray Train compatibility."
                            )
                        elif isinstance(image_data, dict):
                            raise ValueError(
                                f"Sample {idx}, message {msg_idx}, content {content_idx}: image is dict {image_data}, expected path string"
                            )
                        else:
                            raise ValueError(
                                f"Sample {idx}, message {msg_idx}, content {content_idx}: image must be path string, got {type(image_data)}"
                            )

                    # Actually try to load the image to verify it's valid
                    from liquid_finetune.data_loading.image_loader import (
                        is_image_loadable,
                    )

                    if not is_image_loadable(image_data):
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {content_idx}: "
                            f"image not loadable: {image_data}"
                        )

                else:
                    raise ValueError(
                        f"Sample {idx}, message {msg_idx}, content {content_idx}: unsupported content type '{content_type}', expected 'text' or 'image'"
                    )

    logger.info(
        f"VLM dataset validation passed: {len(dataset)} samples with expected format"
    )
    return dataset
