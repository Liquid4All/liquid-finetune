import json
import logging
from pathlib import Path, PurePosixPath

from liquid_finetune.data_loading.validate_dataset_format import normalize_columns

logger = logging.getLogger(__name__)


def load_benchmark_samples(
    path: str,
    limit: int | None = None,
    format: str | None = None,
    image_root: str | None = None,
) -> list[dict]:
    """Load and normalize benchmark samples from any supported format.

    Applies the same normalization as the training pipeline: renames column
    aliases (``conversation`` -> ``messages``), parses JSON strings, and
    prepends ``image_root`` to relative image paths.
    """
    if format is None:
        format = _detect_format(path)

    loader = _FORMAT_LOADERS.get(format)
    if loader is None:
        raise ValueError(
            f"Unknown format: {format!r}. Available: {sorted(_FORMAT_LOADERS)}"
        )

    samples = loader(path, limit)

    # 1. Normalize first, then change legacy formatted data to HF format
    normalize = normalize_columns("vlm_sft", image_root=image_root)
    samples = [_convert_legacy_to_hf_format(normalize(s), image_root) for s in samples]

    logger.info("Loaded %d samples from %s (format=%s)", len(samples), path, format)
    return samples


def _convert_legacy_to_hf_format(sample: dict, image_root: str | None = None) -> dict:
    """Convert flat string content + separate ``images`` field to structured HF format.

    Handles legacy data where content is a plain string with ``<image>`` placeholders
    and images are in a separate top-level list::

        {"messages": [{"role": "user", "content": "<image>Q"}], "images": ["/path.jpg"]}

    Converts to::

        {"messages": [{"role": "user", "content": [
            {"type": "image", "image": "/path.jpg"},
            {"type": "text", "text": "Q"}
        ]}]}

    Prepends ``image_root`` to relative image paths during conversion.
    Already-structured content (list of dicts) is left unchanged.
    """
    messages = sample.get("messages")
    if not messages or not isinstance(messages, list):
        return sample

    # If first message content is already a list, assume structured format
    if isinstance(messages[0].get("content"), list):
        return sample

    images = sample.get("images", [])
    # Prepend image_root to relative paths in the legacy images list
    if image_root:
        images = [
            str(PurePosixPath(image_root) / p)
            if not PurePosixPath(p).is_absolute()
            else p
            for p in images
        ]

    image_idx = 0

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str):
            continue

        if "<image>" in content:
            parts = []
            for i, text_part in enumerate(content.split("<image>")):
                if i > 0 and image_idx < len(images):
                    parts.append({"type": "image", "image": images[image_idx]})
                    image_idx += 1
                if text_part.strip():
                    parts.append({"type": "text", "text": text_part.strip()})
            msg["content"] = parts
        else:
            msg["content"] = [{"type": "text", "text": content}]

    # Images are now inline — remove the separate field
    sample.pop("images", None)
    return sample


def _detect_format(path: str) -> str:
    """Infer data format from file extension, defaulting to jsonl."""
    p = path.lower()
    if p.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if p.endswith(".json"):
        return "json"
    if p.endswith((".parquet", ".pq")):
        return "parquet"
    if p.endswith(".csv"):
        return "csv"
    if Path(path).is_dir():
        return "parquet"
    return "jsonl"


def _load_jsonl(path: str, limit: int | None) -> list[dict]:
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def _load_json(path: str, limit: int | None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON benchmark file must contain a top-level array")
    return data[:limit] if limit else data


def _load_parquet(path: str, limit: int | None) -> list[dict]:
    import glob
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    p = Path(path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.parquet")))
        table = ds.dataset(files, format="parquet").to_table()
    else:
        table = pq.read_table(path)
    if limit:
        table = table.slice(0, limit)
    return table.to_pylist()


def _load_csv(path: str, limit: int | None) -> list[dict]:
    import pyarrow.csv as csv

    table = csv.read_csv(path)
    if limit:
        table = table.slice(0, limit)
    return table.to_pylist()


_FORMAT_LOADERS: dict[str, callable] = {
    "jsonl": _load_jsonl,
    "json": _load_json,
    "parquet": _load_parquet,
    "csv": _load_csv,
}
