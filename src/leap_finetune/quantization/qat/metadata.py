from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch.nn as nn

QAT_METADATA_NAME = "qat_config.json"
QAT_METADATA_VERSION = 1


def find_qat_config(model: nn.Module) -> dict[str, Any] | None:
    direct = getattr(model, "_leap_qat_config", None)
    if direct is not None:
        return dict(direct)
    for module in model.modules():
        value = getattr(module, "_leap_qat_config", None)
        if value is not None:
            return dict(value)
    return None


def write_qat_metadata(path: str | os.PathLike, model_or_config) -> Path | None:
    config = (
        find_qat_config(model_or_config)
        if isinstance(model_or_config, nn.Module)
        else dict(model_or_config or {})
    )
    if not config:
        return None
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / QAT_METADATA_NAME
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"format_version": QAT_METADATA_VERSION, **config}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def load_qat_metadata(path: str | os.PathLike) -> dict[str, Any] | None:
    target = Path(path) / QAT_METADATA_NAME
    if not target.is_file():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload.pop("format_version", None)
    return payload


def validate_qat_resume(
    checkpoint_dir: str | os.PathLike, requested: dict[str, Any]
) -> None:
    saved = load_qat_metadata(checkpoint_dir)
    if saved is None:
        raise ValueError(
            f"QAT resume checkpoint {checkpoint_dir} has no {QAT_METADATA_NAME}"
        )
    if saved != requested:
        raise ValueError(
            f"QAT resume profile mismatch: checkpoint has {saved}, requested {requested}"
        )
