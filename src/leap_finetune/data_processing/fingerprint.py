from __future__ import annotations

import hashlib
import importlib.util
import pathlib
from typing import Any

from .config import PreprocessingOperation
from .importing import split_callable_reference


def callable_source_hash(reference: str) -> str | None:
    module_or_path, _ = split_callable_reference(reference)
    candidate = pathlib.Path(module_or_path)
    if not (candidate.suffix == ".py" or candidate.is_absolute()):
        spec = importlib.util.find_spec(module_or_path)
        if spec is None or spec.origin is None:
            return None
        candidate = pathlib.Path(spec.origin)
    if not candidate.is_file():
        return None
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def preprocessing_fingerprint_data(
    operations: list[PreprocessingOperation | dict],
) -> list[dict[str, Any]]:
    """Return canonical graph metadata for inclusion in dataset cache keys."""
    result = []
    for raw_operation in operations:
        operation = (
            raw_operation
            if isinstance(raw_operation, PreprocessingOperation)
            else PreprocessingOperation.model_validate(raw_operation)
        )
        value = operation.model_dump(mode="json")
        value["callable_source_sha256"] = callable_source_hash(operation.callable)
        value["fit_callable_source_sha256"] = (
            callable_source_hash(operation.fit_callable)
            if operation.fit_callable
            else None
        )
        result.append(value)
    return result
