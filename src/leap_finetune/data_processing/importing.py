from __future__ import annotations

import hashlib
import importlib
import importlib.util
import pathlib
import sys
from collections.abc import Callable


def split_callable_reference(reference: str) -> tuple[str, str]:
    try:
        module_or_path, attribute = reference.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid callable reference {reference!r}; expected module:function"
        ) from exc
    if not module_or_path or not attribute:
        raise ValueError(
            f"Invalid callable reference {reference!r}; expected module:function"
        )
    return module_or_path, attribute


def resolve_callable_reference(reference: str, base_dir: pathlib.Path) -> str:
    """Resolve relative Python-file references against the job config directory."""
    module_or_path, attribute = split_callable_reference(reference)
    candidate = pathlib.Path(module_or_path).expanduser()
    is_file_reference = candidate.suffix == ".py" or "/" in module_or_path
    if not is_file_reference:
        return reference
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return f"{candidate.resolve()}:{attribute}"


def load_callable(reference: str) -> Callable:
    """Import a callable without evaluating user-provided Python expressions."""
    module_or_path, attribute = split_callable_reference(reference)
    candidate = pathlib.Path(module_or_path)
    if candidate.suffix == ".py" or candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"Callable Python file not found: {candidate}")
        digest = hashlib.sha256(
            str(candidate.resolve()).encode() + candidate.read_bytes()
        ).hexdigest()[:16]
        module_name = f"leap_dataset_recipe_{digest}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import callable file: {candidate}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            source = importlib.util.decode_source(candidate.read_bytes())
            exec(compile(source, str(candidate), "exec"), module.__dict__)
    else:
        module = importlib.import_module(module_or_path)

    value = module
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"Configured object is not callable: {reference}")
    return value
