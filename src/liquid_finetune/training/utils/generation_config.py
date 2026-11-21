from __future__ import annotations

from collections.abc import Collection
from typing import Any


_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")


def apply_model_generation_defaults(
    args: Any, model: Any, configured_keys: Collection[str]
) -> None:
    """Use checkpoint sampling defaults unless the training YAML overrides them."""
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    for field in _SAMPLING_FIELDS:
        value = getattr(generation_config, field, None)
        if field not in configured_keys and value is not None:
            setattr(args, field, value)
