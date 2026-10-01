from __future__ import annotations

import copy

from leap_finetune.data_processing.fingerprint import callable_source_hash


def loss_weighting_fingerprint_data(config: dict | None) -> dict | None:
    """Include custom selector source in tokenization cache identity."""
    if not config:
        return config
    value = copy.deepcopy(config)
    for rule in value.get("rules", []):
        selector = rule.get("selector", {})
        reference = selector.get("callable")
        if selector.get("type") == "callable" and isinstance(reference, str):
            selector["callable_source_sha256"] = callable_source_hash(reference)
    return value
