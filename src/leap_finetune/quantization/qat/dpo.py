from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import torch.nn as nn

from leap_finetune.quantization.qat.controller import prepare_model_for_qat

ModelT = TypeVar("ModelT", bound=nn.Module)


def prepare_dpo_reference_model(
    train_config: Mapping[str, Any],
    *,
    policy_uses_peft: bool,
    load_model: Callable[[], ModelT],
    is_vlm: bool = False,
) -> ModelT | None:
    """Create the explicit DPO reference only when TRL cannot reuse the policy."""
    qat_config = train_config.get("qat")
    if qat_config is None:
        return None
    quantize_reference = (
        qat_config.get("quantize_reference", True)
        if isinstance(qat_config, Mapping)
        else qat_config.quantize_reference
    )
    if policy_uses_peft and quantize_reference:
        # TRL computes reference log-probabilities with the policy adapters disabled.
        # The shared base model already contains the same fake-quantization hooks.
        return None

    reference = load_model()
    if quantize_reference:
        prepare_model_for_qat(reference, dict(train_config), is_vlm=is_vlm)
    return reference
