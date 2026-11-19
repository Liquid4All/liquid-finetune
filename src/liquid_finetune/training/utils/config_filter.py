from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)

BASE_RUNTIME_EXCLUDED_KEYS = {
    "training_type",
    "wandb_logging",
    "tracker",
    "trackio_space_id",
    "resume_from_checkpoint",
    "liquid_run_name_template",
}

MODEL_RUNTIME_EXCLUDED_KEYS = {
    "model_config",
    "chat_template",
    "chat_template_path",
    "adapter_path",
    "qat",
}

DISTRIBUTED_RUNTIME_EXCLUDED_KEYS = {
    "reshard_after_forward",
    "fsdp_cpu_offload",
}

MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS = {
    "checkpoint_staging_dir",
    "manual_sharded_checkpoint_format",
}

VLM_RUNTIME_EXCLUDED_KEYS = {
    "min_image_tokens",
    "max_image_tokens",
    "do_image_splitting",
    "group_by_image_tiles",
    "lr_multipliers",
    "vision_encoder_lr_multiplier",
    "freeze_vision_encoder",
}


def _translate_warmup_ratio(filtered: dict[str, Any], valid_keys: set[str]) -> None:
    """Map ``warmup_ratio`` onto ``warmup_steps`` when the config class dropped it.

    Transformers 5.x removed ``TrainingArguments(warmup_ratio=...)``; a float
    ``warmup_steps`` below 1 is now read as the warmup fraction. Without this
    the key would be filtered out below and every run would silently train
    without warmup.
    """
    if "warmup_ratio" not in filtered or "warmup_ratio" in valid_keys:
        return
    if "warmup_steps" not in valid_keys:
        return
    warmup_ratio = filtered.pop("warmup_ratio")
    if warmup_ratio is None:
        return
    if filtered.get("warmup_steps"):
        logger.warning(
            "Both warmup_ratio=%s and warmup_steps=%s are set; using warmup_steps",
            warmup_ratio,
            filtered["warmup_steps"],
        )
        return
    warmup_ratio = float(warmup_ratio)
    if not 0 <= warmup_ratio < 1:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
    filtered["warmup_steps"] = warmup_ratio


def filter_runtime_config_kwargs(
    config: dict[str, Any],
    *,
    excluded_keys: set[str],
    config_cls: type | None = None,
) -> tuple[dict[str, Any], list[str]]:
    filtered = {k: v for k, v in config.items() if k not in excluded_keys}
    if config_cls is None:
        return filtered, []

    valid_keys = set(inspect.signature(config_cls.__init__).parameters) - {"self"}
    _translate_warmup_ratio(filtered, valid_keys)
    dropped = sorted(k for k in filtered if k not in valid_keys)
    if dropped:
        logger.warning(
            "Dropping unrecognized %s config keys (possible typo): %s",
            config_cls.__name__,
            dropped,
        )
    return {k: v for k, v in filtered.items() if k in valid_keys}, dropped
