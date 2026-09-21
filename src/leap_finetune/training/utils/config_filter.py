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
    "leap_run_name_template",
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
    "max_image_tokens",
    "do_image_splitting",
    "group_by_image_tiles",
    "lr_multipliers",
    "vision_encoder_lr_multiplier",
}


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
    dropped = sorted(k for k in filtered if k not in valid_keys)
    if dropped:
        logger.warning(
            "Dropping unrecognized %s config keys (possible typo): %s",
            config_cls.__name__,
            dropped,
        )
    return {k: v for k, v in filtered.items() if k in valid_keys}, dropped
