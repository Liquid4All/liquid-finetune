from __future__ import annotations

import logging
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)


# === Freezing ===


def _matches_param_prefix(name: str, prefix: str) -> bool:
    return (
        name == prefix
        or name.startswith(f"{prefix}.")
        or f".{prefix}." in name
        or name.endswith(f".{prefix}")
    )


def freeze_vlm_modules(model: Any, prefixes: Iterable[str]) -> dict[str, int]:
    """Disable gradients for parameters whose names match configured prefixes."""
    frozen_counts: dict[str, int] = {prefix: 0 for prefix in prefixes}
    frozen_params = 0

    for name, param in model.named_parameters():
        for prefix in prefixes:
            if _matches_param_prefix(name, prefix):
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params += param.numel()
                frozen_counts[prefix] += param.numel()
                break

    for prefix, count in frozen_counts.items():
        if count == 0:
            raise ValueError(f"No parameters matched freeze prefix {prefix!r}")
        logger.info("Frozen VLM module '%s': %d params", prefix, count)
    logger.info("Frozen VLM trainable params disabled: %d", frozen_params)
    return frozen_counts


# === Parameter Groups ===


def build_vlm_param_groups(
    model: Any,
    lr_multipliers: dict[str, float],
    base_lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build optimizer groups with per-component learning rates."""
    grouped: dict[str, list] = {prefix: [] for prefix in lr_multipliers}
    ungrouped: list = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for prefix in lr_multipliers:
            if _matches_param_prefix(name, prefix):
                grouped[prefix].append(param)
                matched = True
                break
        if not matched:
            ungrouped.append(param)

    optimizer_groups: list[dict[str, Any]] = []
    group_names: list[str] = []

    for prefix, params in grouped.items():
        if not params:
            continue
        mult = lr_multipliers[prefix]
        optimizer_groups.append(
            {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay}
        )
        short_name = prefix.removeprefix("model.")
        group_names.append(short_name)
        logger.info(
            "Param group '%s': %d params, lr=%.2e",
            prefix,
            len(params),
            base_lr * mult,
        )

    if ungrouped:
        optimizer_groups.append(
            {"params": ungrouped, "lr": base_lr, "weight_decay": weight_decay}
        )
        group_names.append("ungrouped")
        logger.info(
            "Param group 'ungrouped': %d params, lr=%.2e", len(ungrouped), base_lr
        )

    return optimizer_groups, group_names


# === Optimizer Creation ===


def create_vlm_optimizer(
    optimizer_groups: list[dict[str, Any]],
    optimizer_type: str,
    betas: tuple[float, float],
) -> Any:
    """Create the configured optimizer for VLM trainers."""
    normalized = optimizer_type.lower().replace("-", "_")
    if normalized in {"adamw", "torch_adamw", "fused_adamw"}:
        return torch.optim.AdamW(
            optimizer_groups, betas=betas, fused=torch.cuda.is_available()
        )

    if normalized in {"adamw_8bit", "bitsandbytes_adamw_8bit", "bnb_adamw_8bit"}:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "optimizer_type='adamw_8bit' requires bitsandbytes to be installed "
                "in the training environment"
            ) from exc
        return bnb.optim.AdamW8bit(optimizer_groups, betas=betas)

    if normalized in {"adam_fp8", "adamw_fp8"}:
        raise ValueError(
            "FP8 Adam is not supported by this trainer. Use optimizer_type='adamw_8bit' "
            "for bitsandbytes 8-bit optimizer states, or leave optimizer_type='adamw'."
        )

    raise ValueError(
        f"Unsupported VLM optimizer_type={optimizer_type!r}; expected one of "
        "'adamw' or 'adamw_8bit'"
    )


# === Logging ===


def log_per_group_lrs(
    optimizer: Any,
    group_names: Iterable[str],
    logs: dict[str, float],
) -> None:
    """Inject per-component learning rates into a ``Trainer.log()`` payload."""
    if optimizer is None or "learning_rate" not in logs:
        return
    for name, group in zip(group_names, optimizer.param_groups):
        logs[f"lr/{name}"] = group["lr"]
