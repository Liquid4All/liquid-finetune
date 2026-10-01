from __future__ import annotations

import torch
import torch.nn.functional as F


def shift_loss_weights(loss_weights: torch.Tensor) -> torch.Tensor:
    """Shift weights in lockstep with next-token causal labels."""
    return F.pad(loss_weights, (0, 1), value=0.0)[..., 1:].contiguous()


def validate_loss_weights(
    loss_weights: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if loss_weights.shape != labels.shape:
        raise ValueError(
            "loss_weights must have the same shape as labels, got "
            f"{tuple(loss_weights.shape)} and {tuple(labels.shape)}"
        )
    if not torch.is_floating_point(loss_weights):
        raise TypeError("loss_weights must be a floating-point tensor")
    if not torch.isfinite(loss_weights).all():
        raise ValueError("loss_weights must contain only finite values")
    if torch.any(loss_weights < 0):
        raise ValueError("loss_weights must be nonnegative")


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    num_items_in_batch: torch.Tensor | float | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Weighted next-token CE normalized by effective supervised weight."""
    loss_weights = loss_weights.to(device=logits.device, dtype=torch.float32)
    labels = labels.to(logits.device)
    validate_loss_weights(loss_weights, labels)

    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    shift_weights = shift_loss_weights(loss_weights)
    valid = shift_labels.ne(ignore_index)
    effective_weights = shift_weights * valid

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(shift_labels)
    numerator = (token_loss * effective_weights).sum()

    if num_items_in_batch is None:
        denominator = effective_weights.sum()
    else:
        denominator = torch.as_tensor(
            num_items_in_batch,
            device=numerator.device,
            dtype=numerator.dtype,
        )
    return numerator / denominator.clamp_min(1.0)
