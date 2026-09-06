import logging
import os
from functools import partial
from types import MethodType

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


DEFAULT_CAUSAL_LM_LOSS_BACKEND = "torch_compile"
DEFAULT_CAUSAL_LM_LOSS_CHUNK_SIZE = 128


def _get_loss_backend() -> str:
    backend = os.getenv(
        "LIQUID_CAUSAL_LM_LOSS_BACKEND", DEFAULT_CAUSAL_LM_LOSS_BACKEND
    ).lower()
    if backend not in {"torch_compile", "eager"}:
        raise ValueError(
            "LIQUID_CAUSAL_LM_LOSS_BACKEND must be one of "
            f"{{'torch_compile', 'eager'}}, got {backend!r}"
        )
    if backend == "torch_compile" and not hasattr(torch, "compile"):
        logger.warning("torch.compile is unavailable, falling back to eager LM loss")
        return "eager"
    return backend


_LOSS_BACKEND = _get_loss_backend()


def _shift_causal_labels(
    labels: torch.Tensor,
    ignore_index: int,
    shift_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    if shift_labels is not None:
        return shift_labels
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    return labels[..., 1:].contiguous()


def _move_num_items_to_device(
    num_items_in_batch: torch.Tensor | None,
    loss: torch.Tensor,
) -> torch.Tensor | None:
    if num_items_in_batch is None or not torch.is_tensor(num_items_in_batch):
        return num_items_in_batch
    return num_items_in_batch.to(loss.device)


def _linear_cross_entropy_sum_eager(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = hidden_states @ lm_head_weight.T
    if lm_head_bias is not None:
        logits = logits + lm_head_bias
    return nn.functional.cross_entropy(logits.float(), labels, reduction="sum")


if hasattr(torch, "compile"):

    @torch.compile(fullgraph=True, dynamic=True)
    def _linear_cross_entropy_sum_compiled(
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = hidden_states @ lm_head_weight.T
        if lm_head_bias is not None:
            logits = logits + lm_head_bias
        return nn.functional.cross_entropy(logits.float(), labels, reduction="sum")


else:
    _linear_cross_entropy_sum_compiled = None


def _linear_cross_entropy_sum(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    labels: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    if backend == "torch_compile" and _linear_cross_entropy_sum_compiled is not None:
        return _linear_cross_entropy_sum_compiled(
            hidden_states, lm_head_weight, lm_head_bias, labels
        )
    return _linear_cross_entropy_sum_eager(
        hidden_states, lm_head_weight, lm_head_bias, labels
    )


def chunked_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    chunk_size: int = DEFAULT_CAUSAL_LM_LOSS_CHUNK_SIZE,
    **kwargs,
) -> torch.Tensor:
    """Fallback path when logits are already materialized by the model forward."""
    shift_labels = _shift_causal_labels(labels, ignore_index, shift_labels)

    flat_logits = logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1).to(flat_logits.device)

    total_loss = flat_logits.new_zeros(())
    total_items = flat_logits.new_zeros((), dtype=torch.long)

    for start in range(0, flat_logits.shape[0], chunk_size):
        end = min(start + chunk_size, flat_logits.shape[0])
        labels_chunk = flat_labels[start:end]
        valid = labels_chunk != ignore_index
        if not torch.any(valid):
            continue

        total_loss = total_loss + nn.functional.cross_entropy(
            flat_logits[start:end][valid].float(),
            labels_chunk[valid],
            reduction="sum",
        )
        total_items = total_items + valid.sum()

    num_items_in_batch = _move_num_items_to_device(num_items_in_batch, total_loss)
    if num_items_in_batch is not None:
        return total_loss / num_items_in_batch
    return total_loss / total_items.clamp_min(1)


def linear_chunked_causal_lm_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None = None,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    chunk_size: int = DEFAULT_CAUSAL_LM_LOSS_CHUNK_SIZE,
    backend: str = _LOSS_BACKEND,
) -> torch.Tensor:
    """Compute causal LM loss from hidden states without materializing full logits."""
    shift_labels = _shift_causal_labels(labels, ignore_index, shift_labels)

    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_labels = shift_labels.reshape(-1).to(flat_hidden.device)

    total_loss = flat_hidden.new_zeros(())
    total_items = flat_hidden.new_zeros((), dtype=torch.long)

    for start in range(0, flat_hidden.shape[0], chunk_size):
        end = min(start + chunk_size, flat_hidden.shape[0])
        labels_chunk = flat_labels[start:end]
        valid = labels_chunk != ignore_index
        if not torch.any(valid):
            continue

        total_loss = total_loss + _linear_cross_entropy_sum(
            flat_hidden[start:end][valid],
            lm_head_weight,
            lm_head_bias,
            labels_chunk[valid],
            backend=backend,
        )
        total_items = total_items + valid.sum()

    num_items_in_batch = _move_num_items_to_device(num_items_in_batch, total_loss)
    if num_items_in_batch is not None:
        return total_loss / num_items_in_batch
    return total_loss / total_items.clamp_min(1)


def install_memory_efficient_causal_lm_loss(model: nn.Module) -> None:
    """Patch training-time forward to compute LM loss from hidden states."""
    chunk_size = int(
        os.getenv("LIQUID_CAUSAL_LM_LOSS_CHUNK_SIZE", DEFAULT_CAUSAL_LM_LOSS_CHUNK_SIZE)
    )

    model.loss_function = partial(chunked_causal_lm_loss, chunk_size=chunk_size)

    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        logger.info(
            "Installed fallback chunked LM loss: backend=%s chunk_size=%s",
            _LOSS_BACKEND,
            chunk_size,
        )
        return

    original_forward = model.forward

    def memory_efficient_forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        logits_to_keep = kwargs.get("logits_to_keep", 0)

        if (
            labels is None
            or not self.training
            or args
            or logits_to_keep not in (0, None)
        ):
            # Keep ``num_items_in_batch`` in kwargs: the stock forward hands it
            # to ``model.loss_function`` so the loss stays normalized by the
            # global token count gathered by the trainer. Dropping it here made
            # eval fall back to a local per-token mean, which the trainer then
            # scaled by the process count, inflating eval_loss by the world
            # size on multi-GPU runs.
            return original_forward(*args, **kwargs)

        num_items_in_batch = kwargs.pop("num_items_in_batch", None)

        model_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"labels", "logits_to_keep"}
        }
        outputs = self.model(**model_kwargs)
        hidden_states = outputs.last_hidden_state
        lm_head = self.get_output_embeddings()

        loss = linear_chunked_causal_lm_loss(
            hidden_states=hidden_states,
            labels=labels,
            lm_head_weight=lm_head.weight,
            lm_head_bias=getattr(lm_head, "bias", None),
            num_items_in_batch=num_items_in_batch,
            chunk_size=chunk_size,
            backend=_LOSS_BACKEND,
        )

        return {
            "loss": loss,
            "logits": hidden_states.new_empty((0,)),
            "past_key_values": getattr(outputs, "past_key_values", None),
            "hidden_states": getattr(outputs, "hidden_states", None),
            "attentions": getattr(outputs, "attentions", None),
            "router_logits": getattr(outputs, "router_logits", None),
        }

    model.forward = MethodType(memory_efficient_forward, model)
    logger.info(
        "Installed memory-efficient LM loss: backend=%s chunk_size=%s",
        _LOSS_BACKEND,
        chunk_size,
    )
