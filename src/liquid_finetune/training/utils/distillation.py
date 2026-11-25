"""Shared helpers for teacher-student knowledge distillation.

The divergences follow TRL's generalized Jensen-Shannon divergence (GKD,
https://huggingface.co/papers/2306.13649): ``beta=0`` is the forward
KL(teacher || student), ``beta=1`` is the reverse KL(student || teacher), and
``0 < beta < 1`` is ``beta * KL(teacher || M) + (1 - beta) * KL(student || M)``
with the mixture ``M = beta * teacher + (1 - beta) * student``.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Mapping
from typing import Any

import torch

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Float32 working-set size, in elements, for each vocab-wide chunk (256 MiB).
_CHUNK_ELEMENTS = 1 << 26
_LOG_HALF = -math.log(2.0)
_FLOAT32_EPS = torch.finfo(torch.float32).eps


# === Top-k generalized JSD ===


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Reduce low-precision logits in float32; keep float64 inputs exact."""
    return torch.promote_types(dtype, torch.float32)


def _rows_per_chunk(vocab_size: int) -> int:
    return max(1, _CHUNK_ELEMENTS // max(vocab_size, 1))


def _chunks(num_rows: int, vocab_size: int):
    step = _rows_per_chunk(vocab_size)
    for start in range(0, num_rows, step):
        yield slice(start, min(start + step, num_rows))


def _log_partition(
    logits: torch.Tensor,
    batch_index: torch.Tensor,
    position_index: torch.Tensor,
    vocab_size: int,
    temperature: float,
) -> torch.Tensor:
    """``logsumexp(logits / temperature)`` (at least float32) for selected rows."""
    dtype = _compute_dtype(logits.dtype)
    lse = torch.empty(batch_index.numel(), dtype=dtype, device=logits.device)
    for rows in _chunks(batch_index.numel(), vocab_size):
        chunk = logits[batch_index[rows], position_index[rows], :vocab_size].to(dtype)
        lse[rows] = torch.logsumexp(chunk / temperature, dim=-1)
    return lse


def _gather_support(
    logits: torch.Tensor,
    batch_index: torch.Tensor,
    position_index: torch.Tensor,
    support: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    values = logits[batch_index.unsqueeze(-1), position_index.unsqueeze(-1), support]
    return values.to(_compute_dtype(logits.dtype)) / temperature


class _SupportLogProbs(torch.autograd.Function):
    """``log_softmax(logits / T)`` evaluated only on a per-position support.

    The log-partition is reduced in float32 one chunk of rows at a time and the
    backward pass recomputes each chunk's softmax, so neither direction
    materializes a vocab-sized float32 copy of the logits. The only full-size
    allocation is the logits gradient itself.
    """

    @staticmethod
    def forward(
        ctx, logits, batch_index, position_index, support, vocab_size, temperature
    ):
        lse = _log_partition(
            logits, batch_index, position_index, vocab_size, temperature
        )
        support_log_probs = _gather_support(
            logits, batch_index, position_index, support, temperature
        ) - lse.unsqueeze(-1)
        ctx.save_for_backward(logits, batch_index, position_index, support, lse)
        ctx.vocab_size = vocab_size
        ctx.temperature = temperature
        return support_log_probs

    @staticmethod
    def backward(ctx, grad_output):
        logits, batch_index, position_index, support, lse = ctx.saved_tensors
        vocab_size, temperature = ctx.vocab_size, ctx.temperature
        grad_logits = torch.zeros(
            logits.shape, dtype=logits.dtype, device=logits.device
        )
        dtype = _compute_dtype(logits.dtype)
        grad_output = grad_output.to(dtype)
        for rows in _chunks(batch_index.numel(), vocab_size):
            chunk_batch, chunk_position = batch_index[rows], position_index[rows]
            chunk_grad = grad_output[rows]
            logits_chunk = logits[chunk_batch, chunk_position, :vocab_size].to(dtype)
            # d log_softmax(x / T)_s / dx_j = (1[j == s] - softmax(x / T)_j) / T
            grad_chunk = torch.exp(logits_chunk / temperature - lse[rows, None])
            grad_chunk.mul_(-chunk_grad.sum(dim=-1, keepdim=True))
            grad_chunk.scatter_add_(-1, support[rows], chunk_grad)
            grad_chunk.div_(temperature)
            grad_logits[chunk_batch, chunk_position, :vocab_size] = grad_chunk.to(
                grad_logits.dtype
            )
        return grad_logits, None, None, None, None, None


def _topk_indices(
    logits: torch.Tensor,
    batch_index: torch.Tensor,
    position_index: torch.Tensor,
    vocab_size: int,
    top_k: int,
) -> torch.Tensor:
    indices = torch.empty(
        (batch_index.numel(), top_k), dtype=torch.long, device=logits.device
    )
    for rows in _chunks(batch_index.numel(), vocab_size):
        chunk = logits[batch_index[rows], position_index[rows], :vocab_size]
        indices[rows] = chunk.topk(top_k, dim=-1).indices
    return indices


def _select_support(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    batch_index: torch.Tensor,
    position_index: torch.Tensor,
    *,
    beta: float,
    top_k: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(support, valid)`` token ids for each selected position.

    Forward KL keeps the teacher's top-k, reverse KL the student's top-k, and the
    mixed JSD the union of both (duplicates are marked invalid).
    """
    num_rows = batch_index.numel()
    device = student_logits.device
    if top_k >= vocab_size:
        support = torch.arange(vocab_size, device=device).expand(num_rows, -1)
        return support, torch.ones(support.shape, dtype=torch.bool, device=device)

    with torch.no_grad():
        selections = []
        if beta < 1:
            selections.append(
                _topk_indices(
                    teacher_logits, batch_index, position_index, vocab_size, top_k
                )
            )
        if beta > 0:
            selections.append(
                _topk_indices(
                    student_logits.detach(),
                    batch_index,
                    position_index,
                    vocab_size,
                    top_k,
                )
            )
        if len(selections) == 1:
            support = selections[0]
            return support, torch.ones(support.shape, dtype=torch.bool, device=device)

        support = torch.cat(selections, dim=-1).sort(dim=-1).values
        valid = torch.ones(support.shape, dtype=torch.bool, device=device)
        valid[:, 1:] = support[:, 1:] != support[:, :-1]
        return support, valid


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(x))`` for ``x < 0``."""
    return torch.where(
        x > _LOG_HALF,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def _valid_logsumexp(log_probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = torch.where(valid, log_probs, torch.full_like(log_probs, float("-inf")))
    return torch.logsumexp(masked, dim=-1, keepdim=True)


def _restrict_to_support(
    log_probs: torch.Tensor, valid: torch.Tensor, add_tail: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn support log-probs into a normalized distribution.

    With ``add_tail`` the remaining probability mass becomes one extra bucket, so
    the student is still penalized for mass it places outside the support.
    Without it, both distributions are renormalized over the support.
    """
    support_mass = _valid_logsumexp(log_probs, valid)
    if not add_tail:
        return log_probs - support_mass, valid
    tail = _log1mexp(support_mass.clamp(max=-_FLOAT32_EPS))
    tail_valid = torch.ones_like(valid[:, :1])
    return torch.cat([log_probs, tail], dim=-1), torch.cat([valid, tail_valid], dim=-1)


def _masked_kl(
    p_log_probs: torch.Tensor, q_log_probs: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Per-row ``sum_i p_i (log p_i - log q_i)`` over valid entries."""
    zeros = torch.zeros_like(p_log_probs)
    p_log_probs = torch.where(valid, p_log_probs, zeros)
    q_log_probs = torch.where(valid, q_log_probs, zeros)
    terms = torch.exp(p_log_probs) * (p_log_probs - q_log_probs)
    return torch.where(valid, terms, zeros).sum(dim=-1)


def _generalized_jsd(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if beta == 0:
        return _masked_kl(teacher_log_probs, student_log_probs, valid)
    if beta == 1:
        return _masked_kl(student_log_probs, teacher_log_probs, valid)
    mixture_log_probs = torch.logaddexp(
        student_log_probs + math.log1p(-beta),
        teacher_log_probs + math.log(beta),
    )
    return beta * _masked_kl(teacher_log_probs, mixture_log_probs, valid) + (
        1 - beta
    ) * _masked_kl(student_log_probs, mixture_log_probs, valid)


def topk_generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    beta: float = 0.5,
    top_k: int,
    add_tail: bool = True,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    num_items_in_batch: torch.Tensor | int | None = None,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Generalized JSD restricted to a per-position top-k support.

    Arguments mirror ``trl.experimental.gkd.GKDTrainer.generalized_jsd_loss``:
    logits are ``(batch, seq, vocab)`` and already aligned with ``labels``
    (``-100`` marks positions without a target). At each target position the
    divergence is evaluated on the teacher's top-k tokens for forward KL
    (``beta == 0``), the student's top-k for reverse KL (``beta == 1``), or the
    union of both for the mixed JSD. ``add_tail`` appends one bucket holding the
    probability mass outside the support; otherwise both distributions are
    renormalized over the support. With ``top_k >= vocab`` and ``add_tail=False``
    this equals the full-vocabulary loss.

    Only the target positions are processed, in bounded float32 chunks, so the
    extra memory is ``O(targets * top_k)`` instead of several
    ``(batch, seq, vocab)`` tensors. ``vocab_size`` restricts both models to
    their first ``vocab_size`` logits (e.g. when only embedding padding
    differs). ``reduction`` supports ``"batchmean"``, ``"sum"``, and ``"none"``
    (one value per target position); ``num_items_in_batch`` overrides the
    denominator exactly as in TRL.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if reduction not in {"batchmean", "sum", "none"}:
        raise ValueError(
            "reduction must be 'batchmean', 'sum', or 'none' for the top-k loss, "
            f"got {reduction!r}"
        )
    if (
        student_logits.dim() != 3
        or teacher_logits.shape[:2] != student_logits.shape[:2]
    ):
        raise ValueError(
            "student_logits and teacher_logits must be (batch, seq, vocab) tensors "
            "with matching batch and sequence dimensions"
        )

    width = min(student_logits.size(-1), teacher_logits.size(-1))
    if vocab_size is not None:
        width = min(width, vocab_size)
    top_k = min(top_k, width)

    if labels is None:
        batch_size, seq_len = student_logits.shape[:2]
        positions = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=student_logits.device
        )
    else:
        positions = labels.to(student_logits.device) != IGNORE_INDEX
    batch_index, position_index = positions.nonzero(as_tuple=True)

    support, valid = _select_support(
        student_logits,
        teacher_logits,
        batch_index,
        position_index,
        beta=beta,
        top_k=top_k,
        vocab_size=width,
    )
    student_log_probs = _SupportLogProbs.apply(
        student_logits, batch_index, position_index, support, width, temperature
    )
    with torch.no_grad():
        teacher_lse = _log_partition(
            teacher_logits, batch_index, position_index, width, temperature
        )
        teacher_log_probs = _gather_support(
            teacher_logits, batch_index, position_index, support, temperature
        ) - teacher_lse.unsqueeze(-1)

    student_log_probs, distribution_valid = _restrict_to_support(
        student_log_probs, valid, add_tail
    )
    teacher_log_probs, _ = _restrict_to_support(teacher_log_probs, valid, add_tail)
    divergence = _generalized_jsd(
        student_log_probs, teacher_log_probs, distribution_valid, beta
    )

    if num_items_in_batch is not None:
        total = divergence.sum()
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(total.device)
        return total / num_items_in_batch
    if reduction == "batchmean":
        denominator = (
            max(divergence.numel(), 1)
            if labels is not None
            else max(student_logits.size(0), 1)
        )
        return divergence.sum() / denominator
    if reduction == "sum":
        return divergence.sum()
    return divergence


# === Teacher handling ===


def text_vocab_size(model) -> int:
    """Return the output vocabulary size declared by a (possibly multimodal) model."""
    return int(model.config.get_text_config().vocab_size)


def resolve_shared_vocab_size(
    student_tokenizer,
    teacher_tokenizer,
    *,
    student_vocab_size: int,
    teacher_vocab_size: int,
) -> int:
    """Validate that student and teacher share a vocabulary.

    The tokenizers must map every token to the same id. The models' embedding
    sizes may differ by padding as long as both cover the tokenizer; the
    returned width is the logit range both models share.
    """
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if student_vocab != teacher_vocab:
        mismatched = sorted(
            token
            for token in student_vocab.keys() | teacher_vocab.keys()
            if student_vocab.get(token) != teacher_vocab.get(token)
        )
        examples = ", ".join(
            f"{token!r}: student={student_vocab.get(token)} "
            f"teacher={teacher_vocab.get(token)}"
            for token in mismatched[:5]
        )
        raise ValueError(
            "Distillation requires the student and teacher to share a tokenizer: "
            f"{len(mismatched)} token(s) map to different ids "
            f"(student has {len(student_vocab)} tokens, teacher has "
            f"{len(teacher_vocab)}; e.g. {examples}). Pick a teacher that uses "
            "the student's tokenizer."
        )

    tokenizer_size = len(student_tokenizer)
    for role, size in (
        ("student", student_vocab_size),
        ("teacher", teacher_vocab_size),
    ):
        if size < tokenizer_size:
            raise ValueError(
                f"The {role} model has vocab_size {size}, smaller than the shared "
                f"tokenizer ({tokenizer_size} tokens)."
            )

    shared_size = min(student_vocab_size, teacher_vocab_size)
    if student_vocab_size != teacher_vocab_size:
        logger.info(
            "Student and teacher embedding sizes differ only by padding "
            "(%d vs %d); distilling over the first %d logits",
            student_vocab_size,
            teacher_vocab_size,
            shared_size,
        )
    return shared_size


def freeze_teacher_model(model):
    """Put the teacher in eval mode and stop gradient tracking."""
    model.eval()
    model.requires_grad_(False)
    return model


def load_teacher_model(train_config: Mapping[str, Any]):
    """Load the frozen teacher with the same conventions as the student."""
    from liquid_finetune.checkpointing.model_loading import load_model

    teacher_name = train_config.get("teacher_model_name_or_path")
    if not teacher_name:
        raise ValueError("training_config.teacher_model_name_or_path is required")
    teacher_model, teacher_tokenizer = load_model(
        teacher_name,
        model_config=train_config.get("teacher_model_config"),
        revision=train_config.get("teacher_model_revision"),
        install_memory_efficient_loss=False,
    )
    return freeze_teacher_model(teacher_model), teacher_tokenizer


@contextlib.contextmanager
def report_vocab_size(model, vocab_size: int):
    """Temporarily report ``vocab_size`` on ``model``'s text config.

    TRL's distillation trainers compare ``config.vocab_size`` for equality.
    Once the tokenizers have been validated, embedding sizes that differ only
    by padding are safe, and the loss restricts both models to the shared width.
    """
    text_config = model.config.get_text_config()
    original = text_config.vocab_size
    text_config.vocab_size = vocab_size
    try:
        yield
    finally:
        text_config.vocab_size = original


def prepare_fsdp_teacher(accelerator, teacher):
    """Shard a teacher that TRL prepared with ``prepare_model(evaluation_mode=True)``.

    Under FSDP that call neither shards nor moves the model. Undo its side
    effects and wrap the teacher with TRL's ``prepare_fsdp``, the same way
    ``DPOTrainer`` prepares its reference model.
    """
    from trl.models import prepare_fsdp

    accelerator._models[:] = [
        model for model in accelerator._models if model is not teacher
    ]
    original_forward = teacher.__dict__.pop("_original_forward", None)
    if original_forward is not None:
        teacher.forward = original_forward
    return prepare_fsdp(teacher, accelerator)


# === Prompts for student generation ===


def conversation_prompt(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the conversation a student should continue for one dataset row.

    A ``prompt`` column is used as-is, with a plain string becoming a single
    user turn. Otherwise ``messages`` is used without its final assistant turn.
    """
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        if prompt.strip():
            return [{"role": "user", "content": prompt}]
    elif prompt is not None and len(prompt) > 0:
        return [dict(message) for message in prompt]

    messages = row.get("messages")
    if messages is not None and len(messages) > 0:
        messages = [dict(message) for message in messages]
        if messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        if messages:
            return messages
    raise ValueError(
        "Each row needs a non-empty `prompt` (string or message list) or a "
        "`messages` conversation with turns before the final assistant reply"
    )
