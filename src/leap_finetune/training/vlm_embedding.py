from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, ProcessorMixin, Trainer, TrainingArguments

from leap_finetune.checkpointing.callback import LeapCheckpointCallback
from leap_finetune.checkpointing.model_loading import load_vlm_model
from leap_finetune.data_loading.vlm_embedding import (
    build_candidate_mask,
    collate_raw_vlm_embedding_samples,
    create_vlm_embedding_collate_fn,
)
from leap_finetune.training.default_configs.vlm_sft_configs import (
    VLM_SFT_EXCLUDED_KEYS,
)
from leap_finetune.training.peft.peft import (
    apply_peft_to_model,
    load_peft_adapter,
    merge_and_save_peft_model,
)
from leap_finetune.training.utils.config_filter import filter_runtime_config_kwargs
from leap_finetune.training.utils.logging import finish_tracker, is_rank_zero
from leap_finetune.training.utils.trainer_lifecycle import run_training_safely
from leap_finetune.training.utils.trainer_mixins import RayDataLoaderMixin
from leap_finetune.training.utils.worker_setup import (
    init_tracking_from_config,
    resolve_train_eval_datasets,
)

logger = logging.getLogger(__name__)

_RUNTIME_KEYS = {
    "adapter_path",
    "cached_loss",
    "do_image_splitting",
    "gradient_checkpointing_scope",
    "leap_run_name_template",
    "loss",
    "max_image_tokens",
    "micro_batch_size",
    "min_image_tokens",
    "temperature",
}

_GRADIENT_CHECKPOINTING_COMPONENTS = {
    "language": "language_model",
    "vision": "vision_tower",
}


def pool_last_attended_token(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """L2-normalize the final attended state for left- or right-padded batches."""

    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError(
            "expected [batch, sequence, hidden] states and [batch, sequence] mask"
        )
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("hidden-state and attention-mask shapes do not match")
    if not torch.all(attention_mask.sum(dim=1) > 0):
        raise ValueError("every embedding sequence needs at least one attended token")
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    last_indices = (positions * attention_mask.to(dtype=positions.dtype)).argmax(dim=1)
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    pooled = hidden_states[batch_indices, last_indices]
    return F.normalize(pooled.float(), p=2, dim=-1)


def contrastive_embedding_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    temperature: float,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query-to-document InfoNCE over positives and explicit hard negatives."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if query.shape != positive.shape or query.shape != negative.shape:
        raise ValueError(
            "query, positive, and negative embeddings must have equal shapes"
        )
    candidates = torch.cat((positive, negative), dim=0)
    logits = query @ candidates.transpose(0, 1)
    if candidate_mask is not None:
        if candidate_mask.shape != logits.shape:
            raise ValueError(
                "candidate mask must have shape "
                f"{tuple(logits.shape)}, got {tuple(candidate_mask.shape)}"
            )
        candidate_mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
        labels = torch.arange(query.shape[0], device=query.device)
        if not torch.all(candidate_mask[labels, labels]):
            raise ValueError("candidate mask must retain every paired positive")
        logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
    labels = torch.arange(query.shape[0], device=query.device)
    return F.cross_entropy(logits / temperature, labels)


def _vlm_backbone(model):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base_model, "model", None)
    if backbone is None:
        raise TypeError(
            "VLM embedding training requires a model with a `.model` backbone"
        )
    return backbone


def configure_gradient_checkpointing(model: nn.Module, scope: str) -> bool:
    """Configure whole-model or component-selective activation checkpointing.

    Returns whether ``TrainingArguments.gradient_checkpointing`` should remain
    enabled. Selective modes are enabled directly on the requested backbone
    component so Trainer does not immediately re-enable checkpointing globally.
    """

    valid_scopes = {"all", "none", *_GRADIENT_CHECKPOINTING_COMPONENTS}
    if scope not in valid_scopes:
        choices = ", ".join(sorted(valid_scopes))
        raise ValueError(
            f"gradient_checkpointing_scope must be one of {choices}, got {scope!r}"
        )
    if scope == "all":
        return True
    if scope == "none":
        return False

    component_name = _GRADIENT_CHECKPOINTING_COMPONENTS[scope]
    component = getattr(_vlm_backbone(model), component_name, None)
    enable = getattr(component, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise TypeError(
            f"VLM backbone component {component_name!r} does not support "
            "gradient checkpointing"
        )
    enable()
    return False


class LFMVLMEmbeddingTrainer(RayDataLoaderMixin, Trainer):
    """Trainer that preserves the generative checkpoint and optimizes native states."""

    def __init__(self, *, temperature: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.embedding_temperature = temperature

    def _encode(self, model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = _vlm_backbone(model)(**batch, use_cache=False, return_dict=True)
        return pool_last_attended_token(
            outputs.last_hidden_state, batch["attention_mask"]
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        embeddings = {
            role: self._encode(
                model,
                cast(dict[str, torch.Tensor], inputs[role]),
            )
            for role in ("query", "positive", "negative")
        }
        loss = contrastive_embedding_loss(
            embeddings["query"],
            embeddings["positive"],
            embeddings["negative"],
            temperature=self.embedding_temperature,
            candidate_mask=cast(torch.Tensor | None, inputs.get("candidate_mask")),
        )
        return (loss, embeddings) if return_outputs else loss


@dataclass(frozen=True, slots=True)
class _RngState:
    cpu: torch.Tensor
    accelerator: torch.Tensor | None


def _capture_rng_state() -> _RngState:
    return _RngState(
        cpu=torch.get_rng_state(),
        accelerator=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )


def _restore_rng_state(state: _RngState) -> None:
    torch.set_rng_state(state.cpu)
    if state.accelerator is not None:
        torch.cuda.set_rng_state(state.accelerator)


class CachedLFMVLMEmbeddingTrainer(LFMVLMEmbeddingTrainer):
    """Exact logical-batch InfoNCE with memory-bounded gradient replay."""

    def __init__(
        self,
        *,
        processor: Any,
        micro_batch_size: int,
        **kwargs: Any,
    ) -> None:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        super().__init__(**kwargs)
        self.micro_batch_size = micro_batch_size
        self._micro_collator = create_vlm_embedding_collate_fn(processor)

    def _chunks(self, samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [
            samples[start : start + self.micro_batch_size]
            for start in range(0, len(samples), self.micro_batch_size)
        ]

    def _encode_chunk(
        self,
        model: nn.Module,
        samples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        batch = self._prepare_inputs(self._micro_collator(samples))
        return {
            role: self._encode(
                model,
                cast(dict[str, torch.Tensor], batch[role]),
            )
            for role in ("query", "positive", "negative")
        }

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Cache embeddings, differentiate the full loss, then replay chunks."""

        del num_items_in_batch
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        samples = cast(list[dict[str, Any]], inputs.get("samples"))
        if not samples:
            raise ValueError("cached VLM embedding training needs raw samples")

        chunks = self._chunks(samples)
        rng_states: list[_RngState] = []
        cached_by_role: dict[str, list[torch.Tensor]] = {
            "query": [],
            "positive": [],
            "negative": [],
        }
        with torch.no_grad():
            for chunk in chunks:
                rng_states.append(_capture_rng_state())
                with self.compute_loss_context_manager():
                    embeddings = self._encode_chunk(model, chunk)
                for role, value in embeddings.items():
                    cached_by_role[role].append(value.detach())

        cached = {
            role: torch.cat(values, dim=0).requires_grad_()
            for role, values in cached_by_role.items()
        }
        candidate_mask = build_candidate_mask(samples)
        loss = contrastive_embedding_loss(
            cached["query"],
            cached["positive"],
            cached["negative"],
            temperature=self.embedding_temperature,
            candidate_mask=candidate_mask,
        )
        loss.backward()
        embedding_gradients = {
            role: cast(torch.Tensor, value.grad).detach()
            for role, value in cached.items()
        }

        offset = 0
        for chunk, rng_state in zip(chunks, rng_states, strict=True):
            _restore_rng_state(rng_state)
            with self.compute_loss_context_manager():
                embeddings = self._encode_chunk(model, chunk)
                end = offset + len(chunk)
                surrogate = sum(
                    (
                        embeddings[role]
                        * embedding_gradients[role][offset:end].to(
                            embeddings[role].device
                        )
                    ).sum()
                    for role in ("query", "positive", "negative")
                )
            self.accelerator.backward(surrogate)
            offset = end

        return loss.detach()


def vlm_embedding_run(
    training_config: dict,
    train_dataset=None,
    eval_dataset=None,
) -> None:
    """Run native-1024 assistant-prefix VLM embedding adaptation."""

    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )
    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")
    train_config = training_config.get("train_config", {})
    output_dir = train_config.get("output_dir", "")
    resume_from = train_config.get("resume_from_checkpoint")
    adapter_path = train_config.get("adapter_path")
    temperature = float(train_config.get("temperature", 0.02))
    cached_loss = bool(train_config.get("cached_loss", False))
    micro_batch_size = int(train_config.get("micro_batch_size", 32))
    checkpointing_scope = train_config.get("gradient_checkpointing_scope", "auto")

    filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=VLM_SFT_EXCLUDED_KEYS | _RUNTIME_KEYS,
        config_cls=TrainingArguments,
    )
    if not filtered.get("dataloader_num_workers", 0):
        filtered.pop("dataloader_prefetch_factor", None)
    if checkpointing_scope == "auto":
        checkpointing_scope = (
            "all" if filtered.get("gradient_checkpointing", False) else "none"
        )
    filtered["gradient_checkpointing"] = checkpointing_scope == "all"
    filtered.setdefault("remove_unused_columns", False)
    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir or None,
        resume_from_checkpoint=resume_from,
    )
    args = TrainingArguments(
        report_to=tracker,
        run_name=job_name,
        **filtered,
    )
    model, processor = load_vlm_model(
        model_name,
        min_image_tokens=train_config.get("min_image_tokens"),
        max_image_tokens=train_config.get("max_image_tokens"),
        do_image_splitting=train_config.get("do_image_splitting", True),
    )
    processor_any = cast(Any, processor)
    processor_any.tokenizer.padding_side = "left"
    trainable_model = cast(PreTrainedModel, model)
    if adapter_path:
        trainable_model = load_peft_adapter(trainable_model, adapter_path)
    elif peft_config:
        trainable_model = apply_peft_to_model(trainable_model, peft_config)
    configure_gradient_checkpointing(trainable_model, checkpointing_scope)

    trainer_class = (
        CachedLFMVLMEmbeddingTrainer if cached_loss else LFMVLMEmbeddingTrainer
    )
    trainer_kwargs: dict[str, Any] = {}
    if cached_loss:
        if args.gradient_accumulation_steps != 1:
            raise ValueError("cached loss requires gradient_accumulation_steps=1")
        trainer_kwargs.update(
            processor=processor,
            micro_batch_size=micro_batch_size,
            data_collator=collate_raw_vlm_embedding_samples,
        )
    else:
        trainer_kwargs["data_collator"] = create_vlm_embedding_collate_fn(processor)
    trainer = trainer_class(
        temperature=temperature,
        model=trainable_model,
        processing_class=processor,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **trainer_kwargs,
    )
    trainer.add_callback(
        LeapCheckpointCallback(
            run_name_template=train_config.get("leap_run_name_template")
        )
    )
    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    if (peft_config or adapter_path) and is_rank_zero():
        merge_and_save_peft_model(
            trainable_model,
            cast(ProcessorMixin, processor),
            str(args.output_dir),
            train_config.get("leap_run_name_template"),
        )
    finish_tracker(tracker)


__all__ = [
    "LFMVLMEmbeddingTrainer",
    "CachedLFMVLMEmbeddingTrainer",
    "configure_gradient_checkpointing",
    "contrastive_embedding_loss",
    "pool_last_attended_token",
    "vlm_embedding_run",
]
