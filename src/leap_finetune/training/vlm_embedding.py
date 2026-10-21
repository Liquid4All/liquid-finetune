from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from transformers import PreTrainedModel, ProcessorMixin, Trainer, TrainingArguments

from leap_finetune.checkpointing.callback import LeapCheckpointCallback
from leap_finetune.checkpointing.model_loading import load_vlm_model
from leap_finetune.data_loading.vlm_embedding import (
    OrderedChunkCollator,
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
    "collate_workers",
    "do_image_splitting",
    "global_batch_size",
    "gradient_checkpointing_scope",
    "leap_run_name_template",
    "loss",
    "max_image_tokens",
    "micro_batch_size",
    "min_image_tokens",
    "retain_graphs",
    "sort_rows",
    "temperature",
}

_EMBEDDING_ROLES = ("query", "positive", "negative")
_MASK_FIELDS = (
    "loss_kind",
    "positive_key",
    "negative_key",
    "candidate_set_id",
    "candidate_group",
    "loss_group",
    "modality",
)

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


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _unwrap_data_parallel(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


def gather_rank_rows(local: torch.Tensor) -> tuple[torch.Tensor, int]:
    """All-gather variable-length row blocks in rank order.

    Returns the concatenated rows and this rank's offset into them.
    """

    world = dist.get_world_size()
    count = torch.tensor([local.shape[0]], device=local.device, dtype=torch.int64)
    counts = [torch.zeros_like(count) for _ in range(world)]
    dist.all_gather(counts, count)
    sizes = [int(value.item()) for value in counts]
    padded = local.new_zeros((max(sizes), *local.shape[1:]))
    padded[: local.shape[0]] = local
    blocks = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(blocks, padded.contiguous())
    rows = torch.cat([block[:size] for block, size in zip(blocks, sizes)], dim=0)
    return rows, sum(sizes[: dist.get_rank()])


def gather_rank_metadata(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All-gather the candidate-mask fields of every rank's rows in rank order."""

    local = [
        {key: sample[key] for key in _MASK_FIELDS if key in sample}
        for sample in samples
    ]
    gathered: list[list[dict[str, Any]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return [row for block in gathered for row in cast(list[dict[str, Any]], block)]


def all_reduce_gradients(parameters: list[nn.Parameter]) -> None:
    """Sum gradients across ranks, treating absent local gradients as zero."""

    if not parameters:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    grads = [cast(torch.Tensor, parameter.grad) for parameter in parameters]
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    for grad, reduced in zip(grads, _unflatten_dense_tensors(flat, grads)):
        grad.copy_(reduced)


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


def per_device_batch_size(global_batch_size: int, world_size: int) -> int:
    """Split one logical contrastive batch evenly across data-parallel ranks."""

    if global_batch_size <= 0 or world_size <= 0:
        raise ValueError("global batch size and world size must be positive")
    if global_batch_size % world_size:
        raise ValueError(
            f"global batch size {global_batch_size} is not divisible by "
            f"{world_size} ranks"
        )
    return global_batch_size // world_size


def whole_warmup_steps(warmup_ratio: float, max_steps: int) -> int:
    """Round a warmup ratio down to whole optimizer steps.

    Transformers rounds ratios up, so a short run with any positive ratio
    spends its first optimizer step at learning rate zero.
    """

    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if max_steps <= 0:
        raise ValueError("whole-step warmup needs a positive max_steps")
    return math.floor(warmup_ratio * max_steps)


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
        collate_workers: int = 0,
        sort_rows: bool = True,
        retain_graphs: bool = False,
        **kwargs: Any,
    ) -> None:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        super().__init__(**kwargs)
        self.micro_batch_size = micro_batch_size
        self.sort_rows = sort_rows
        self.retain_graphs = retain_graphs
        self._chunk_collator = OrderedChunkCollator(processor, workers=collate_workers)
        self._last_step_metrics: dict[str, float] = {}

    def train(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().train(*args, **kwargs)
        finally:
            self._chunk_collator.close()

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if "loss" in logs and self._last_step_metrics:
            logs = {**logs, **self._last_step_metrics}
        super().log(logs, *args, **kwargs)

    def _chunks(self, samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [
            samples[start : start + self.micro_batch_size]
            for start in range(0, len(samples), self.micro_batch_size)
        ]

    def _encode_chunk(
        self,
        model: nn.Module,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        batch = self._prepare_inputs(batch)
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
        """Cache embeddings, differentiate the full loss, then replay chunks.

        Under data parallelism each rank encodes its share of the logical
        batch, all ranks compute the identical loss over the gathered global
        batch, each replays its own rows, and LoRA gradients are summed once.
        When the rank's rows fit in one micro-batch, or ``retain_graphs`` keeps
        every chunk's autograd graph, the cache pass is skipped and the loss
        backpropagates through a single encoding of each chunk. Rows are
        optionally sorted by modality and text length first; the loss is
        permutation-equivariant, so this only reduces padding.
        """

        del num_items_in_batch
        model = _unwrap_data_parallel(model)
        distributed = _is_distributed()
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        samples = cast(list[dict[str, Any]], inputs.get("samples"))
        if not samples:
            raise ValueError("cached VLM embedding training needs raw samples")
        if self.sort_rows:
            samples = sorted(samples, key=row_length_key)

        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        chunks = self._chunks(samples)
        chunk_seeds = torch.randint(0, 2**62, (len(chunks),)).tolist()
        jobs = list(zip(chunks, chunk_seeds, strict=True))
        collated: Iterator[dict[str, Any]] = self._chunk_collator.iterate(jobs)
        collate_wait = 0.0

        def next_batch() -> dict[str, Any]:
            nonlocal collate_wait
            waited = time.perf_counter()
            batch = next(collated)
            collate_wait += time.perf_counter() - waited
            return batch

        if len(chunks) == 1 or self.retain_graphs:
            return self._single_pass_step(
                model,
                samples,
                (next_batch() for _ in chunks),
                started=started,
                collate_wait=lambda: collate_wait,
                distributed=distributed,
            )

        batches: list[dict[str, Any]] = []
        rng_states: list[_RngState] = []
        cached_by_role: dict[str, list[torch.Tensor]] = {
            "query": [],
            "positive": [],
            "negative": [],
        }
        with torch.no_grad():
            for _ in chunks:
                batch = next_batch()
                batches.append(batch)
                rng_states.append(_capture_rng_state())
                with self.compute_loss_context_manager():
                    embeddings = self._encode_chunk(model, batch)
                for role, value in embeddings.items():
                    cached_by_role[role].append(value.detach())

        cache_seconds = time.perf_counter() - started
        local = torch.stack(
            [torch.cat(cached_by_role[role], dim=0) for role in _EMBEDDING_ROLES],
            dim=1,
        )
        if distributed:
            global_rows, offset = gather_rank_rows(local)
            metadata = gather_rank_metadata(samples)
        else:
            global_rows, offset, metadata = local, 0, samples
        cached = global_rows.detach().requires_grad_()
        loss = contrastive_embedding_loss(
            cached[:, 0],
            cached[:, 1],
            cached[:, 2],
            temperature=self.embedding_temperature,
            candidate_mask=build_candidate_mask(metadata),
        )
        loss.backward()
        local_gradients = cast(torch.Tensor, cached.grad)[
            offset : offset + len(samples)
        ].detach()
        embedding_gradients = {
            role: local_gradients[:, index]
            for index, role in enumerate(_EMBEDDING_ROLES)
        }

        offset = 0
        for chunk, batch, rng_state in zip(chunks, batches, rng_states, strict=True):
            _restore_rng_state(rng_state)
            with self.compute_loss_context_manager():
                embeddings = self._encode_chunk(model, batch)
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

        return self._finish_step(
            model,
            loss,
            started=started,
            cache_seconds=cache_seconds,
            collate_wait=collate_wait,
            rows=len(samples),
            distributed=distributed,
        )

    def _single_pass_step(
        self,
        model: nn.Module,
        samples: list[dict[str, Any]],
        batches: Iterable[dict[str, Any]],
        *,
        started: float,
        collate_wait: Callable[[], float],
        distributed: bool,
    ) -> torch.Tensor:
        encoded: list[torch.Tensor] = []
        for batch in batches:
            with self.compute_loss_context_manager():
                embeddings = self._encode_chunk(model, batch)
            encoded.append(
                torch.stack([embeddings[role] for role in _EMBEDDING_ROLES], dim=1)
            )
        local = torch.cat(encoded)
        if distributed:
            global_rows, offset = gather_rank_rows(local.detach())
            global_rows = torch.cat(
                [
                    global_rows[:offset].to(local.device),
                    local,
                    global_rows[offset + len(samples) :].to(local.device),
                ]
            )
            metadata = gather_rank_metadata(samples)
        else:
            global_rows, metadata = local, samples
        loss = contrastive_embedding_loss(
            global_rows[:, 0],
            global_rows[:, 1],
            global_rows[:, 2],
            temperature=self.embedding_temperature,
            candidate_mask=build_candidate_mask(metadata),
        )
        self.accelerator.backward(loss)
        return self._finish_step(
            model,
            loss,
            started=started,
            cache_seconds=0.0,
            collate_wait=collate_wait(),
            rows=len(samples),
            distributed=distributed,
        )

    def _finish_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        *,
        started: float,
        cache_seconds: float,
        collate_wait: float,
        rows: int,
        distributed: bool,
    ) -> torch.Tensor:
        if distributed:
            all_reduce_gradients(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
            )
        self._last_step_metrics = _step_metrics(
            started=started,
            cache_seconds=cache_seconds,
            collate_wait=collate_wait,
            rows=rows,
        )
        if is_rank_zero():
            print(
                json.dumps({"vlm_embedding_step": self._last_step_metrics}), flush=True
            )
        return loss.detach()


def row_length_key(sample: dict[str, Any]) -> tuple[tuple[bool, ...], int]:
    """Group rows by which roles carry an image, then by total text length."""

    roles = [
        value if isinstance(value := sample.get(role), dict) else {}
        for role in _EMBEDDING_ROLES
    ]
    images = tuple(
        role.get("image_ref") is not None or role.get("image") is not None
        for role in roles
    )
    text = sum(len(str(role.get("text") or "")) for role in roles)
    return images, text


def _step_metrics(
    *, started: float, cache_seconds: float, collate_wait: float, rows: int
) -> dict[str, float]:
    seconds = time.perf_counter() - started
    metrics = {
        "perf/step_seconds": seconds,
        "perf/cache_pass_seconds": cache_seconds,
        "perf/collate_wait_seconds": collate_wait,
        "perf/collate_wait_fraction": collate_wait / seconds if seconds else 0.0,
        "perf/rows_per_second": rows / seconds if seconds else 0.0,
    }
    if torch.cuda.is_available():
        metrics["perf/peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        metrics["perf/peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    return metrics


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
    collate_workers = int(train_config.get("collate_workers", 0))
    checkpointing_scope = train_config.get("gradient_checkpointing_scope", "auto")

    filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=VLM_SFT_EXCLUDED_KEYS | _RUNTIME_KEYS,
        config_cls=TrainingArguments,
    )
    if not filtered.get("dataloader_num_workers", 0):
        filtered.pop("dataloader_prefetch_factor", None)
    global_batch_size = train_config.get("global_batch_size")
    if global_batch_size is not None:
        if not cached_loss:
            raise ValueError("global_batch_size requires cached_loss")
        world_size = dist.get_world_size() if _is_distributed() else 1
        filtered["per_device_train_batch_size"] = per_device_batch_size(
            int(global_batch_size), world_size
        )
        filtered["per_device_eval_batch_size"] = filtered["per_device_train_batch_size"]
        if is_rank_zero():
            print(
                json.dumps(
                    {
                        "vlm_embedding_batch": {
                            "global": int(global_batch_size),
                            "world_size": world_size,
                            "per_device": filtered["per_device_train_batch_size"],
                        }
                    }
                ),
                flush=True,
            )
    warmup_ratio = filtered.pop("warmup_ratio", None)
    if warmup_ratio is not None and int(filtered.get("max_steps", -1)) > 0:
        filtered["warmup_steps"] = whole_warmup_steps(
            float(warmup_ratio), int(filtered["max_steps"])
        )
    elif warmup_ratio is not None:
        filtered["warmup_ratio"] = warmup_ratio
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
            collate_workers=collate_workers,
            sort_rows=bool(train_config.get("sort_rows", True)),
            retain_graphs=bool(train_config.get("retain_graphs", False)),
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
    "all_reduce_gradients",
    "contrastive_embedding_loss",
    "gather_rank_metadata",
    "gather_rank_rows",
    "per_device_batch_size",
    "pool_last_attended_token",
    "vlm_embedding_run",
    "whole_warmup_steps",
]
