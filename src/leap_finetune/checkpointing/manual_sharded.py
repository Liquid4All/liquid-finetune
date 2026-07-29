import hashlib
import json
import logging
import os
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from transformers.trainer import SCALER_NAME, SCHEDULER_NAME
from transformers.trainer import TRAINER_STATE_NAME

from leap_finetune.checkpointing.hf_export import save_manual_sharded_model_export
from leap_finetune.quantization.qat.metadata import write_qat_metadata
from leap_finetune.checkpointing.paths import (
    current_checkpoint_output_dir,
    resolve_checkpoint_output_dir,
    rotate_named_checkpoints,
    update_latest_pointer,
)

logger = logging.getLogger(__name__)

MANUAL_SHARDED_FORMAT_VERSION = 3
MANUAL_SHARDED_METADATA_NAME = "manual_sharded_checkpoint.json"
MANUAL_SHARDED_RESUME_DIR = "manual_sharded_resume"
MANUAL_SHARDED_CHECKPOINT_FORMATS = {"sharded", "hf", "both"}
LEGACY_MANUAL_SHARDED_CHECKPOINT_FORMATS = {
    "resume_only": "sharded",
    "hf_only": "hf",
}

__all__ = [
    "MANUAL_SHARDED_CHECKPOINT_FORMATS",
    "MANUAL_SHARDED_FORMAT_VERSION",
    "MANUAL_SHARDED_METADATA_NAME",
    "MANUAL_SHARDED_RESUME_DIR",
    "build_manual_sharded_export_metadata_from_config",
    "current_checkpoint_output_dir",
    "export_manual_sharded_checkpoint_as_hf",
    "finalize_manual_sharded_export_metadata",
    "load_manual_sharded_checkpoint_metadata",
    "load_manual_sharded_model_checkpoint",
    "load_manual_sharded_optimizer_checkpoint",
    "normalize_manual_sharded_checkpoint_format",
    "save_manual_sharded_checkpoint",
    "save_manual_sharded_model_export",
    "should_run_final_manual_sharded_save",
]


# ==== Checkpoint format ====
# Manual-sharded runs can save distributed resume state, an HF export, or both.
# The metadata file makes those checkpoint directories self-describing.
def _manual_sharded_resume_dir(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, MANUAL_SHARDED_RESUME_DIR)


def _manual_sharded_metadata_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, MANUAL_SHARDED_METADATA_NAME)


def _global_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def normalize_manual_sharded_checkpoint_format(checkpoint_format: str) -> str:
    """Return canonical checkpoint format, accepting old config aliases."""
    if checkpoint_format in MANUAL_SHARDED_CHECKPOINT_FORMATS:
        return checkpoint_format
    if checkpoint_format in LEGACY_MANUAL_SHARDED_CHECKPOINT_FORMATS:
        return LEGACY_MANUAL_SHARDED_CHECKPOINT_FORMATS[checkpoint_format]
    raise ValueError(
        f"Unsupported manual_sharded_checkpoint_format={checkpoint_format!r}. "
        f"Expected one of {sorted(MANUAL_SHARDED_CHECKPOINT_FORMATS)}."
    )


def _json_safe(value: Any) -> Any:
    """Convert metadata values to JSON-safe objects without losing structure."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)


def _chat_template_metadata(processing_class) -> dict[str, Any]:
    chat_template = getattr(processing_class, "chat_template", None)

    metadata: dict[str, Any] = {}
    if isinstance(chat_template, str):
        metadata["chat_template"] = chat_template
        metadata["chat_template_sha256"] = hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()
    return metadata


def build_manual_sharded_export_metadata_from_config(
    training_config: dict,
    *,
    processing_class=None,
) -> dict[str, Any]:
    """Build durable checkpoint metadata from the original run config."""
    train_config = training_config.get("train_config", {}) or {}
    metadata: dict[str, Any] = {
        "training_config": _json_safe(training_config),
        "base_model_name": training_config.get("model_name"),
        "model_config": training_config.get("model_config") or {},
        "max_length": train_config.get("max_length"),
        "chat_template_path": train_config.get("chat_template_path"),
    }
    if train_config.get("chat_template") is not None:
        metadata["chat_template"] = train_config.get("chat_template")
    metadata.update(_chat_template_metadata(processing_class))
    return {k: v for k, v in _json_safe(metadata).items() if v is not None}


def finalize_manual_sharded_export_metadata(
    export_metadata: dict[str, Any] | None,
    *,
    processing_class=None,
) -> dict[str, Any]:
    """Finalize durable checkpoint metadata with the active tokenizer template."""
    metadata = dict(export_metadata or {})
    metadata.update(_chat_template_metadata(processing_class))
    return {k: v for k, v in _json_safe(metadata).items() if v is not None}


def _save_root_metadata(
    checkpoint_dir: str,
    *,
    save_only_model: bool,
    checkpoint_format: str,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "format_version": MANUAL_SHARDED_FORMAT_VERSION,
        "checkpoint_format": checkpoint_format,
        "resume_dir": MANUAL_SHARDED_RESUME_DIR,
        "save_only_model": bool(save_only_model),
        "has_optimizer_state": (
            checkpoint_format in {"sharded", "both"} and not bool(save_only_model)
        ),
        "has_resume_state": checkpoint_format in {"sharded", "both"},
    }
    if export_metadata:
        metadata.update(_json_safe(export_metadata))
    with open(_manual_sharded_metadata_path(checkpoint_dir), "w") as handle:
        json.dump(metadata, handle, indent=2)
    training_metadata = (export_metadata or {}).get("training_config", {})
    qat_config = (training_metadata.get("train_config", {}) or {}).get("qat")
    if qat_config:
        write_qat_metadata(checkpoint_dir, qat_config)


def _load_root_metadata(checkpoint_dir: str) -> dict[str, Any]:
    metadata_path = _manual_sharded_metadata_path(checkpoint_dir)
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path) as handle:
        return json.load(handle)


def load_manual_sharded_checkpoint_metadata(checkpoint_dir: str) -> dict[str, Any]:
    """Load manual-sharded checkpoint metadata for export tooling."""
    return _load_root_metadata(checkpoint_dir)


def _save_scheduler_state(trainer, resume_dir: str) -> None:
    if trainer.lr_scheduler is None or _global_rank() != 0:
        return
    torch.save(
        trainer.lr_scheduler.state_dict(), os.path.join(resume_dir, SCHEDULER_NAME)
    )


def _load_scheduler_state(trainer, resume_dir: str) -> None:
    scheduler_path = os.path.join(resume_dir, SCHEDULER_NAME)
    if trainer.lr_scheduler is not None and os.path.isfile(scheduler_path):
        trainer.lr_scheduler.load_state_dict(
            torch.load(scheduler_path, map_location="cpu", weights_only=True)
        )


def _save_scaler_state(trainer, resume_dir: str) -> None:
    scaler = getattr(trainer.accelerator, "scaler", None)
    if scaler is None or _global_rank() != 0:
        return
    torch.save(scaler.state_dict(), os.path.join(resume_dir, SCALER_NAME))


def _load_scaler_state(trainer, resume_dir: str) -> None:
    scaler = getattr(trainer.accelerator, "scaler", None)
    scaler_path = os.path.join(resume_dir, SCALER_NAME)
    if scaler is not None and os.path.isfile(scaler_path):
        scaler.load_state_dict(
            torch.load(scaler_path, map_location="cpu", weights_only=True)
        )


def _save_rng_state(trainer, checkpoint_dir: str) -> None:
    from transformers.trainer import Trainer

    Trainer._save_rng_state(trainer, checkpoint_dir)


# ==== Trainer checkpoint save ====
# This is the resumable path: distributed model/optimizer shards plus rank-0
# Trainer metadata, scheduler/scaler state, RNG state, and latest/rotation logic.
def save_manual_sharded_trainer_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
    ep_group: dist.ProcessGroup | None = None,
    checkpoint_format: str = "sharded",
    export_metadata: dict[str, Any] | None = None,
) -> None:
    """Save a resumable manual-sharded training checkpoint.

    These checkpoints are optimized for distributed resume, not direct model
    consumption. Use `export_manual_sharded_checkpoint_as_hf` to convert one into a
    standalone HF model directory when needed.
    """
    if trainer.hp_search_backend is None and trial is None:
        trainer.store_flos()

    run_dir = trainer._get_output_dir(trial=trial)
    output_dir = resolve_checkpoint_output_dir(
        run_dir=run_dir,
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
    )
    resume_dir = _manual_sharded_resume_dir(output_dir)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(resume_dir, exist_ok=True)
    _world_barrier()

    model_state = get_model_state_dict(model, options=options)
    resume_state: dict[str, Any] = {"model": model_state}
    if not trainer.args.save_only_model and trainer.optimizer is not None:
        resume_state["optimizer"] = get_optimizer_state_dict(
            model,
            trainer.optimizer,
            options=options,
        )

    logger.info(
        "Manual-sharded distributed checkpoint start rank=%s output_dir=%s ep=%s",
        _global_rank(),
        output_dir,
        ep_group is not None,
    )
    dcp.save(resume_state, storage_writer=FileSystemWriter(resume_dir))
    _world_barrier()

    if _global_rank() == 0:
        _save_scheduler_state(trainer, resume_dir)
        _save_scaler_state(trainer, resume_dir)
        _save_root_metadata(
            output_dir,
            save_only_model=trainer.args.save_only_model,
            checkpoint_format=checkpoint_format,
            export_metadata=export_metadata,
        )
    _world_barrier()

    if _global_rank() == 0:
        _save_rng_state(trainer, output_dir)
        trainer.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        update_latest_pointer(run_dir, output_dir)
        if (
            trainer.args.save_total_limit is not None
            and trainer.args.save_total_limit > 0
        ):
            rotate_named_checkpoints(run_dir, trainer.args.save_total_limit)
    _world_barrier()


def save_manual_sharded_hf_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
    ep_group: dist.ProcessGroup | None = None,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    """Save a named checkpoint directory containing only an HF model export.

    Unlike the resumable distributed checkpoint path, this does not save
    optimizer state or `manual_sharded_resume/`. It still writes trainer state
    metadata and updates the run-level `latest` pointer / rotation bookkeeping.
    """
    if trainer.hp_search_backend is None and trial is None:
        trainer.store_flos()

    run_dir = trainer._get_output_dir(trial=trial)
    output_dir = resolve_checkpoint_output_dir(
        run_dir=run_dir,
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
    )

    os.makedirs(output_dir, exist_ok=True)
    _world_barrier()

    save_manual_sharded_model_export(
        model=model,
        accelerator=trainer.accelerator,
        output_dir=output_dir,
        processing_class=trainer.processing_class,
        data_collator=trainer.data_collator,
        training_args=trainer.args,
        ep_group=ep_group,
        checkpoint_staging_dir=getattr(trainer, "checkpoint_staging_dir", None),
        export_metadata=export_metadata,
    )

    if _global_rank() == 0:
        _save_root_metadata(
            output_dir,
            save_only_model=True,
            checkpoint_format="hf",
            export_metadata=export_metadata,
        )
        _save_rng_state(trainer, output_dir)
        trainer.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        update_latest_pointer(run_dir, output_dir)
        if (
            trainer.args.save_total_limit is not None
            and trainer.args.save_total_limit > 0
        ):
            rotate_named_checkpoints(run_dir, trainer.args.save_total_limit)
    _world_barrier()


def save_manual_sharded_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
    checkpoint_format: str,
    ep_group: dist.ProcessGroup | None = None,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    """Save a manual-sharded checkpoint according to the configured format."""
    checkpoint_format = normalize_manual_sharded_checkpoint_format(checkpoint_format)

    if checkpoint_format == "sharded":
        save_manual_sharded_trainer_checkpoint(
            trainer=trainer,
            model=model,
            trial=trial,
            ep_group=ep_group,
            checkpoint_format="sharded",
            export_metadata=export_metadata,
        )
        return

    if checkpoint_format == "hf":
        save_manual_sharded_hf_checkpoint(
            trainer=trainer,
            model=model,
            trial=trial,
            ep_group=ep_group,
            export_metadata=export_metadata,
        )
        return

    # "both": write resumable state first so trainer metadata / latest pointer are
    # handled once, then layer the HF export into the same named checkpoint dir.
    save_manual_sharded_trainer_checkpoint(
        trainer=trainer,
        model=model,
        trial=trial,
        ep_group=ep_group,
        checkpoint_format="both",
        export_metadata=export_metadata,
    )
    output_dir = resolve_checkpoint_output_dir(
        run_dir=trainer._get_output_dir(trial=trial),
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
    )
    save_manual_sharded_model_export(
        model=model,
        accelerator=trainer.accelerator,
        output_dir=output_dir,
        processing_class=trainer.processing_class,
        data_collator=trainer.data_collator,
        training_args=trainer.args,
        ep_group=ep_group,
        checkpoint_staging_dir=getattr(trainer, "checkpoint_staging_dir", None),
        export_metadata=export_metadata,
    )


def should_run_final_manual_sharded_save(
    *,
    trainer,
    requested_save_strategy: str,
) -> bool:
    """Whether a final explicit manual-sharded save should run after training."""
    if requested_save_strategy == "no":
        return False

    output_dir = current_checkpoint_output_dir(
        output_dir=trainer._get_output_dir(trial=None),
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
        manual_sharded=True,
    )
    return not os.path.isdir(output_dir)


def load_manual_sharded_model_checkpoint(
    *, model: torch.nn.Module, checkpoint_dir: str
) -> bool:
    """Load manual-sharded model state for resume if present."""
    resume_dir = _manual_sharded_resume_dir(checkpoint_dir)
    if not os.path.isdir(resume_dir):
        return False

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state = get_model_state_dict(model, options=options)
    load_state = {"model": model_state}
    dcp.load(load_state, storage_reader=FileSystemReader(resume_dir))
    set_model_state_dict(model, load_state["model"], options=options)
    _world_barrier()
    return True


def load_manual_sharded_optimizer_checkpoint(*, trainer, checkpoint_dir: str) -> bool:
    """Load optimizer / scheduler / scaler state for manual-sharded resume."""
    resume_dir = _manual_sharded_resume_dir(checkpoint_dir)
    metadata = _load_root_metadata(checkpoint_dir)
    if not os.path.isdir(resume_dir) or metadata.get("has_optimizer_state") is not True:
        return False

    if trainer.optimizer is None:
        trainer.create_optimizer()
    if trainer.lr_scheduler is None:
        trainer.create_scheduler(
            num_training_steps=trainer.args.max_steps,
            optimizer=trainer.optimizer,
        )

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    optim_state = get_optimizer_state_dict(
        trainer.model,
        trainer.optimizer,
        options=options,
    )
    load_state = {"optimizer": optim_state}
    dcp.load(load_state, storage_reader=FileSystemReader(resume_dir))
    set_optimizer_state_dict(
        trainer.model,
        trainer.optimizer,
        load_state["optimizer"],
        options=options,
    )
    _world_barrier()
    _load_scheduler_state(trainer, resume_dir)
    _load_scaler_state(trainer, resume_dir)
    _world_barrier()
    return True


def export_manual_sharded_checkpoint_as_hf(
    *,
    model: torch.nn.Module,
    accelerator,
    checkpoint_dir: str,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    ep_group: dist.ProcessGroup | None = None,
    checkpoint_staging_dir: str | None = None,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    """Export a resumable manual-sharded checkpoint as a standalone HF model.

    The caller provides a model constructed with the same wrapping/topology as the
    original training run. This function restores the sharded model weights from
    `checkpoint_dir` and then writes a normal HF export to `output_dir`.
    """
    loaded = load_manual_sharded_model_checkpoint(
        model=model,
        checkpoint_dir=checkpoint_dir,
    )
    if not loaded:
        raise FileNotFoundError(
            f"No manual-sharded resume state found under {checkpoint_dir!r}"
        )

    checkpoint_metadata = _load_root_metadata(checkpoint_dir)
    if export_metadata:
        checkpoint_metadata.update(_json_safe(export_metadata))

    save_manual_sharded_model_export(
        model=model,
        accelerator=accelerator,
        output_dir=output_dir,
        processing_class=processing_class,
        data_collator=data_collator,
        training_args=training_args,
        ep_group=ep_group,
        checkpoint_staging_dir=checkpoint_staging_dir,
        export_metadata=checkpoint_metadata,
    )
