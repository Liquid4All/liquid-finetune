import logging
import os
import pathlib
import re
import shutil
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from transformers.trainer import TRAINING_ARGS_NAME

from liquid_finetune.checkpointing.model_loading import (
    LFM_TIED_WORD_EMBEDDING_MODEL_TYPES,
    normalize_model_config_overrides,
)

logger = logging.getLogger(__name__)

HF_EXPORT_MAX_SHARD_SIZE = "50GB"


def _global_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _checkpoint_staging_dir(output_dir: str, staging_root: str) -> str:
    checkpoint_name = os.path.basename(output_dir.rstrip(os.sep))
    return os.path.join(staging_root, checkpoint_name)


def _promote_directory_contents(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


# ==== HF export tensor layout ====
# Training keeps LFM2-MoE expert tensors packed for efficient grouped compute.
# Standalone HF checkpoints need the public per-expert key layout.
_LFM2_MOE_GATE_UP_PROJ_RE = re.compile(
    r"^(?P<prefix>.*\.feed_forward\.experts)\.gate_up_proj$"
)
_LFM2_MOE_DOWN_PROJ_RE = re.compile(
    r"^(?P<prefix>.*\.feed_forward\.experts)\.down_proj$"
)


def _canonicalize_hf_export_state_dict(
    model_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Convert training-packed LFM2 MoE tensors to the public HF checkpoint layout."""
    canonical_state_dict: dict[str, Any] = {}

    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor):
            gate_up_match = _LFM2_MOE_GATE_UP_PROJ_RE.match(key)
            if gate_up_match is not None:
                if value.ndim != 3 or value.shape[1] % 2 != 0:
                    raise ValueError(
                        f"Expected packed gate/up tensor {key!r} to have shape "
                        "[num_experts, 2 * intermediate_size, hidden_size], got "
                        f"{tuple(value.shape)}"
                    )

                prefix = gate_up_match.group("prefix")
                gate_proj, up_proj = value.chunk(2, dim=1)
                for expert_idx in range(value.shape[0]):
                    canonical_state_dict[f"{prefix}.{expert_idx}.w1.weight"] = (
                        gate_proj[expert_idx]
                    )
                    canonical_state_dict[f"{prefix}.{expert_idx}.w3.weight"] = up_proj[
                        expert_idx
                    ]
                continue

            down_match = _LFM2_MOE_DOWN_PROJ_RE.match(key)
            if down_match is not None:
                if value.ndim != 3:
                    raise ValueError(
                        f"Expected packed down tensor {key!r} to have shape "
                        "[num_experts, hidden_size, intermediate_size], got "
                        f"{tuple(value.shape)}"
                    )

                prefix = down_match.group("prefix")
                for expert_idx in range(value.shape[0]):
                    canonical_state_dict[f"{prefix}.{expert_idx}.w2.weight"] = value[
                        expert_idx
                    ]
                continue

        canonical_state_dict[key] = value

    return canonical_state_dict


def _is_lfm2_packed_expert_key(key: str) -> bool:
    return (
        _LFM2_MOE_GATE_UP_PROJ_RE.match(key) is not None
        or _LFM2_MOE_DOWN_PROJ_RE.match(key) is not None
    )


def _reassemble_ep_expert_state_dict(
    model_state_dict: dict[str, Any],
    ep_group: dist.ProcessGroup | None,
) -> dict[str, Any]:
    """Gather EP-sliced packed experts back to the full HF expert dimension."""
    if ep_group is None or not dist.is_available() or not dist.is_initialized():
        return model_state_dict

    ep_size = dist.get_world_size(ep_group)
    if ep_size == 1:
        return model_state_dict

    rank = _global_rank()
    should_keep_full = rank == 0
    current_device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    reassembled = dict(model_state_dict)
    expert_keys = [
        key
        for key, value in model_state_dict.items()
        if isinstance(value, torch.Tensor) and _is_lfm2_packed_expert_key(key)
    ]
    local_count = torch.tensor([len(expert_keys)], device=current_device)
    gathered_counts = [torch.empty_like(local_count) for _ in range(ep_size)]
    dist.all_gather(gathered_counts, local_count, group=ep_group)
    counts = [int(count.item()) for count in gathered_counts]
    if len(set(counts)) != 1:
        raise RuntimeError(
            "EP HF export expected every rank in an EP group to expose the same "
            f"packed expert keys, got counts={counts}"
        )
    if not expert_keys:
        return reassembled

    logger.info(
        "EP HF export reassembly start rank=%s ep_size=%s packed_tensors=%s",
        rank,
        ep_size,
        len(expert_keys),
    )

    for index, key in enumerate(expert_keys, start=1):
        value = model_state_dict[key]
        if value.ndim < 1:
            raise ValueError(f"Expected packed expert tensor {key!r}, got scalar")

        # === EP export reassembly ===
        # Runtime EP physically keeps only this rank's contiguous expert slice.
        # HF checkpoints must expose the original global expert dimension, so the
        # first EP group gathers local packed tensors in ep-rank order before the
        # usual LFM2 packed -> public expert-key canonicalization runs.
        local_value = value.contiguous().to(current_device, non_blocking=True)
        gathered = [torch.empty_like(local_value) for _ in range(ep_size)]
        dist.all_gather(gathered, local_value, group=ep_group)

        if should_keep_full:
            reassembled[key] = torch.cat(gathered, dim=0).cpu()

        del local_value, gathered
        if index == 1 or index % 10 == 0 or index == len(expert_keys):
            logger.info(
                "EP HF export reassembled %s/%s packed expert tensors on rank=%s",
                index,
                len(expert_keys),
                rank,
            )

    if should_keep_full:
        logger.info(
            "Reassembled EP expert tensors for HF export across %s EP ranks", ep_size
        )
    return reassembled


def _move_state_dict_tensors_to_cpu(
    model_state_dict: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in model_state_dict.items()
    }


def _unwrap_model_for_metadata(model: torch.nn.Module) -> torch.nn.Module:
    """Best-effort unwrap for metadata attributes on wrapped modules."""
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        for attr in ("module", "_orig_mod"):
            wrapped = getattr(current, attr, None)
            if wrapped is not None:
                current = wrapped
                break
        else:
            return current
    return current


def _copy_custom_model_code_from_config(config: Any, export_dir: str) -> None:
    """Copy local custom-code files referenced by auto_map when available."""
    auto_map = getattr(config, "auto_map", None)
    source_dir = pathlib.Path(getattr(config, "_name_or_path", "") or "")
    if not auto_map or not source_dir.is_dir():
        return

    module_files = set()
    for target in auto_map.values():
        if isinstance(target, (list, tuple)):
            targets = target
        else:
            targets = [target]
        for item in targets:
            if isinstance(item, str) and "." in item:
                module_files.add(f"{item.split('.', 1)[0]}.py")

    for filename in module_files:
        src = source_dir / filename
        if src.is_file():
            shutil.copy2(src, pathlib.Path(export_dir) / filename)


def _unwrap_model_for_hf_export(model: torch.nn.Module, accelerator) -> torch.nn.Module:
    if accelerator is None:
        return _unwrap_model_for_metadata(model)
    try:
        return accelerator.unwrap_model(model, keep_torch_compile=False)
    except TypeError:
        return accelerator.unwrap_model(model)


def _enforce_lfm_tied_lm_head_for_hf_export(
    *,
    config: Any,
    state_dict: dict[str, Any],
) -> None:
    """Preserve the LFM tied-embedding architecture in HF exports."""
    if getattr(config, "model_type", "") not in LFM_TIED_WORD_EMBEDDING_MODEL_TYPES:
        return

    config.tie_word_embeddings = True
    removed = [
        key
        for key in list(state_dict)
        if key == "lm_head.weight" or key.endswith(".lm_head.weight")
    ]
    for key in removed:
        state_dict.pop(key, None)

    if removed:
        logger.info(
            "Dropped %s redundant lm_head weight tensor(s) from tied HF export; "
            "loaders will reconstruct the output head from input embeddings.",
            len(removed),
        )


def _rope_type(rope_config: dict[str, Any]) -> str | None:
    return rope_config.get("rope_type") or rope_config.get("type")


def _apply_yarn_export_config(config: Any, original_max_position_embeddings) -> None:
    rope_config = None
    if isinstance(getattr(config, "rope_parameters", None), dict):
        rope_config = dict(config.rope_parameters)
    elif isinstance(getattr(config, "rope_scaling", None), dict):
        rope_config = dict(config.rope_scaling)

    if not rope_config or _rope_type(rope_config) != "yarn":
        return

    if (
        "original_max_position_embeddings" not in rope_config
        and original_max_position_embeddings is not None
    ):
        rope_config["original_max_position_embeddings"] = int(
            original_max_position_embeddings
        )

    # Keep the LFM-native field and the standard HF/vLLM field in sync. Different
    # consumers look at different names, and both must describe the trained RoPE.
    rope_parameters = dict(rope_config)
    rope_parameters["rope_type"] = _rope_type(rope_parameters)
    config.rope_parameters = rope_parameters

    rope_scaling = dict(rope_config)
    rope_scaling["type"] = _rope_type(rope_scaling)
    rope_scaling["rope_type"] = _rope_type(rope_scaling)
    if isinstance(getattr(type(config), "rope_scaling", None), property):
        # Newer Transformers configs expose rope_scaling as a compatibility
        # property over rope_parameters. Write the literal key as well so
        # config.json remains readable by HF/vLLM consumers that expect it.
        config.__dict__["rope_scaling"] = rope_scaling
    else:
        config.rope_scaling = rope_scaling


def _apply_model_config_for_hf_export(
    *,
    model_to_save: torch.nn.Module,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    """Patch the config so HF/vLLM loaders see the same model semantics as training."""
    config = getattr(model_to_save, "config", None)
    if config is None:
        return

    original_max_position_embeddings = getattr(config, "max_position_embeddings", None)
    model_config = {}
    if export_metadata:
        model_config = dict(export_metadata.get("model_config") or {})

    if model_config:
        normalized = normalize_model_config_overrides(config, model_config)
        for key, value in normalized.items():
            setattr(config, key, value)

    max_length = export_metadata.get("max_length") if export_metadata else None
    if max_length is not None:
        current_max = getattr(config, "max_position_embeddings", None)
        if current_max is None or int(current_max) < int(max_length):
            config.max_position_embeddings = int(max_length)

    _apply_yarn_export_config(config, original_max_position_embeddings)


def _save_hf_pretrained_model(
    *,
    model_to_save: torch.nn.Module,
    state_dict: dict[str, Any],
    export_dir: str,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    if not hasattr(model_to_save, "save_pretrained"):
        raise TypeError(
            "Manual-sharded HF export requires an unwrapped model with "
            "save_pretrained(...)."
        )

    _apply_model_config_for_hf_export(
        model_to_save=model_to_save,
        export_metadata=export_metadata,
    )
    config = getattr(model_to_save, "config", None)
    if config is not None:
        _enforce_lfm_tied_lm_head_for_hf_export(
            config=config,
            state_dict=state_dict,
        )

    os.makedirs(export_dir, exist_ok=True)
    model_to_save.save_pretrained(
        export_dir,
        state_dict=state_dict,
        is_main_process=True,
        max_shard_size=HF_EXPORT_MAX_SHARD_SIZE,
    )
    config = getattr(model_to_save, "config", None)
    if config is not None:
        _copy_custom_model_code_from_config(config, export_dir)


def _save_root_hf_export(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    staging_dir: str | None,
    ep_group: dist.ProcessGroup | None = None,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    logger.warning(
        "Manual-sharded HF export gathers a full CPU state dict on rank 0; "
        "use manual_sharded_checkpoint_format=sharded for low-memory resumable saves."
    )
    # === Full-state gather for HF export ===
    # DCP/FSDP2 makes full_state_dict + cpu_offload rank0-only. EP needs every EP
    # rank's local expert slice, so EP exports first gather full GPU state on each
    # rank, rebuild global experts, then CPU-offload only on the writing rank.
    model_state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=ep_group is None,
        ),
    )
    model_state_dict = _reassemble_ep_expert_state_dict(model_state_dict, ep_group)

    export_dir = output_dir
    if staging_dir is not None:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=True)
        export_dir = staging_dir

    if _global_rank() == 0:
        model_state_dict = _move_state_dict_tensors_to_cpu(model_state_dict)
        model_state_dict = _canonicalize_hf_export_state_dict(model_state_dict)
        model_to_save = _unwrap_model_for_hf_export(model, accelerator)
        _save_hf_pretrained_model(
            model_to_save=model_to_save,
            state_dict=model_state_dict,
            export_dir=export_dir,
            export_metadata=export_metadata,
        )

        if processing_class is not None:
            processing_class.save_pretrained(export_dir)
        elif (
            data_collator is not None
            and hasattr(data_collator, "tokenizer")
            and data_collator.tokenizer is not None
        ):
            data_collator.tokenizer.save_pretrained(export_dir)

        torch.save(training_args, os.path.join(export_dir, TRAINING_ARGS_NAME))

        if staging_dir is not None:
            _promote_directory_contents(staging_dir, output_dir)
            shutil.rmtree(staging_dir, ignore_errors=True)
    else:
        model_state_dict.clear()

    _world_barrier()


# ==== Current-model HF export ====
# These functions export the live FSDP2/EP model into a normal HF directory.
def save_fsdp2_model_checkpoint(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    checkpoint_staging_dir: str | None = None,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    """Save the HF-export layer for a manual FSDP2 model."""
    global_rank = _global_rank()
    logger.info(
        "FSDP2 root export start rank=%s output_dir=%s", global_rank, output_dir
    )
    staging_dir = None
    if checkpoint_staging_dir:
        staging_dir = _checkpoint_staging_dir(output_dir, checkpoint_staging_dir)
    _save_root_hf_export(
        model=model,
        output_dir=output_dir,
        processing_class=processing_class,
        data_collator=data_collator,
        training_args=training_args,
        staging_dir=staging_dir,
        accelerator=accelerator,
        ep_group=None,
        export_metadata=export_metadata,
    )
    logger.info("FSDP2 root export end rank=%s output_dir=%s", global_rank, output_dir)


def save_ep_model_checkpoint(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    ep_group: dist.ProcessGroup,
    checkpoint_staging_dir: str | None = None,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    """Save the HF-export layer for an EP/FSDP2 model."""
    global_rank = _global_rank()
    logger.info("EP root export start rank=%s output_dir=%s", global_rank, output_dir)
    staging_dir = None
    if checkpoint_staging_dir:
        staging_dir = _checkpoint_staging_dir(output_dir, checkpoint_staging_dir)
    _save_root_hf_export(
        model=model,
        output_dir=output_dir,
        processing_class=processing_class,
        data_collator=data_collator,
        training_args=training_args,
        staging_dir=staging_dir,
        accelerator=accelerator,
        ep_group=ep_group,
        export_metadata=export_metadata,
    )
    logger.info("EP root export end rank=%s output_dir=%s", global_rank, output_dir)


def save_manual_sharded_model_export(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    ep_group: dist.ProcessGroup | None = None,
    checkpoint_staging_dir: str | None = None,
    export_metadata: dict[str, Any] | None = None,
) -> None:
    """Export the current manual-sharded model as an HF model directory."""
    if ep_group is not None:
        save_ep_model_checkpoint(
            model=model,
            accelerator=accelerator,
            output_dir=output_dir,
            processing_class=processing_class,
            data_collator=data_collator,
            training_args=training_args,
            ep_group=ep_group,
            checkpoint_staging_dir=checkpoint_staging_dir,
            export_metadata=export_metadata,
        )
        return

    save_fsdp2_model_checkpoint(
        model=model,
        accelerator=accelerator,
        output_dir=output_dir,
        processing_class=processing_class,
        data_collator=data_collator,
        training_args=training_args,
        checkpoint_staging_dir=checkpoint_staging_dir,
        export_metadata=export_metadata,
    )
