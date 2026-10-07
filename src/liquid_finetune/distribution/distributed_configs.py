# === Shared distributed strategy defaults ===

DEEPSPEED_ZERO2_CONFIG = {
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_clipping": "auto",
    "gradient_accumulation_steps": "auto",
    "bf16": {"enabled": "auto"},
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False,
    },
}


DEEPSPEED_ZERO2_WITH_OPTIMIZER_CONFIG = {
    **DEEPSPEED_ZERO2_CONFIG,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    },
}


MOE_DEEPSPEED_ZERO0_CONFIG = {
    "zero_optimization": {
        "stage": 0,
        "overlap_comm": True,
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_clipping": "auto",
    "gradient_accumulation_steps": "auto",
    "bf16": {"enabled": "auto"},
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False,
    },
}


# === HF FSDP defaults ===
# These are the baseline FSDP1 shapes used by config tests and older HF paths.
# The optimized MoE runners resolve their runtime FSDP2 flags below.

MOE_FSDP_CONFIG = {
    "fsdp": ["shard_grad_op", "auto_wrap"],
    "fsdp_config": {
        "transformer_layer_cls_to_wrap": "Lfm2MoeDecoderLayer",
        "backward_prefetch": "backward_pre",
        "sync_module_states": True,
        "use_orig_params": True,
    },
}

MOE_FSDP_CONFIG_LARGE = {
    "fsdp": ["full_shard", "auto_wrap"],
    "fsdp_config": {
        "transformer_layer_cls_to_wrap": "Lfm2MoeDecoderLayer",
        "backward_prefetch": "backward_pre",
        "sync_module_states": True,
        "use_orig_params": True,
        "activation_checkpointing": True,
    },
}


def resolve_reshard_after_forward(train_config: dict, default: bool) -> bool:
    """Prefer explicit YAML reshard policy, otherwise use the runner default."""
    config_value = train_config.get("reshard_after_forward")
    if config_value is not None:
        return bool(config_value)
    return default


def resolve_fsdp_cpu_offload(train_config: dict, default: bool = False) -> bool:
    """Prefer explicit YAML CPU offload policy, otherwise use the runner default."""
    config_value = train_config.get("fsdp_cpu_offload")
    if config_value is not None:
        return bool(config_value)
    return default


def strip_distributed_training_config(train_config: dict, *, num_workers: int) -> dict:
    """Drop distributed strategy config for effective single-worker runs."""
    if num_workers != 1:
        return train_config

    stripped = dict(train_config)
    for key in ("deepspeed", "fsdp", "fsdp_config"):
        stripped.pop(key, None)
    return stripped
