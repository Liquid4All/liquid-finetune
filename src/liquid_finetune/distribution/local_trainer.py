import os

import torch
from accelerate.utils import set_seed

from liquid_finetune.checkpointing.model_info import is_moe_model_from_name
from liquid_finetune.checkpointing.model_loading import load_tokenizer
from liquid_finetune.data_loading.dataset_loader import DatasetLoader
from liquid_finetune.data_loading.local_data import create_local_datasets
from liquid_finetune.distribution.distributed_configs import (
    strip_distributed_training_config,
)
from liquid_finetune.training import TRAINING_LOOPS

_LOCAL_TYPES = frozenset(
    {"sft", "dpo", "vlm_sft", "vlm_dpo", "grpo", "vlm_grpo", "moe_sft", "moe_dpo"}
)


def should_use_local(job_config: dict) -> bool:
    """Return whether this job can use the one-process Trainer path."""
    if os.getenv("LIQUID_LAUNCHER", "auto").lower() == "ray":
        return False
    training_type = job_config["training_type"]
    is_moe = is_moe_model_from_name(job_config["model_name"])
    if training_type not in _LOCAL_TYPES:
        return False
    if training_type in {"grpo", "vlm_grpo"}:
        train_config = job_config.get("training_config") or {}
        if train_config.get("vllm_mode", "colocate") != "colocate":
            return False
        rollout_config = job_config.get("grpo_rollout") or {}
        if int(rollout_config.get("tensor_parallel_size", 1) or 1) != 1:
            return False
    if training_type.startswith("moe_") and not is_moe:
        return False
    if is_moe:
        base_type = training_type.removeprefix("moe_")
        peft_config = job_config.get("peft_config")
        moe_config = (job_config.get("training_config") or {}).get("moe_training") or {}
        peft_enabled = (
            peft_config.get("use_peft")
            if isinstance(peft_config, dict)
            else getattr(peft_config, "use_peft", None)
        )
        if (
            base_type not in {"sft", "dpo"}
            or peft_config is None
            or peft_enabled is False
        ):
            return False
        if int(moe_config.get("expert_parallel_size", 1) or 1) != 1:
            return False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return False
    ray_config = job_config.get("ray_config") or {}
    if ray_config.get("address") or os.getenv("RAY_ADDRESS"):
        return False
    if int(os.getenv("LIQUID_NUM_WORKERS", "1")) != 1:
        return False
    return int(ray_config.get("num_workers", 1) or 1) == 1


def local_trainer(job_config: dict):
    """Run supported single-GPU trainers in the current process without Ray Train."""
    set_seed(42)
    training_type = job_config["training_type"]
    train_config = strip_distributed_training_config(
        job_config["training_config"], num_workers=1
    )
    dataset_config = job_config["dataset"]
    if not isinstance(dataset_config, DatasetLoader):
        raise ValueError("Local training requires a DatasetLoader")

    is_moe = is_moe_model_from_name(job_config["model_name"])
    base_type = training_type.removeprefix("moe_")
    loop_type = f"moe_{base_type}" if is_moe else training_type

    tokenizer = None
    if base_type in {"sft", "dpo"}:
        tokenizer = load_tokenizer(
            job_config["model_name"],
            chat_template=train_config.get("chat_template"),
            chat_template_path=train_config.get("chat_template_path"),
        )
    train_dataset, eval_dataset = create_local_datasets(
        dataset_config,
        tokenizer=tokenizer,
        training_config=train_config,
    )
    loop_config = {
        "model_name": job_config["model_name"],
        "job_name": job_config.get("job_name", "liquid-ft-run"),
        "train_config": train_config,
        "peft_config": job_config.get("peft_config"),
        "model_config": job_config.get("model_config"),
        "benchmark_configs": job_config.get("benchmark_configs"),
        "rewards": job_config.get("rewards"),
        "rl_env": job_config.get("rl_env"),
        "async_eval": job_config.get("async_eval"),
        "config_dir": job_config.get("config_dir"),
    }
    print("\nTraining locally on 1 GPU without Ray Train")
    return TRAINING_LOOPS[loop_type](
        loop_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
