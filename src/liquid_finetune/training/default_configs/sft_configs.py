from liquid_finetune import SFT_OUTPUT_PATH
from liquid_finetune.distribution.distributed_configs import (
    DEEPSPEED_ZERO2_WITH_OPTIMIZER_CONFIG,
    MOE_DEEPSPEED_ZERO0_CONFIG,
)
from liquid_finetune.training.utils.config_filter import (
    BASE_RUNTIME_EXCLUDED_KEYS,
    DISTRIBUTED_RUNTIME_EXCLUDED_KEYS,
    MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS,
    MODEL_RUNTIME_EXCLUDED_KEYS,
)

# Keys that do not belong in TrainingArguments.
# Some of these are still consumed by preprocessing/collation (for example
# assistant/completion-only loss masking) and must not be dropped upstream.
SFT_EXCLUDED_KEYS = {
    "packing",
    "max_length",
    "drop_overlength",
    "padding_free",
    "shuffle_dataset",
    "dataset_text_field",
    "dataset_kwargs",
    "dataset_num_proc",
    "completion_only_loss",
    "assistant_only_loss",
} | (
    BASE_RUNTIME_EXCLUDED_KEYS
    | MODEL_RUNTIME_EXCLUDED_KEYS
    | DISTRIBUTED_RUNTIME_EXCLUDED_KEYS
    | MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS
)


########################
#     SFT CONFIGS      #
########################


DEFAULT_SFT = {
    "training_type": "sft",
    "output_dir": SFT_OUTPUT_PATH,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "group_by_length": True,
    "packing": False,
    "padding_free": False,
    "max_length": 2048,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.2,
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "ddp_find_unused_parameters": False,
    "deepspeed": DEEPSPEED_ZERO2_WITH_OPTIMIZER_CONFIG,
}


########################
#   MOE SFT CONFIGS    #
########################

# Base MoE SFT config - distributed strategy is applied automatically in runner
# based on PEFT presence: DeepSpeed for LoRA, FSDP for full fine-tuning
MOE_SFT = {
    "training_type": "moe_sft",
    "output_dir": SFT_OUTPUT_PATH,
    "num_train_epochs": 2,  # MoE models typically need fewer epochs
    "per_device_train_batch_size": 2,  # Reduced to save memory
    "group_by_length": True,
    "packing": False,
    "padding_free": False,
    "max_length": 2048,
    "gradient_accumulation_steps": 1,  # Set to 1 to match Accelerate config for testing
    "learning_rate": 5e-5,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.2,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "max_grad_norm": 1.0,
    "bf16": True,
    "manual_sharded_checkpoint_format": "hf",
    # Distributed strategy will be set automatically:
    # - With PEFT: uses MOE_DEEPSPEED_CONFIG
    # - Without PEFT: uses FSDP_CONFIG
    "deepspeed": MOE_DEEPSPEED_ZERO0_CONFIG,
}
