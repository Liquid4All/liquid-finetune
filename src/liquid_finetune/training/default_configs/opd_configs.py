from liquid_finetune import OPD_OUTPUT_PATH
from liquid_finetune.distribution.distributed_configs import DEEPSPEED_ZERO2_CONFIG
from liquid_finetune.training.default_configs.distill_configs import (
    DISTILL_LOSS_KEYS,
    DISTILL_TEACHER_KEYS,
)
from liquid_finetune.training.utils.config_filter import (
    BASE_RUNTIME_EXCLUDED_KEYS,
    MODEL_RUNTIME_EXCLUDED_KEYS,
)

# Keys that are not DistillationConfig fields. Prompts reach the workers
# untokenized, so no SFT preprocessing keys apply.
OPD_EXCLUDED_KEYS = (
    BASE_RUNTIME_EXCLUDED_KEYS
    | MODEL_RUNTIME_EXCLUDED_KEYS
    | DISTILL_TEACHER_KEYS
    | DISTILL_LOSS_KEYS
)


########################
#     OPD CONFIGS      #
########################


DEFAULT_OPD = {
    "training_type": "opd",
    "output_dir": OPD_OUTPUT_PATH,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-6,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "steps",
    "save_steps": 200,
    # Prompt-only data has no completions to score; use `evals:` benchmarks.
    "eval_strategy": "no",
    "bf16": True,
    "gradient_checkpointing": True,
    "ddp_find_unused_parameters": False,
    # --- On-policy distillation (TRL DistillationConfig) ---
    "beta": 1.0,  # reverse KL on student samples
    # Sample from the student's own distribution: no temperature, nucleus, or
    # top-k truncation.
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_length": 2048,
    "max_completion_length": 512,
    "num_generations": 1,
    "log_completions": True,
    "num_completions_to_print": 4,
    # --- Memory-saving top-k loss (None = exact full-vocabulary loss) ---
    "loss_top_k": None,
    "loss_add_tail": True,
    # --- vLLM student rollouts (colocate by default, like GRPO) ---
    "use_vllm": True,
    "vllm_mode": "colocate",
    "vllm_gpu_memory_utilization": 0.3,
    "vllm_enable_sleep_mode": True,
    "deepspeed": DEEPSPEED_ZERO2_CONFIG,
}
