from liquid_finetune import DISTILL_OUTPUT_PATH
from liquid_finetune.distribution.distributed_configs import DEEPSPEED_ZERO2_CONFIG
from liquid_finetune.training.default_configs.sft_configs import SFT_EXCLUDED_KEYS

# Teacher keys are consumed by the distillation loop, which loads the teacher
# with the same conventions as the student instead of TRL's string loader.
DISTILL_TEACHER_KEYS = {
    "teacher_model_name_or_path",
    "teacher_model_revision",
    "teacher_model_config",
}

# Top-k loss options added on top of the TRL distillation loss.
DISTILL_LOSS_KEYS = {"loss_top_k", "loss_add_tail"}

# `distill` reuses the SFT tokenization pipeline, so the SFT preprocessing keys
# (max_length, packing, assistant_only_loss, ...) stay out of GKDConfig.
DISTILL_EXCLUDED_KEYS = SFT_EXCLUDED_KEYS | DISTILL_TEACHER_KEYS | DISTILL_LOSS_KEYS


########################
#   DISTILL CONFIGS    #
########################


DEFAULT_DISTILL = {
    "training_type": "distill",
    "output_dir": DISTILL_OUTPUT_PATH,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_length": 2048,
    # Off-policy KD trains on the dataset's assistant turns, so the chat
    # template must mark them with {% generation %} blocks.
    "assistant_only_loss": True,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "bf16": True,
    "ddp_find_unused_parameters": False,
    # --- Distillation (TRL GKDConfig) ---
    "lmbda": 0.0,  # fraction of steps trained on student generations
    "beta": 0.0,  # 0 = forward KL, 1 = reverse KL, in between = JSD
    "seq_kd": False,
    "temperature": 1.0,  # sampling temperature for on-policy steps
    "max_new_tokens": 256,
    # --- Memory-saving top-k loss (None = exact full-vocabulary loss) ---
    "loss_top_k": None,
    "loss_add_tail": True,
    # Like DPO's reference model, TRL initializes the frozen teacher as a ZeRO
    # stage-0 replica unless the student uses ZeRO-3.
    "deepspeed": DEEPSPEED_ZERO2_CONFIG,
}
