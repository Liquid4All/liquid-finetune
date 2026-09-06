from liquid_finetune import KTO_OUTPUT_PATH
from liquid_finetune.distribution.distributed_configs import DEEPSPEED_ZERO2_CONFIG


########################
#     KTO CONFIGS      #
########################


DEFAULT_KTO = {
    "training_type": "kto",
    "output_dir": KTO_OUTPUT_PATH,
    "num_train_epochs": 1,
    # Must be > 1: KTO estimates the KL term from mismatched prompt/completion
    # pairs built within each per-device batch.
    "per_device_train_batch_size": 8,
    "learning_rate": 1e-6,
    "lr_scheduler_type": "linear",
    "beta": 0.1,
    "loss_type": "kto",
    "desirable_weight": 1.0,
    "undesirable_weight": 1.0,
    "max_length": 2048,
    # KTOTrainer's collator reads the raw prompt/completion/label columns.
    "remove_unused_columns": False,
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "ddp_find_unused_parameters": False,
    "deepspeed": DEEPSPEED_ZERO2_CONFIG,
    "chat_template": None,
    "chat_template_path": None,
    "manual_sharded_checkpoint_format": "hf",
}
