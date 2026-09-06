from liquid_finetune import SFT_OUTPUT_PATH
from liquid_finetune.distribution.distributed_configs import DEEPSPEED_ZERO2_CONFIG
from liquid_finetune.training.utils.config_filter import (
    BASE_RUNTIME_EXCLUDED_KEYS,
    DISTRIBUTED_RUNTIME_EXCLUDED_KEYS,
    MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS,
    MODEL_RUNTIME_EXCLUDED_KEYS,
    VLM_RUNTIME_EXCLUDED_KEYS,
)


########################
#     SFT CONFIGS      #
########################

VLM_SFT_EXCLUDED_KEYS = (
    BASE_RUNTIME_EXCLUDED_KEYS
    | MODEL_RUNTIME_EXCLUDED_KEYS
    | DISTRIBUTED_RUNTIME_EXCLUDED_KEYS
    | MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS
    | VLM_RUNTIME_EXCLUDED_KEYS
)

# Per-component LR multipliers (applied to base learning_rate).
# Vision encoder trains at a lower LR to preserve pretrained features.
DEFAULT_LR_MULTIPLIERS = {
    "model.vision_tower": 0.1,
    "model.multi_modal_projector": 1.0,
    "model.language_model": 1.0,
}


DEFAULT_VLM_SFT = {
    "training_type": "vlm_sft",
    "max_image_tokens": None,  # None = processor default (256); set int to override
    "do_image_splitting": True,  # for VLMs, split large images into multiple tiles
    "group_by_image_tiles": False,
    "output_dir": SFT_OUTPUT_PATH,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.2,
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "eval_on_start": True,
    "gradient_checkpointing": True,
    "remove_unused_columns": False,  # preserve pixel_values, spatial_shapes, pixel_attention_mask
    "dataloader_drop_last": True,  # avoid batch size mismatches in DDP
    # Keep image/token collation off the training critical path. Two workers is
    # the portable default across AMD and H100; increase explicitly on
    # CPU-rich AMD nodes when image preprocessing is the bottleneck.
    "dataloader_num_workers": 2,
    "dataloader_prefetch_factor": 2,
    "dataloader_persistent_workers": True,
    "dataloader_pin_memory": True,
    "lr_multipliers": DEFAULT_LR_MULTIPLIERS,
    # No optimizer block here: VLMTrainer.create_optimizer() builds per-component
    # param groups that DeepSpeed's optimizer integration would discard.
    "deepspeed": DEEPSPEED_ZERO2_CONFIG,
}
