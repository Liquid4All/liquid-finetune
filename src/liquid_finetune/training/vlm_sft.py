import logging
import math

import torch
from transformers import Trainer, TrainingArguments

from liquid_finetune.data_loading.tokenize_data import create_vlm_collate_fn
from liquid_finetune.data_loading.vlm_batching import add_vlm_tile_counts
from liquid_finetune.training.default_configs.vlm_sft_configs import (
    DEFAULT_LR_MULTIPLIERS,
    VLM_SFT_EXCLUDED_KEYS,
)
from liquid_finetune.training.utils.worker_setup import (
    resolve_train_eval_datasets,
    init_tracking_from_config,
)
from liquid_finetune.checkpointing.callback import LiquidCheckpointCallback
from liquid_finetune.checkpointing.model_loading import load_vlm_model
from liquid_finetune.evaluation import (
    create_vlm_benchmarks_from_config,
    make_eval_callback,
)
from liquid_finetune.training.utils.logging import (
    finish_tracker,
    get_wandb_run_id,
    is_rank_zero,
)
from liquid_finetune.training.peft.peft import (
    apply_peft_to_model,
    load_peft_adapter,
    merge_and_save_peft_model,
)
from liquid_finetune.training.utils.trainer_mixins import (
    RayDataLoaderMixin,
)
from liquid_finetune.training.utils.trainer_lifecycle import (
    run_training_safely,
)
from liquid_finetune.training.utils.config_filter import filter_runtime_config_kwargs
from liquid_finetune.training.utils.vlm_optimizer import (
    build_vlm_param_groups,
    log_per_group_lrs,
)

logger = logging.getLogger(__name__)


# === VLM Trainer with per-component learning rates ===


class LFMVLMTrainer(RayDataLoaderMixin, Trainer):
    """VLM Trainer with per-component LR multipliers and Ray data integration.

    Vision encoder trains at a lower LR to preserve pretrained features,
    while the projector and LLM backbone train at the base rate.
    """

    def __init__(
        self,
        lr_multipliers: dict[str, float] | None = None,
        group_by_image_tiles: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr_multipliers = lr_multipliers or DEFAULT_LR_MULTIPLIERS
        self.group_by_image_tiles = group_by_image_tiles
        self._optimizer_group_names: list[str] = []

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        optimizer_groups, self._optimizer_group_names = build_vlm_param_groups(
            self.model,
            self.lr_multipliers,
            base_lr=self.args.learning_rate,
            weight_decay=float(self.args.weight_decay),
        )

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.AdamW(
            optimizer_groups, betas=betas, fused=torch.cuda.is_available()
        )
        return self.optimizer

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        log_per_group_lrs(self.optimizer, self._optimizer_group_names, logs)
        super().log(logs, *args, **kwargs)


# === Training loop ===


def vlm_sft_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    """Run VLM SFT locally or inside a Ray Train worker."""

    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "liquid-ft-run")
    group_by_image_tiles = bool(
        training_config.get("train_config", {}).get("group_by_image_tiles", False)
    )

    # Extract VLM-specific params and run name template before filtering
    train_config = training_config.get("train_config", {})
    max_image_tokens = train_config.get("max_image_tokens")
    do_image_splitting = train_config.get("do_image_splitting", True)
    run_name_template = train_config.get("liquid_run_name_template")
    lr_multipliers = dict(DEFAULT_LR_MULTIPLIERS)
    if "lr_multipliers" in train_config:
        lr_multipliers.update(train_config["lr_multipliers"])
    if "vision_encoder_lr_multiplier" in train_config:
        lr_multipliers["model.vision_tower"] = train_config[
            "vision_encoder_lr_multiplier"
        ]

    # Resume path is already resolved by the config parser.
    resume_from = train_config.get("resume_from_checkpoint")
    adapter_path = train_config.get("adapter_path")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    # Filter out non-TrainingArguments parameters
    excluded_keys = VLM_SFT_EXCLUDED_KEYS | {"liquid_run_name_template"}
    train_config_filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=excluded_keys,
        config_cls=TrainingArguments,
    )

    # Configure experiment tracking
    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    # Compute max_steps from materialized dataset size
    # (Trainer can't infer it from our bypassed DataLoader)
    num_samples = len(train_dataset)
    train_batch_size = train_config_filtered.get("per_device_train_batch_size", 4)
    grad_accum = train_config_filtered.get("gradient_accumulation_steps", 1)
    epochs = train_config_filtered.get("num_train_epochs", 3)
    steps_per_epoch = math.ceil(num_samples / train_batch_size)
    # int(): num_train_epochs can be a float (1.0) -> float max_steps crashes HF's
    # range(). max(1, ...): avoid 0 steps on tiny datasets / large grad accum.
    max_steps = max(1, int(steps_per_epoch * epochs // grad_accum))

    logger.info(
        "Computed max_steps=%d (samples=%d, batch=%d, accum=%d, epochs=%s)",
        max_steps,
        num_samples,
        train_batch_size,
        grad_accum,
        epochs,
    )

    # Build training args — use max_steps instead of num_train_epochs
    train_config_filtered.pop("num_train_epochs", None)
    if not train_config_filtered.get("dataloader_num_workers", 0):
        train_config_filtered.pop("dataloader_prefetch_factor", None)
    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        "per_device_eval_batch_size": train_batch_size,
        "remove_unused_columns": False,
        "max_steps": max_steps,
        **train_config_filtered,
    }
    training_args = TrainingArguments(**config_kwargs)

    # Load model + processor
    model, processor = load_vlm_model(
        model_name,
        max_image_tokens=max_image_tokens,
        do_image_splitting=do_image_splitting,
    )
    if group_by_image_tiles:
        train_dataset = add_vlm_tile_counts(train_dataset, processor)

    if adapter_path:
        model = load_peft_adapter(model, adapter_path)
    elif peft_config:
        model = apply_peft_to_model(model, peft_config)

    collate_fn = create_vlm_collate_fn(processor)

    # Initialize trainer with per-component LR multipliers
    # processing_class ensures processor + tokenizer are saved in checkpoints
    trainer = LFMVLMTrainer(
        lr_multipliers=lr_multipliers,
        group_by_image_tiles=group_by_image_tiles,
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.add_callback(LiquidCheckpointCallback(run_name_template=run_name_template))
    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_vlm_benchmarks_from_config(benchmark_configs, processor)
        if benchmarks:
            trainer.add_callback(
                make_eval_callback(
                    benchmarks=benchmarks,
                    async_eval_cfg=training_config.get("async_eval"),
                    benchmark_configs=benchmark_configs,
                    server_url=training_config.get("async_eval_server_url"),
                    eval_gpu_ids=training_config.get("async_eval_gpu_ids", ""),
                    output_dir=output_dir,
                    wandb_run_id=get_wandb_run_id(),
                    config_dir=training_config.get("config_dir"),
                )
            )

    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    # Save PEFT model if applicable
    if (peft_config or adapter_path) and is_rank_zero():
        merge_and_save_peft_model(
            model, processor, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)
