import logging
import math
from copy import deepcopy

import numpy as np
from transformers import ProcessorMixin
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForVisionPreference

from leap_finetune.quantization.qat import (
    finalize_qat_after_peft,
    prepare_dpo_reference_model,
    prepare_model_for_qat,
)
from leap_finetune.evaluation import (
    BenchmarkEvalCallback,
    create_vlm_benchmarks_from_config,
)
from leap_finetune.checkpointing.callback import LeapCheckpointCallback
from leap_finetune.data_loading.vlm_batching import add_vlm_tile_counts
from leap_finetune.checkpointing.model_loading import load_vlm_model
from leap_finetune.data_loading.image_loader import load_image
from leap_finetune.training.default_configs.vlm_sft_configs import (
    DEFAULT_LR_MULTIPLIERS,
)
from leap_finetune.training.peft.peft import (
    apply_peft_to_model,
    load_peft_adapter,
    merge_and_save_peft_model,
)
from leap_finetune.training.utils.logging import (
    finish_tracker,
    is_rank_zero,
)
from leap_finetune.training.utils.trainer_lifecycle import run_training_safely
from leap_finetune.training.utils.trainer_mixins import RayDataLoaderMixin
from leap_finetune.training.utils.vlm_optimizer import (
    build_vlm_param_groups,
    create_vlm_optimizer,
    freeze_vlm_modules,
    log_per_group_lrs,
)
from leap_finetune.training.utils.worker_setup import (
    init_tracking_from_config,
    resolve_train_eval_datasets,
)
from leap_finetune.training.utils.config_filter import (
    BASE_RUNTIME_EXCLUDED_KEYS,
    DISTRIBUTED_RUNTIME_EXCLUDED_KEYS,
    MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS,
    MODEL_RUNTIME_EXCLUDED_KEYS,
    VLM_RUNTIME_EXCLUDED_KEYS,
    filter_runtime_config_kwargs,
)

logger = logging.getLogger(__name__)


def _to_builtin(obj):
    """Convert Arrow/Pandas nested values back to plain Python containers."""
    if isinstance(obj, np.ndarray):
        return [_to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, list):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj


class PathLoadingVisionPreferenceCollator(DataCollatorForVisionPreference):
    """TRL VLM-DPO collator that accepts image paths in Arrow/Ray datasets."""

    @staticmethod
    def _open_rgb_image(path: str):
        return load_image(path)

    def torch_call(self, examples):
        examples = [deepcopy(example) for example in examples]
        opened_images = []
        try:
            for example in examples:
                for key in ("prompt", "chosen", "rejected", "images"):
                    if key in example:
                        example[key] = _to_builtin(example[key])
                if "image" in example and isinstance(example["image"], str):
                    img = self._open_rgb_image(example["image"])
                    opened_images.append(img)
                    example["image"] = img
                elif "images" in example:
                    loaded = []
                    for image in example["images"]:
                        if isinstance(image, str):
                            img = self._open_rgb_image(image)
                            opened_images.append(img)
                            loaded.append(img)
                        else:
                            loaded.append(image)
                    example["images"] = loaded
            return super().torch_call(examples)
        finally:
            for img in opened_images:
                if hasattr(img, "close"):
                    img.close()


class LFMVLMDPOTrainer(RayDataLoaderMixin, DPOTrainer):
    """VLM DPO trainer with Ray dataloading and VLM optimizer groups."""

    def __init__(
        self,
        lr_multipliers: dict[str, float] | None = None,
        optimizer_type: str = "adamw",
        group_by_image_tiles: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr_multipliers = lr_multipliers or DEFAULT_LR_MULTIPLIERS
        self.optimizer_type = optimizer_type
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
        self.optimizer = create_vlm_optimizer(
            optimizer_groups, optimizer_type=self.optimizer_type, betas=betas
        )
        return self.optimizer

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        log_per_group_lrs(self.optimizer, self._optimizer_group_names, logs)
        super().log(logs, *args, **kwargs)


def vlm_dpo_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    """Run VLM DPO locally or inside a Ray Train worker."""
    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")
    train_config = training_config.get("train_config", {})
    group_by_image_tiles = bool(train_config.get("group_by_image_tiles", False))

    max_image_tokens = train_config.get("max_image_tokens")
    do_image_splitting = train_config.get("do_image_splitting", True)
    run_name_template = train_config.get("leap_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    adapter_path = train_config.get("adapter_path")
    output_dir = train_config.get("output_dir", "")
    freeze_vision_encoder = bool(train_config.get("freeze_vision_encoder", False))
    optimizer_type = train_config.get("optimizer_type", "adamw")
    lr_multipliers = dict(DEFAULT_LR_MULTIPLIERS)
    if "lr_multipliers" in train_config:
        lr_multipliers.update(train_config["lr_multipliers"])

    excluded_keys = (
        BASE_RUNTIME_EXCLUDED_KEYS
        | MODEL_RUNTIME_EXCLUDED_KEYS
        | DISTRIBUTED_RUNTIME_EXCLUDED_KEYS
        | MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS
        | VLM_RUNTIME_EXCLUDED_KEYS
        | {"freeze_vision_encoder", "optimizer_type"}
    )
    train_config_filtered, unsupported = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=excluded_keys,
        config_cls=DPOConfig,
    )
    if train_config_filtered.get("precompute_ref_log_probs"):
        logger.info(
            "Disabling precompute_ref_log_probs for VLM-DPO; TRL precomputes "
            "inside DPOTrainer.__init__ before Ray/Accelerate moves the model."
        )
        train_config_filtered["precompute_ref_log_probs"] = False

    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    num_samples = len(train_dataset)
    train_batch_size = train_config_filtered.get("per_device_train_batch_size", 1)
    grad_accum = train_config_filtered.get("gradient_accumulation_steps", 1)
    epochs = train_config_filtered.get("num_train_epochs", 1)
    steps_per_epoch = math.ceil(num_samples / train_batch_size)
    max_steps = max(1, steps_per_epoch * epochs // grad_accum)

    train_config_filtered.pop("num_train_epochs", None)
    if not train_config_filtered.get("dataloader_num_workers", 0):
        train_config_filtered.pop("dataloader_prefetch_factor", None)
    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        "remove_unused_columns": False,
        "max_steps": max_steps,
        **train_config_filtered,
    }
    if "per_device_eval_batch_size" not in config_kwargs:
        config_kwargs["per_device_eval_batch_size"] = train_batch_size
    if unsupported:
        logger.info("Dropping unsupported DPOConfig keys: %s", unsupported)
    training_args = DPOConfig(**config_kwargs)

    model, processor = load_vlm_model(
        model_name,
        max_image_tokens=max_image_tokens,
        do_image_splitting=do_image_splitting,
    )
    prepare_model_for_qat(
        model,
        train_config,
        is_vlm=True,
        uses_peft=bool(peft_config or adapter_path),
        resume_from_checkpoint=resume_from,
    )
    ref_model = prepare_dpo_reference_model(
        train_config,
        policy_uses_peft=bool(peft_config or adapter_path),
        load_model=lambda: load_vlm_model(
            model_name,
            max_image_tokens=max_image_tokens,
            do_image_splitting=do_image_splitting,
        )[0],
        is_vlm=True,
    )
    if adapter_path:
        model = load_peft_adapter(model, adapter_path)
    elif peft_config:
        model = apply_peft_to_model(model, peft_config)
    finalize_qat_after_peft(model)
    if freeze_vision_encoder:
        freeze_vlm_modules(model, ["model.vision_tower"])
    if group_by_image_tiles:
        train_dataset = add_vlm_tile_counts(train_dataset, processor)

    if not isinstance(processor, ProcessorMixin):
        raise TypeError("VLM DPO requires an AutoProcessor-compatible processor")
    collator = PathLoadingVisionPreferenceCollator(
        processor=processor,
        max_length=training_args.max_length,
        pad_to_multiple_of=training_args.pad_to_multiple_of,
    )

    trainer = LFMVLMDPOTrainer(
        lr_multipliers=lr_multipliers,
        optimizer_type=optimizer_type,
        group_by_image_tiles=group_by_image_tiles,
        model=model,
        ref_model=ref_model,
        args=training_args,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))

    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_vlm_benchmarks_from_config(benchmark_configs, processor)
        if benchmarks:
            trainer.add_callback(
                BenchmarkEvalCallback(
                    benchmarks,
                    best_metric_config=benchmark_configs.get("best_checkpoint_metrics"),
                )
            )

    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    if (peft_config or adapter_path) and is_rank_zero():
        merge_and_save_peft_model(
            model, processor, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)
