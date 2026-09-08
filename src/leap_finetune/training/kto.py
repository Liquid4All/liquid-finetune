import logging
from typing import cast

from leap_finetune.distribution.ray_runtime import normalize_visible_devices  # noqa: F401

from ray.train.huggingface.transformers import prepare_trainer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from trl import KTOConfig, KTOTrainer

from leap_finetune.training.utils.worker_setup import (
    default_eval_batch_size,
    get_ray_train_eval_datasets,
    init_tracking_from_config,
    load_causal_lm_for_training,
    setup_training_worker,
)
from leap_finetune.checkpointing.callback import LeapCheckpointCallback
from leap_finetune.evaluation import (
    create_llm_benchmarks_from_config,
    make_eval_callback,
)
from leap_finetune.training.utils.logging import (
    finish_tracker,
    get_wandb_run_id,
    is_rank_zero,
)
from leap_finetune.training.peft.peft import (
    apply_peft_to_model,
    load_peft_adapter,
    merge_and_save_peft_model,
)
from leap_finetune.training.utils.trainer_mixins import (
    RayDataLoaderMixin,
)
from leap_finetune.training.utils.trainer_lifecycle import (
    run_training_safely,
)
from leap_finetune.training.utils.config_filter import (
    BASE_RUNTIME_EXCLUDED_KEYS,
    DISTRIBUTED_RUNTIME_EXCLUDED_KEYS,
    MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS,
    MODEL_RUNTIME_EXCLUDED_KEYS,
    filter_runtime_config_kwargs,
)

logger = logging.getLogger(__name__)


class LFMKTOTrainer(RayDataLoaderMixin, KTOTrainer):
    """KTO trainer with Ray-sharded data loaders."""

    def get_train_dataloader(self):
        # KTO precomputes KL completions in fixed train-size groups. Keep the
        # loader in that order and drop incomplete groups.
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            drop_last=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No evaluation dataset configured for this run")
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            drop_last=True,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # The plain Ray DataLoaders skip Accelerate's device placement and
        # KTOTrainer.prediction_step does not prepare inputs itself.
        inputs = self._prepare_inputs(inputs)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def kto_run(training_config: dict) -> None:
    """KTO training loop for Ray-sharded unpaired preference datasets.

    Unlike DPO, KTO rows reach the workers untokenized (prompt/completion/
    label): TRL's KTOTrainer tokenizes and derives its KL mismatched-pair
    columns inside __init__, batched by per_device_train_batch_size, so
    replicating that pipeline in Ray would couple us to experimental TRL
    internals. Each worker processes only its own shard.
    """
    setup_training_worker()
    train_dataset, eval_dataset = get_ray_train_eval_datasets()

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")
    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("leap_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    adapter_path = train_config.get("adapter_path")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    excluded_keys = (
        BASE_RUNTIME_EXCLUDED_KEYS
        | MODEL_RUNTIME_EXCLUDED_KEYS
        | DISTRIBUTED_RUNTIME_EXCLUDED_KEYS
        | MANUAL_SHARDED_RUNTIME_EXCLUDED_KEYS
    )
    train_config_filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=excluded_keys,
        config_cls=KTOConfig,
    )

    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    default_eval_batch_size(train_config_filtered)
    train_batch_size = train_config_filtered["per_device_train_batch_size"]
    eval_batch_size = train_config_filtered["per_device_eval_batch_size"]
    if train_batch_size < 2:
        raise ValueError("KTO requires per_device_train_batch_size >= 2")
    if eval_batch_size != train_batch_size:
        raise ValueError(
            "KTO requires per_device_eval_batch_size to match "
            "per_device_train_batch_size"
        )
    for dataset_name, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        if dataset is not None and len(dataset) < train_batch_size:
            raise ValueError(
                f"KTO {dataset_name} dataset must have at least "
                f"{train_batch_size} rows per worker"
            )

    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    training_args = KTOConfig(**config_kwargs)

    model, tokenizer = load_causal_lm_for_training(
        training_config,
        model_name=model_name,
        train_config=train_config,
    )

    if adapter_path:
        # KTOTrainer clones the pretrained adapter into a frozen "ref" adapter
        # so the loaded policy also serves as the reference model.
        model = load_peft_adapter(model, adapter_path)
    elif peft_config:
        model = apply_peft_to_model(model, peft_config)

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = LFMKTOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=cast(PreTrainedTokenizerBase, tokenizer),
    )
    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))
    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_llm_benchmarks_from_config(benchmark_configs, tokenizer)
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

    if (peft_config or adapter_path) and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)
