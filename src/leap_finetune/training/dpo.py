import logging
from typing import cast

from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from leap_finetune.quantization.qat import (
    finalize_qat_after_peft,
    prepare_dpo_reference_model,
    prepare_model_for_qat,
)
from leap_finetune.training.utils.worker_setup import (
    default_eval_batch_size,
    resolve_train_eval_datasets,
    init_tracking_from_config,
    load_causal_lm_for_training,
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


class LFMDPOTrainer(RayDataLoaderMixin, DPOTrainer):
    """DPO trainer with Ray-sharded data loaders."""

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def dpo_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    """Run DPO locally or inside a Ray Train worker."""
    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

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
        config_cls=DPOConfig,
    )

    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    default_eval_batch_size(train_config_filtered)

    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    training_args = DPOConfig(**config_kwargs)

    model, tokenizer = load_causal_lm_for_training(
        training_config,
        model_name=model_name,
        train_config=train_config,
    )
    prepare_model_for_qat(
        model,
        train_config,
        uses_peft=bool(peft_config or adapter_path),
        resume_from_checkpoint=resume_from,
    )
    ref_model = prepare_dpo_reference_model(
        train_config,
        policy_uses_peft=bool(peft_config or adapter_path),
        load_model=lambda: load_causal_lm_for_training(
            training_config, model_name=model_name, train_config=train_config
        )[0],
    )

    if adapter_path:
        model = load_peft_adapter(model, adapter_path)
    elif peft_config:
        model = apply_peft_to_model(model, peft_config)
    finalize_qat_after_peft(model)

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = LFMDPOTrainer(
        model=model,
        ref_model=ref_model,
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
