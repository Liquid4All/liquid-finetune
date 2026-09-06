import logging

from transformers import Trainer, TrainingArguments
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from liquid_finetune.training.default_configs.sft_configs import SFT_EXCLUDED_KEYS
from liquid_finetune.training.utils.worker_setup import (
    default_eval_batch_size,
    resolve_train_eval_datasets,
    init_tracking_from_config,
    load_causal_lm_for_training,
)
from liquid_finetune.checkpointing.callback import LiquidCheckpointCallback
from liquid_finetune.evaluation import (
    create_llm_benchmarks_from_config,
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
    CausalLMLossTokenCountMixin,
    RayDataLoaderMixin,
)
from liquid_finetune.training.utils.trainer_lifecycle import (
    run_training_safely,
)
from liquid_finetune.training.utils.config_filter import filter_runtime_config_kwargs

logger = logging.getLogger(__name__)


class LFMDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """TRL collator that applies masks produced by Liquid's distributed tokenizer."""

    def torch_call(self, examples):
        masked_examples = []
        for example in examples:
            item = dict(example)
            masks = [
                item.pop(name)
                for name in ("assistant_masks", "completion_mask")
                if name in item
            ]
            if masks:
                input_ids = item["input_ids"]
                if any(len(mask) != len(input_ids) for mask in masks):
                    raise ValueError("SFT loss mask length must match input_ids")
                labels = list(item.get("labels", input_ids))
                item["labels"] = [
                    label if all(mask[index] for mask in masks) else -100
                    for index, label in enumerate(labels)
                ]
            masked_examples.append(item)
        return super().torch_call(masked_examples)


class LFMSFTTrainer(RayDataLoaderMixin, CausalLMLossTokenCountMixin, Trainer):
    """SFT trainer with Ray-sharded data loaders."""


def build_sft_data_collator(tokenizer, train_config: dict):
    padding_free = train_config.get("padding_free", train_config.get("packing", False))
    return LFMDataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        padding_free=padding_free,
    )


def sft_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    """Run SFT locally or inside a Ray Train worker."""
    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "liquid-ft-run")
    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("liquid_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    adapter_path = train_config.get("adapter_path")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    excluded_keys = SFT_EXCLUDED_KEYS | {
        "liquid_run_name_template",
        "model_config",
    }
    train_config_filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=excluded_keys,
        config_cls=TrainingArguments,
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
        "remove_unused_columns": False,
        **train_config_filtered,
    }
    training_args = TrainingArguments(**config_kwargs)

    model, tokenizer = load_causal_lm_for_training(
        training_config,
        model_name=model_name,
        train_config=train_config,
    )

    if adapter_path:
        model = load_peft_adapter(model, adapter_path)
    elif peft_config:
        model = apply_peft_to_model(model, peft_config)

    data_collator = build_sft_data_collator(tokenizer, train_config)
    trainer = LFMSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.add_callback(LiquidCheckpointCallback(run_name_template=run_name_template))
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
