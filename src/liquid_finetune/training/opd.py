from __future__ import annotations

# ruff: noqa: E402

import logging
from typing import cast

import torch
from transformers import PreTrainedTokenizerBase

from liquid_finetune.distribution.ray_runtime import normalize_visible_devices

normalize_visible_devices()

from trl.experimental.distillation import DistillationConfig, DistillationTrainer

from liquid_finetune.checkpointing.callback import LiquidCheckpointCallback
from liquid_finetune.checkpointing.model_info import is_moe_model_from_name
from liquid_finetune.evaluation import (
    create_llm_benchmarks_from_config,
    make_eval_callback,
)
from liquid_finetune.training.default_configs.opd_configs import OPD_EXCLUDED_KEYS
from liquid_finetune.training.peft.peft import (
    apply_peft_to_model,
    load_peft_adapter,
    merge_and_save_peft_model,
)
from liquid_finetune.training.utils.config_filter import filter_runtime_config_kwargs
from liquid_finetune.training.utils.distillation import (
    IGNORE_INDEX,
    conversation_prompt,
    load_teacher_model,
    prepare_fsdp_teacher,
    report_vocab_size,
    resolve_shared_vocab_size,
    text_vocab_size,
    topk_generalized_jsd_loss,
)
from liquid_finetune.training.utils.logging import (
    finish_tracker,
    get_wandb_run_id,
    is_rank_zero,
)
from liquid_finetune.training.utils.trainer_lifecycle import run_training_safely
from liquid_finetune.training.utils.worker_setup import (
    init_tracking_from_config,
    load_causal_lm_for_training,
    resolve_train_eval_datasets,
)

logger = logging.getLogger(__name__)


class LFMOPDTrainer(DistillationTrainer):
    """On-policy distillation trainer with an optional top-k loss and FSDP teachers.

    The student samples completions for every prompt (with vLLM or
    ``generate``) and is trained toward the teacher on those completions. With
    ``loss_top_k=None`` and matching embedding sizes the loss is TRL's exact
    generalized JSD.
    """

    def __init__(
        self,
        model,
        teacher_model,
        args: DistillationConfig,
        *,
        loss_top_k: int | None = None,
        loss_add_tail: bool = True,
        **kwargs,
    ):
        student_vocab_size = text_vocab_size(model)
        teacher_vocab_size = text_vocab_size(teacher_model)
        self.distill_vocab_size = min(student_vocab_size, teacher_vocab_size)
        self._vocab_sizes_match = student_vocab_size == teacher_vocab_size
        if loss_top_k is not None and loss_top_k < 1:
            raise ValueError("loss_top_k must be a positive integer or None")
        if loss_top_k is not None and loss_top_k >= self.distill_vocab_size:
            logger.info(
                "loss_top_k=%d covers the full vocabulary; using the exact loss",
                loss_top_k,
            )
            loss_top_k = None
        if args.use_liger_kernel and (
            loss_top_k is not None or not self._vocab_sizes_match
        ):
            raise ValueError(
                "use_liger_kernel computes the full-vocabulary JSD from the "
                "models' output heads; it cannot be combined with loss_top_k or "
                "with student/teacher embedding sizes that differ."
            )
        self.loss_top_k = loss_top_k
        self.loss_add_tail = loss_add_tail

        # The tokenizers were validated by the caller; embedding padding may differ.
        with report_vocab_size(teacher_model, student_vocab_size):
            super().__init__(
                model=model, teacher_model=teacher_model, args=args, **kwargs
            )
        if self.is_fsdp_enabled:
            self.teacher_model = prepare_fsdp_teacher(self.accelerator, teacher_model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.use_liger_loss or (self.loss_top_k is None and self._vocab_sizes_match):
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )

        # Same normalization as DistillationTrainer: count the generated
        # completion tokens and compensate for gradient averaging across ranks.
        if self.model.training and self._buffered_num_items is not None:
            num_items_in_batch = (
                self._buffered_num_items.clamp(min=1.0) / self.accelerator.num_processes
            )

        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        teacher_logits = self._get_teacher_logits(inputs)
        width = self.distill_vocab_size
        if self.loss_top_k is None:
            prompt_length = self._compute_prompt_length(inputs)
            loss = self.generalized_jsd_loss(
                student_logits=student_outputs.logits[
                    :, prompt_length - 1 : -1, :width
                ],
                teacher_logits=teacher_logits[:, prompt_length - 1 : -1, :width],
                labels=inputs["labels"][:, prompt_length:],
                beta=self.beta,
                temperature=self.temperature,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            # Prompt and padding labels are already -100, so aligning labels
            # with the unshifted logits selects the same completion targets.
            labels = torch.nn.functional.pad(
                inputs["labels"][:, 1:], (0, 1), value=IGNORE_INDEX
            )
            loss = topk_generalized_jsd_loss(
                student_outputs.logits,
                teacher_logits,
                labels,
                beta=self.beta,
                top_k=self.loss_top_k,
                add_tail=self.loss_add_tail,
                temperature=self.temperature,
                num_items_in_batch=num_items_in_batch,
                vocab_size=width,
            )
        del teacher_logits
        return (loss, student_outputs) if return_outputs else loss


def to_prompt_dataset(dataset):
    """Convert rows to the prompt-only conversations the student continues."""
    if dataset is None:
        return None
    drop = [name for name in ("messages",) if name in dataset.column_names]
    return dataset.map(
        lambda row: {"prompt": conversation_prompt(row)}, remove_columns=drop
    )


# === On-policy distillation loop ===
#
# Like GRPO, OPD generates completions online with TRL's RepeatSampler and
# accelerate-prepared loaders. The Ray driver gives every worker the full
# dataset; TRL distributes prompt groups across ranks.


def opd_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "liquid-ft-run")
    if is_moe_model_from_name(model_name):
        raise ValueError("On-policy distillation into MoE students is not supported")

    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("liquid_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    adapter_path = train_config.get("adapter_path")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    train_config_filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=OPD_EXCLUDED_KEYS,
        config_cls=DistillationConfig,
    )
    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    # The student and teacher loaders below handle remote code. Keep TRL's
    # flag false so native vLLM registrations use their native config classes.
    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    training_args = DistillationConfig(**config_kwargs)

    # The distillation loss reads full logits, so keep the stock forward
    # instead of the memory-efficient cross-entropy patch.
    model, tokenizer = load_causal_lm_for_training(
        training_config,
        model_name=model_name,
        train_config=train_config,
        install_memory_efficient_loss=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Over-long prompts keep their most recent turns and the generation prompt.
    tokenizer.truncation_side = "left"
    teacher_model, teacher_tokenizer = load_teacher_model(train_config)
    resolve_shared_vocab_size(
        tokenizer,
        teacher_tokenizer,
        student_vocab_size=text_vocab_size(model),
        teacher_vocab_size=text_vocab_size(teacher_model),
    )

    if adapter_path:
        model = load_peft_adapter(model, adapter_path)
    elif peft_config:
        model = apply_peft_to_model(model, peft_config)

    trainer = LFMOPDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=to_prompt_dataset(train_dataset),
        eval_dataset=to_prompt_dataset(eval_dataset),
        processing_class=cast(PreTrainedTokenizerBase, tokenizer),
        loss_top_k=train_config.get("loss_top_k"),
        loss_add_tail=train_config.get("loss_add_tail", True),
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
