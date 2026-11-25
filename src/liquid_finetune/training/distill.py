import logging
from typing import cast

import torch
from transformers import PreTrainedTokenizerBase
from trl.experimental.gkd import GKDConfig, GKDTrainer
from trl.experimental.utils import empty_cache

from liquid_finetune.checkpointing.callback import LiquidCheckpointCallback
from liquid_finetune.checkpointing.model_info import is_moe_model_from_name
from liquid_finetune.evaluation import (
    create_llm_benchmarks_from_config,
    make_eval_callback,
)
from liquid_finetune.training.default_configs.distill_configs import (
    DISTILL_EXCLUDED_KEYS,
)
from liquid_finetune.training.peft.peft import (
    apply_peft_to_model,
    load_peft_adapter,
    merge_and_save_peft_model,
)
from liquid_finetune.training.sft import LFMDataCollatorForLanguageModeling
from liquid_finetune.training.utils.config_filter import filter_runtime_config_kwargs
from liquid_finetune.training.utils.distillation import (
    IGNORE_INDEX,
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
from liquid_finetune.training.utils.trainer_mixins import RayDataLoaderMixin
from liquid_finetune.training.utils.worker_setup import (
    default_eval_batch_size,
    init_tracking_from_config,
    load_causal_lm_for_training,
    resolve_train_eval_datasets,
)

logger = logging.getLogger(__name__)


# === Data collation ===


def _final_supervised_span_start(supervised: torch.Tensor) -> int | None:
    """Index where the last contiguous run of supervised tokens starts."""
    supervised_positions = supervised.nonzero().flatten()
    if supervised_positions.numel() == 0:
        return None
    end = int(supervised_positions[-1])
    unsupervised = (~supervised[: end + 1]).nonzero().flatten()
    return int(unsupervised[-1]) + 1 if unsupervised.numel() else 0


class LFMDistillDataCollator:
    """Collate pretokenized SFT rows for GKD.

    Labels come from the SFT assistant/completion masks. On-policy (``lmbda``)
    and sequence-level (``seq_kd``) steps also need the generation prompt: the
    tokens before the final supervised span, left-padded for ``generate``.
    """

    def __init__(self, pad_token_id: int, *, return_prompts: bool = False):
        self.pad_token_id = pad_token_id
        self.return_prompts = return_prompts
        self._language_modeling = LFMDataCollatorForLanguageModeling(
            pad_token_id=pad_token_id, padding_free=False
        )

    def __call__(self, examples):
        batch = self._language_modeling(examples)
        if self.return_prompts:
            batch.update(self._prompts(batch))
        return batch

    def _prompts(self, batch) -> dict[str, torch.Tensor]:
        prompts = []
        for input_ids, labels, attention_mask in zip(
            batch["input_ids"], batch["labels"], batch["attention_mask"]
        ):
            start = _final_supervised_span_start(labels != IGNORE_INDEX)
            if start is None:
                start = int(attention_mask.sum())
            if start == 0:
                raise ValueError(
                    "On-policy distillation needs a prompt before the supervised "
                    "completion, but a row is supervised from its first token. "
                    "Enable assistant_only_loss or completion_only_loss."
                )
            prompts.append(input_ids[:start])

        width = max(len(prompt) for prompt in prompts)
        prompt_ids = torch.full(
            (len(prompts), width), self.pad_token_id, dtype=prompts[0].dtype
        )
        prompt_mask = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            prompt_ids[row, width - len(prompt) :] = prompt
            prompt_mask[row, width - len(prompt) :] = 1
        return {"prompts": prompt_ids, "prompt_attention_mask": prompt_mask}


# === Trainer ===


class LFMDistillTrainer(RayDataLoaderMixin, GKDTrainer):
    """GKD trainer with Ray-sharded loaders, an optional top-k loss, and FSDP teachers.

    With ``loss_top_k=None`` the loss is TRL's full-vocabulary generalized JSD.
    Otherwise the divergence is restricted to the top-k tokens of the teacher
    (forward KL), the student (reverse KL), or both (JSD), plus an optional tail
    bucket for the remaining mass.
    """

    def __init__(
        self,
        model,
        teacher_model,
        args: GKDConfig,
        *,
        loss_top_k: int | None = None,
        loss_add_tail: bool = True,
        **kwargs,
    ):
        student_vocab_size = text_vocab_size(model)
        teacher_vocab_size = text_vocab_size(teacher_model)
        self.distill_vocab_size = min(student_vocab_size, teacher_vocab_size)
        if loss_top_k is not None and loss_top_k < 1:
            raise ValueError("loss_top_k must be a positive integer or None")
        if loss_top_k is not None and loss_top_k >= self.distill_vocab_size:
            logger.info(
                "loss_top_k=%d covers the full vocabulary; using the exact loss",
                loss_top_k,
            )
            loss_top_k = None
        if args.use_liger_kernel and (
            loss_top_k is not None or student_vocab_size != teacher_vocab_size
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Ray loaders yield CPU batches. GKD generates from `prompts` before the
        # parent step moves inputs to the device.
        inputs = self._prepare_inputs(inputs)
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.use_liger_gkd_loss:
            result = super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
            if return_outputs:
                loss, outputs = result
                return self._scale_for_data_parallel(loss, num_items_in_batch), outputs
            return self._scale_for_data_parallel(result, num_items_in_batch)

        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_logits = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits

        if self.loss_top_k is None:
            width = self.distill_vocab_size
            loss = self.generalized_jsd_loss(
                student_logits=student_outputs.logits[:, :-1, :width],
                teacher_logits=teacher_logits[:, :-1, :width],
                labels=inputs["labels"][:, 1:],
                beta=self.beta,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            # Align labels with unshifted logits instead of slicing the logits,
            # so autograd never materializes a padded copy of their gradient.
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
                num_items_in_batch=num_items_in_batch,
                vocab_size=self.distill_vocab_size,
            )
        del teacher_logits
        empty_cache()

        loss = self._scale_for_data_parallel(loss, num_items_in_batch)
        return (loss, student_outputs) if return_outputs else loss

    def _scale_for_data_parallel(self, loss, num_items_in_batch):
        """Match ``Trainer.compute_loss`` scaling for globally normalized losses.

        ``num_items_in_batch`` counts targets across all ranks, and DDP/ZeRO
        average gradients across ranks, so the per-rank loss is multiplied by
        the world size to keep the gradient equal to the global token mean.
        """
        if num_items_in_batch is None or not self.args.average_tokens_across_devices:
            return loss
        return loss * self.accelerator.num_processes


# === Training loop ===


def distill_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    """Run off-policy or mixed knowledge distillation locally or on a Ray worker."""
    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "liquid-ft-run")
    if is_moe_model_from_name(model_name):
        raise ValueError("Distillation into MoE students is not supported yet")

    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("liquid_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    adapter_path = train_config.get("adapter_path")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    train_config_filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=DISTILL_EXCLUDED_KEYS,
        config_cls=GKDConfig,
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
    training_args = GKDConfig(**config_kwargs)

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

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    data_collator = LFMDistillDataCollator(
        tokenizer.pad_token_id,
        return_prompts=training_args.lmbda > 0 or training_args.seq_kd,
    )
    trainer = LFMDistillTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
