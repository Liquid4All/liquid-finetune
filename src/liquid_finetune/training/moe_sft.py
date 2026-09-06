import logging
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from liquid_finetune.data_loading.length_grouping import get_length_grouped_sampler
from liquid_finetune.distribution.distributed_configs import (
    resolve_fsdp_cpu_offload,
    resolve_reshard_after_forward,
)
from liquid_finetune.training.default_configs.sft_configs import SFT_EXCLUDED_KEYS
from liquid_finetune.training.utils.worker_setup import (
    default_eval_batch_size,
    resolve_train_eval_datasets,
    init_tracking_from_config,
    load_causal_lm_for_training,
)
from liquid_finetune.training.sft import build_sft_data_collator
from liquid_finetune.checkpointing.callback import LiquidCheckpointCallback
from liquid_finetune.evaluation import (
    BenchmarkEvalCallback,
    create_llm_benchmarks_from_config,
)
from liquid_finetune.training.moe_utils.metrics import MoEMetricsCallback
from liquid_finetune.training.utils.logging import (
    finish_tracker,
    is_rank_zero,
)
from liquid_finetune.training.moe_utils.memory_trace import (
    init_memory_trace,
    wrap_optimizer_step,
    write_memory_trace_event,
)
from liquid_finetune.training.utils.trainer_mixins import (
    CausalLMLossTokenCountMixin,
    ManualShardedCheckpointMixin,
    validate_manual_sharded_training_args,
)
from liquid_finetune.training.utils.trainer_lifecycle import run_training_safely
from liquid_finetune.training.utils.config_filter import filter_runtime_config_kwargs
from liquid_finetune.checkpointing.manual_sharded import (
    build_manual_sharded_export_metadata_from_config,
    save_manual_sharded_checkpoint,
    should_run_final_manual_sharded_save,
)
from liquid_finetune.training.moe_utils.losses import (
    MoETrainingConfig,
    apply_moe_losses,
)
from liquid_finetune.training.moe_utils.ep_runtime import (
    apply_ep_to_model,
    apply_fsdp2,
    create_dp_mesh,
    create_ep_mesh,
    log_cuda_memory,
    shard_experts,
)
from liquid_finetune.training.peft.peft import (
    apply_peft_to_model,
    merge_and_save_peft_model,
)

logger = logging.getLogger(__name__)

MOE_SFT_EXCLUDED_KEYS = SFT_EXCLUDED_KEYS | {
    "liquid_run_name_template",
    "moe_training",
    "model_config",
}


class LFMMoeSFTTrainer(
    CausalLMLossTokenCountMixin, ManualShardedCheckpointMixin, Trainer
):
    """SFT Trainer for MoE models with EP/FSDP2 support."""

    def __init__(
        self,
        ep_config: dict | None = None,
        manual_fsdp2: bool = False,
        run_name_template: str | None = None,
        checkpoint_staging_dir: str | None = None,
        manual_sharded_checkpoint_format: str = "hf",
        manual_sharded_export_metadata: dict | None = None,
        **kwargs,
    ):
        self.manual_sharded = manual_fsdp2
        self.run_name_template = run_name_template
        self.checkpoint_staging_dir = checkpoint_staging_dir
        self.manual_sharded_checkpoint_format = manual_sharded_checkpoint_format
        self.manual_sharded_export_metadata = dict(manual_sharded_export_metadata or {})
        super().__init__(**kwargs)
        self.ep_config = ep_config

    def get_train_dataloader(self):
        args = getattr(self, "args", None)
        batch_size = getattr(
            args, "per_device_train_batch_size", self._train_batch_size
        )
        group_by_length = getattr(args, "group_by_length", True)
        sampler_generator = None
        if group_by_length or self.ep_config is not None:
            seed = (
                42 + int(self.ep_config["dp_rank"])
                if self.ep_config is not None
                else getattr(args, "seed", None)
            )
            if seed is not None:
                sampler_generator = torch.Generator().manual_seed(int(seed))

        sampler = None
        if group_by_length:
            sampler = get_length_grouped_sampler(
                self.train_dataset,
                batch_size,
                generator=sampler_generator,
            )
        dataloader_kwargs = {}
        if sampler is None and sampler_generator is not None:
            dataloader_kwargs["generator"] = sampler_generator
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            **dataloader_kwargs,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No evaluation dataset configured for this run")
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )

    def create_optimizer(self):
        if self.optimizer is not None:
            return wrap_optimizer_step(self.optimizer, self._memory_trace_step)

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.AdamW(
            (param for param in self.model.parameters() if param.requires_grad),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=betas,
            fused=torch.cuda.is_available(),
        )
        return wrap_optimizer_step(self.optimizer, self._memory_trace_step)

    def _memory_trace_step(self) -> int:
        return int(getattr(getattr(self, "state", None), "global_step", 0))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        output = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        write_memory_trace_event("after_forward_loss", step=self._memory_trace_step())
        return output

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        write_memory_trace_event("train_step_start", step=self._memory_trace_step())
        loss = super().training_step(model, inputs, num_items_in_batch, **kwargs)
        write_memory_trace_event("after_backward", step=self._memory_trace_step())
        return loss


def moe_sft_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    """MoE SFT training loop with revised non-EP and EP paths."""
    # ==== 1. Worker setup and config ====
    # Resolve supplied local datasets or the Ray worker shard, then keep MoE-only
    # keys out of Hugging Face TrainingArguments.
    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "liquid-ft-run")

    moe_config_dict = training_config.get("train_config", {}).get("moe_training", {})
    _validate_supported_moe_sft_config(moe_config_dict)
    moe_config = MoETrainingConfig.from_dict(moe_config_dict)
    ep_size = moe_config_dict.get("expert_parallel_size", 1) or 1
    train_config = training_config.get("train_config", {})
    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")

    use_ep = ep_size > 1
    use_fsdp2 = peft_config is None and not use_ep

    run_name_template = train_config.get("liquid_run_name_template")

    excluded_keys = MOE_SFT_EXCLUDED_KEYS | (
        {"deepspeed"} if (use_ep or use_fsdp2) else set()
    )

    train_config_filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=excluded_keys,
        config_cls=TrainingArguments,
    )
    requested_save_strategy = train_config_filtered.get("save_strategy", "no")
    manual_sharded_checkpoint_format = train_config.get(
        "manual_sharded_checkpoint_format", "hf"
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

    if use_ep or use_fsdp2:
        validate_manual_sharded_training_args(
            config_kwargs,
            checkpoint_format=manual_sharded_checkpoint_format,
        )
        if use_ep:
            logger.info("EP mode: ep_size=%s, FSDP2 on dp_mesh", ep_size)
        else:
            logger.info("Non-EP mode: manual FSDP2 on full DP mesh")

    training_args = TrainingArguments(**config_kwargs)

    # ==== 2. Load model ====
    # All ranks load the same HF model first. EP then slices expert tensors in
    # memory; non-EP keeps full experts and only patches aux-loss injection.
    init_memory_trace(training_args.output_dir, framework="liquid")
    model, tokenizer = load_causal_lm_for_training(
        training_config,
        model_name=model_name,
        train_config=train_config,
    )
    write_memory_trace_event("after_model_load", always=True)
    log_cuda_memory("after_load_model")

    if use_ep or use_fsdp2:
        logger.info("Waiting for all ranks to finish model loading...")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    ep_config = None
    device_mesh = None
    dp_mesh = None
    num_experts = getattr(model.config, "num_experts", None)

    # ==== 3. Patch MoE runtime ====
    # EP path: create DPxEP mesh, keep only local experts, then replace each MoE
    # block forward with all-to-all token dispatch. Non-EP path: keep upstream
    # grouped-mm expert compute and inject aux/z losses around routing.
    if use_ep:
        ep_config, device_mesh = create_ep_mesh(ep_size, num_experts)
        dp_mesh = device_mesh["dp"]
        shard_experts(model, ep_config)
        write_memory_trace_event("after_shard_experts", always=True)
        log_cuda_memory("after_shard_experts", summary=True)
        apply_ep_to_model(model, ep_config, moe_config=moe_config)
    else:
        apply_moe_losses(model, moe_config)
        if use_fsdp2:
            dp_mesh = create_dp_mesh()

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    # ==== 4. Apply FSDP2 ====
    # EP shards params only over the DP submesh so expert ownership stays local.
    # Non-EP uses one DP mesh over all workers with the same block-level wrapping.
    if use_ep and device_mesh is not None:
        reshard_after_forward = resolve_reshard_after_forward(
            train_config, default=False
        )
        logger.info(
            "Applying EP FSDP2 with reshard_after_forward=%s",
            reshard_after_forward,
        )
        model = apply_fsdp2(
            model,
            device_mesh["dp"],
            reshard_after_forward=reshard_after_forward,
        )
        write_memory_trace_event("after_apply_fsdp2", always=True)
        log_cuda_memory("after_apply_fsdp2", summary=True)
    elif use_fsdp2 and dp_mesh is not None:
        reshard_after_forward = resolve_reshard_after_forward(
            train_config, default=True
        )
        cpu_offload = resolve_fsdp_cpu_offload(train_config)
        logger.info(
            "Applying non-EP FSDP2 with reshard_after_forward=%s cpu_offload=%s",
            reshard_after_forward,
            cpu_offload,
        )
        model = apply_fsdp2(
            model,
            dp_mesh,
            reshard_after_forward=reshard_after_forward,
            cpu_offload=cpu_offload,
        )
        write_memory_trace_event("after_apply_fsdp2", always=True)
        log_cuda_memory("after_apply_fsdp2", summary=True)

    # ==== 5. Train and checkpoint ====
    # Manual-sharded saves own the FSDP2/EP export path; normal Trainer save logic
    # is only used for non-manual paths.
    data_collator = build_sft_data_collator(
        tokenizer, training_config.get("train_config", {})
    )
    manual_sharded_export_metadata = build_manual_sharded_export_metadata_from_config(
        training_config,
        processing_class=tokenizer,
    )

    trainer = LFMMoeSFTTrainer(
        ep_config=ep_config,
        manual_fsdp2=(use_ep or use_fsdp2),
        run_name_template=run_name_template,
        checkpoint_staging_dir=train_config.get("checkpoint_staging_dir"),
        manual_sharded_checkpoint_format=manual_sharded_checkpoint_format,
        manual_sharded_export_metadata=manual_sharded_export_metadata,
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.callback_handler.callbacks.insert(0, MoEMetricsCallback())
    trainer.add_callback(
        LiquidCheckpointCallback(
            run_name_template=run_name_template,
            manual_sharded=(use_ep or use_fsdp2),
        )
    )
    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_llm_benchmarks_from_config(benchmark_configs, tokenizer)
        if benchmarks:
            trainer.add_callback(BenchmarkEvalCallback(benchmarks))
    trainer = prepare_trainer(trainer)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    logger.info("Starting trainer.train() for MoE SFT")
    run_training_safely(trainer, resume_from_checkpoint=resume_from)
    logger.info("trainer.train() returned for MoE SFT")
    if (use_ep or use_fsdp2) and should_run_final_manual_sharded_save(
        trainer=trainer,
        requested_save_strategy=requested_save_strategy,
    ):
        logger.info("Running explicit final manual-sharded checkpoint save")
        save_manual_sharded_checkpoint(
            trainer=trainer,
            model=trainer.model,
            trial=None,
            checkpoint_format=manual_sharded_checkpoint_format,
            ep_group=ep_config["ep_group"] if ep_config is not None else None,
            export_metadata=trainer.get_manual_sharded_export_metadata(),
        )
        logger.info("Explicit final manual-sharded checkpoint save completed")
    logger.info("MoE SFT training completed successfully")

    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )
    finish_tracker(tracker)


def _validate_supported_moe_sft_config(moe_config: dict) -> None:
    capacity_factor = moe_config.get("capacity_factor")
    token_drop_policy = moe_config.get("token_drop_policy")

    if capacity_factor is None and token_drop_policy in (None, "probs"):
        return

    raise ValueError(
        "MoE SFT currently supports uncapped routing only. "
        "Remove capacity_factor/token_drop_policy from moe_training."
    )
