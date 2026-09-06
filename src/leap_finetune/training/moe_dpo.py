import logging
from typing import cast

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from leap_finetune.data_loading.length_grouping import get_length_grouped_sampler
from leap_finetune.distribution.distributed_configs import (
    resolve_fsdp_cpu_offload,
    resolve_reshard_after_forward,
)
from leap_finetune.training.utils.worker_setup import (
    default_eval_batch_size,
    resolve_train_eval_datasets,
    init_tracking_from_config,
    load_causal_lm_for_training,
)
from leap_finetune.checkpointing.callback import LeapCheckpointCallback
from leap_finetune.evaluation import (
    BenchmarkEvalCallback,
    create_llm_benchmarks_from_config,
)
from leap_finetune.training.moe_utils.metrics import MoEMetricsCallback
from leap_finetune.training.utils.logging import (
    finish_tracker,
    is_rank_zero,
)
from leap_finetune.training.utils.trainer_mixins import (
    ManualShardedCheckpointMixin,
    validate_manual_sharded_training_args,
    RayPrecomputedRefLogpsMixin,
)
from leap_finetune.training.utils.trainer_lifecycle import run_training_safely
from leap_finetune.training.utils.config_filter import filter_runtime_config_kwargs
from leap_finetune.checkpointing.manual_sharded import (
    build_manual_sharded_export_metadata_from_config,
    save_manual_sharded_checkpoint,
    should_run_final_manual_sharded_save,
)
from leap_finetune.training.moe_utils.losses import (
    MoETrainingConfig,
    apply_moe_losses,
)
from leap_finetune.training.moe_utils.ep_runtime import (
    apply_ep_to_model,
    apply_fsdp2,
    create_dp_mesh,
    create_ep_mesh,
    shard_experts,
)
from leap_finetune.training.peft.peft import (
    apply_peft_to_model,
    merge_and_save_peft_model,
)

logger = logging.getLogger(__name__)


MOE_DPO_EXCLUDED_KEYS = {
    "training_type",
    "wandb_logging",
    "leap_run_name_template",
    "moe_training",
    "model_config",
    "chat_template",
    "chat_template_path",
    "reshard_after_forward",
    "fsdp_cpu_offload",
    "checkpoint_staging_dir",
    "manual_sharded_checkpoint_format",
}


class LFMMoeDPOTrainer(
    RayPrecomputedRefLogpsMixin, ManualShardedCheckpointMixin, DPOTrainer
):
    """DPO Trainer for MoE models with EP/FSDP2 support."""

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

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

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
            return self.optimizer

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.AdamW(
            (param for param in self.model.parameters() if param.requires_grad),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=betas,
            fused=torch.cuda.is_available(),
        )
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self.ep_config:
            inputs = self._replicate_batch_across_ep(inputs)
        return super().training_step(model, inputs, num_items_in_batch, **kwargs)

    def _replicate_batch_across_ep(self, inputs: dict) -> dict:
        """Broadcast batch from EP rank 0 to all EP ranks in the group."""
        ep_group = self.ep_config["ep_group"]
        src_rank = dist.get_process_group_ranks(ep_group)[0]
        device = torch.cuda.current_device()
        is_src = dist.get_rank() == src_rank

        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(device)

            shape = torch.tensor(value.shape, device=device, dtype=torch.long)
            dist.broadcast(shape, src=src_rank, group=ep_group)
            if not is_src:
                value = torch.empty(
                    tuple(shape.tolist()), device=device, dtype=value.dtype
                )

            dist.broadcast(value, src=src_rank, group=ep_group)
            inputs[key] = value

        return inputs


def moe_dpo_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    """MoE DPO training loop with local and EP support."""
    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    moe_config_dict = training_config.get("train_config", {}).get("moe_training", {})
    moe_config = MoETrainingConfig.from_dict(moe_config_dict)
    ep_size = moe_config_dict.get("expert_parallel_size", 1) or 1
    train_config = training_config.get("train_config", {})
    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")
    run_name_template = train_config.get("leap_run_name_template")
    use_ep = ep_size > 1
    use_fsdp2 = peft_config is None and not use_ep

    reshard_after_forward = resolve_reshard_after_forward(
        train_config,
        default=False if use_ep else True,
    )
    cpu_offload = resolve_fsdp_cpu_offload(train_config)

    excluded_keys = MOE_DPO_EXCLUDED_KEYS
    if use_ep or use_fsdp2:
        excluded_keys = excluded_keys | {"deepspeed"}

    train_config_filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=excluded_keys,
        config_cls=DPOConfig,
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
        **train_config_filtered,
    }
    if use_ep or use_fsdp2:
        validate_manual_sharded_training_args(
            config_kwargs,
            checkpoint_format=manual_sharded_checkpoint_format,
        )
    if use_ep:
        logger.info(
            "EP mode: ep_size=%s, FSDP2 on dp_mesh, reshard_after_forward=%s",
            ep_size,
            reshard_after_forward,
        )
    elif use_fsdp2:
        logger.info(
            "Non-EP DPO mode: manual FSDP2 on full DP mesh "
            "reshard_after_forward=%s cpu_offload=%s",
            reshard_after_forward,
            cpu_offload,
        )

    training_args = DPOConfig(**config_kwargs)

    model, tokenizer = load_causal_lm_for_training(
        training_config,
        model_name=model_name,
        train_config=train_config,
    )

    ep_config = None
    device_mesh = None
    dp_mesh = None
    num_experts = getattr(model.config, "num_experts", None)

    if use_ep or use_fsdp2:
        logger.info("Waiting for all ranks to finish model loading...")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if use_ep:
        ep_config, device_mesh = create_ep_mesh(ep_size, num_experts)
        shard_experts(model, ep_config)
        apply_ep_to_model(model, ep_config, moe_config=moe_config)
    else:
        apply_moe_losses(model, moe_config)
        if use_fsdp2:
            dp_mesh = create_dp_mesh()

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    if use_ep and device_mesh is not None:
        # FSDP shards only over DP; EP ownership stays local to each EP rank.
        model = apply_fsdp2(
            model,
            device_mesh["dp"],
            reshard_after_forward=reshard_after_forward,
        )
    elif use_fsdp2 and dp_mesh is not None:
        model = apply_fsdp2(
            model,
            dp_mesh,
            reshard_after_forward=reshard_after_forward,
            cpu_offload=cpu_offload,
        )

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    manual_sharded_export_metadata = build_manual_sharded_export_metadata_from_config(
        training_config,
        processing_class=tokenizer,
    )

    trainer = LFMMoeDPOTrainer(
        ep_config=ep_config,
        manual_fsdp2=(use_ep or use_fsdp2),
        run_name_template=run_name_template,
        checkpoint_staging_dir=train_config.get("checkpoint_staging_dir"),
        manual_sharded_checkpoint_format=manual_sharded_checkpoint_format,
        manual_sharded_export_metadata=manual_sharded_export_metadata,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=cast(PreTrainedTokenizerBase, tokenizer),
    )

    trainer.add_callback(
        LeapCheckpointCallback(
            run_name_template=run_name_template,
            manual_sharded=(use_ep or use_fsdp2),
        )
    )
    trainer.add_callback(MoEMetricsCallback())
    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_llm_benchmarks_from_config(benchmark_configs, tokenizer)
        if benchmarks:
            trainer.add_callback(BenchmarkEvalCallback(benchmarks))
    trainer = prepare_trainer(trainer)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    run_training_safely(trainer, resume_from_checkpoint=resume_from)
    if (use_ep or use_fsdp2) and should_run_final_manual_sharded_save(
        trainer=trainer,
        requested_save_strategy=requested_save_strategy,
    ):
        save_manual_sharded_checkpoint(
            trainer=trainer,
            model=trainer.model,
            trial=None,
            checkpoint_format=manual_sharded_checkpoint_format,
            ep_group=ep_config["ep_group"] if ep_config is not None else None,
            export_metadata=trainer.get_manual_sharded_export_metadata(),
        )
    logger.info("MoE DPO training completed successfully")

    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )
    finish_tracker(tracker)
