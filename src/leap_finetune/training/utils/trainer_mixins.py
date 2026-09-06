import logging
import os

import torch

from accelerate.utils import tqdm
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from trl.models import prepare_deepspeed

from leap_finetune.checkpointing.manual_sharded import (
    finalize_manual_sharded_export_metadata,
    load_manual_sharded_model_checkpoint,
    load_manual_sharded_optimizer_checkpoint,
    MANUAL_SHARDED_CHECKPOINT_FORMATS,
    normalize_manual_sharded_checkpoint_format,
    save_manual_sharded_checkpoint,
    save_manual_sharded_model_export,
)

logger = logging.getLogger(__name__)


class RayDataLoaderMixin:
    """Bypasses Accelerate's DistributedSampler for Ray-sharded data.

    Ray already shards data across workers via get_dataset_shard(), so we
    return plain DataLoaders to avoid double-sharding. Uses
    args.per_device_train_batch_size directly (NOT self._train_batch_size,
    which HF Trainer auto-multiplies by world_size).
    """

    def get_train_dataloader(self):
        dataloader_kwargs = {}

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
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
        )


class RayPrecomputedRefLogpsMixin:
    """Precompute reference log-probs on each already-sharded Ray dataset."""

    def _precompute_ref_logps(
        self, dataset: Dataset, name: str, batch_size: int
    ) -> Dataset:
        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "`precompute_ref_log_probs=True` is not supported with IterableDataset. "
                "Use a map-style dataset or disable precomputation."
            )

        data_seed = (
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
            generator=torch.Generator().manual_seed(data_seed),
        )

        if self.ref_model is None and self.is_deepspeed_enabled:
            if self._precompute_engine is None:
                self._precompute_engine = prepare_deepspeed(
                    self.model, self.accelerator
                )
            model = self._precompute_engine
        else:
            model = self.ref_model or self.model

        ref_chosen_logps = []
        ref_rejected_logps = []
        for padded_batch in tqdm(
            iterable=dataloader,
            desc=f"Computing reference log probs for local {name} dataset shard",
            disable=bool(os.environ.get("TQDM_DISABLE", "")),
        ):
            padded_batch = self._prepare_inputs(padded_batch)
            ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(
                model, padded_batch
            )
            ref_chosen_logps.append(ref_chosen_logp.cpu())
            ref_rejected_logps.append(ref_rejected_logp.cpu())

        chosen = torch.cat(ref_chosen_logps).float().numpy()
        rejected = torch.cat(ref_rejected_logps).float().numpy()
        dataset = dataset.add_column("ref_chosen_logps", chosen)
        return dataset.add_column("ref_rejected_logps", rejected)


def validate_manual_sharded_training_args(
    config_kwargs: dict,
    *,
    checkpoint_format: str | None = None,
) -> None:
    """Reject Trainer args that conflict with the manual-sharded runtime."""
    if config_kwargs.get("gradient_checkpointing"):
        raise ValueError(
            "gradient_checkpointing=True is not supported in manual-sharded EP/FSDP2 "
            "runs. This path already applies activation checkpointing in the FSDP2 "
            "wrapper; remove gradient_checkpointing from training_config."
        )
    checkpoint_format = checkpoint_format or config_kwargs.get(
        "manual_sharded_checkpoint_format"
    )
    if checkpoint_format is not None:
        try:
            normalize_manual_sharded_checkpoint_format(checkpoint_format)
        except ValueError as exc:
            raise ValueError(
                "manual_sharded_checkpoint_format must be one of "
                f"{sorted(MANUAL_SHARDED_CHECKPOINT_FORMATS)}"
            ) from exc


class ManualShardedCheckpointMixin:
    """Trainer overrides for manual-sharded MoE runs.

    This mixin does not implement sharding itself. It only routes Trainer save/load
    hooks to the repository's manual-sharded checkpoint format so EP and non-EP MoE
    runs share one checkpoint contract.
    """

    manual_sharded: bool
    ep_config: dict | None
    run_name_template: str | None = None
    checkpoint_staging_dir: str | None = None
    manual_sharded_checkpoint_format: str = "hf"
    manual_sharded_export_metadata: dict | None = None

    def get_manual_sharded_export_metadata(self) -> dict:
        return finalize_manual_sharded_export_metadata(
            self.manual_sharded_export_metadata,
            processing_class=getattr(self, "processing_class", None),
        )

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        if getattr(self, "manual_sharded", False):

            def _manual_prepare_model(
                model, device_placement=None, evaluation_mode=False
            ):
                del device_placement, evaluation_mode
                self.accelerator._models.append(model)
                return model

            self.accelerator.prepare_model = _manual_prepare_model

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if not getattr(self, "manual_sharded", False):
            return super().save_model(output_dir, _internal_call)

        output_dir = output_dir or self.args.output_dir
        mode_name = "EP" if self.ep_config is not None else "FSDP2"
        logger.info("%s trainer save_model start output_dir=%s", mode_name, output_dir)
        save_manual_sharded_model_export(
            model=self.model,
            accelerator=self.accelerator,
            output_dir=output_dir,
            processing_class=self.processing_class,
            data_collator=self.data_collator,
            training_args=self.args,
            ep_group=self.ep_config["ep_group"] if self.ep_config is not None else None,
            checkpoint_staging_dir=getattr(self, "checkpoint_staging_dir", None),
            export_metadata=self.get_manual_sharded_export_metadata(),
        )
        logger.info("%s trainer save_model end output_dir=%s", mode_name, output_dir)

    def _save_checkpoint(self, model, trial) -> None:
        if not getattr(self, "manual_sharded", False):
            return super()._save_checkpoint(model, trial)

        step = getattr(getattr(self, "state", None), "global_step", None)
        output_dir = getattr(getattr(self, "args", None), "output_dir", None)
        mode_name = "EP" if self.ep_config is not None else "FSDP2"
        logger.info(
            "%s trainer _save_checkpoint start step=%s output_dir=%s",
            mode_name,
            step,
            output_dir,
        )
        save_manual_sharded_checkpoint(
            trainer=self,
            model=model,
            trial=trial,
            checkpoint_format=getattr(self, "manual_sharded_checkpoint_format", "hf"),
            ep_group=self.ep_config["ep_group"] if self.ep_config is not None else None,
            export_metadata=self.get_manual_sharded_export_metadata(),
        )
        logger.info("%s trainer _save_checkpoint end step=%s", mode_name, step)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        if not getattr(self, "manual_sharded", False):
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

        model = model or self.model
        loaded = load_manual_sharded_model_checkpoint(
            model=model,
            checkpoint_dir=resume_from_checkpoint,
        )
        if not loaded:
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        if not getattr(self, "manual_sharded", False):
            return super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            return
        loaded = load_manual_sharded_optimizer_checkpoint(
            trainer=self,
            checkpoint_dir=checkpoint,
        )
        if not loaded:
            return super()._load_optimizer_and_scheduler(checkpoint)
