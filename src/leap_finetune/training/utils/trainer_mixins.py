import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from leap_finetune.checkpointing.manual_sharded import (
    MANUAL_SHARDED_CHECKPOINT_FORMATS,
    finalize_manual_sharded_export_metadata,
    load_manual_sharded_model_checkpoint,
    load_manual_sharded_optimizer_checkpoint,
    normalize_manual_sharded_checkpoint_format,
    save_manual_sharded_checkpoint,
    save_manual_sharded_model_export,
)
from leap_finetune.data_loading.length_grouping import (
    get_length_grouped_sampler,
    get_tile_count_grouped_sampler,
)
from leap_finetune.loss_weighting.loss import (
    weighted_causal_lm_loss,
)

logger = logging.getLogger(__name__)


class RayDataLoaderMixin:
    """Bypasses Accelerate's DistributedSampler for Ray-sharded data.

    Ray already shards data across workers via get_dataset_shard(), so we
    return plain DataLoaders to avoid double-sharding. Uses
    args.per_device_train_batch_size directly (NOT self._train_batch_size,
    which HF Trainer auto-multiplies by world_size).
    """

    def _ray_dataloader_kwargs(self, *, training: bool) -> dict:
        """Translate HF dataloader arguments for the Ray-sharded loader.

        Ray owns distributed sharding, so this only configures local PyTorch
        loading. Worker-only options must not be passed with ``num_workers=0``.
        """
        num_workers = int(getattr(self.args, "dataloader_num_workers", 0) or 0)
        if num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative")

        kwargs = {
            "num_workers": num_workers,
            "pin_memory": bool(getattr(self.args, "dataloader_pin_memory", True)),
        }
        if training:
            kwargs["drop_last"] = bool(
                getattr(self.args, "dataloader_drop_last", False)
            )

        if num_workers > 0:
            prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", None)
            if prefetch_factor is not None:
                prefetch_factor = int(prefetch_factor)
                if prefetch_factor < 1:
                    raise ValueError("dataloader_prefetch_factor must be positive")
                kwargs["prefetch_factor"] = prefetch_factor
            kwargs["persistent_workers"] = bool(
                getattr(self.args, "dataloader_persistent_workers", False)
            )

        return kwargs

    def get_train_dataloader(self):
        batch_size = self.args.per_device_train_batch_size
        sampler = None
        dataloader_kwargs = self._ray_dataloader_kwargs(training=True)
        group_generator = None
        if getattr(self.args, "seed", None) is not None:
            group_generator = torch.Generator().manual_seed(int(self.args.seed))

        if getattr(self, "group_by_image_tiles", False):
            sampler = get_tile_count_grouped_sampler(
                self.train_dataset,
                batch_size,
                generator=group_generator,
            )

        if sampler is None and getattr(self.args, "group_by_length", False):
            sampler = get_length_grouped_sampler(
                self.train_dataset,
                batch_size,
                generator=group_generator,
            )
        if sampler is None and group_generator is not None:
            dataloader_kwargs["generator"] = group_generator

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            shuffle=sampler is None,
            sampler=sampler,
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
            **self._ray_dataloader_kwargs(training=False),
        )


class CausalLMLossTokenCountMixin:
    """Align Trainer token counts with decoder-only causal-LM loss targets.

    Transformers 5.3 counts non-ignored labels before the model shifts them,
    while the causal loss in this repository scores labels[..., 1:].
    Passing shifted labels through to the upstream helper preserves its
    distributed gathering and device handling while keeping the denominator
    aligned with the scored tokens.
    """

    def _get_num_items_in_batch(self, batch_samples, device):
        if not batch_samples or "labels" not in batch_samples[0]:
            return super()._get_num_items_in_batch(batch_samples, device)

        shifted_samples = []
        for batch in batch_samples:
            shifted_batch = dict(batch)
            shifted_batch["labels"] = batch["labels"][..., 1:]
            shifted_samples.append(shifted_batch)
        return super()._get_num_items_in_batch(shifted_samples, device)


class TokenWeightedLossMixin:
    """Compute weighted causal CE and gather its denominator across workers."""

    def __init__(self, *args, token_weighting=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._leap_token_weighting = token_weighting or None
        self._leap_weight_diagnostics = {}
        if self._leap_token_weighting:
            # Trainer must not divide each microbatch by gradient accumulation;
            # our denominator already covers the complete accumulated batch.
            self.model_accepts_loss_kwargs = True

    def _loss_weight_rule_names(self):
        config = self._leap_token_weighting
        if not isinstance(config, dict):
            return []
        return [rule["name"] for rule in config.get("rules", [])]

    @staticmethod
    def _diagnostic_state_template():
        return {
            "rows": 0.0,
            "tokens": 0.0,
            "weight": 0.0,
            "unweighted_numerator": 0.0,
            "unweighted_denominator": 0.0,
            "rule_rows": 0.0,
            "rule_matches": {},
        }

    def _diagnostic_state(self, phase):
        return self._leap_weight_diagnostics.setdefault(
            phase, self._diagnostic_state_template()
        )

    def _get_num_items_in_batch(self, batch_samples, device):
        if not batch_samples or "loss_weights" not in batch_samples[0]:
            return super()._get_num_items_in_batch(batch_samples, device)

        effective_weight = torch.zeros((), dtype=torch.float32, device=device)
        for batch in batch_samples:
            labels = batch["labels"][..., 1:].to(device)
            weights = batch["loss_weights"][..., 1:].to(device)
            effective_weight = effective_weight + (weights * labels.ne(-100)).sum()

        if getattr(self.args, "average_tokens_across_devices", False):
            effective_weight = self.accelerator.gather(effective_weight).sum()
        return effective_weight

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        row_diagnostics = inputs.pop("_loss_weight_diagnostics", None)
        loss_weights = inputs.pop("loss_weights", None)
        if loss_weights is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = weighted_causal_lm_loss(
            logits,
            labels,
            loss_weights,
            num_items_in_batch=num_items_in_batch,
        )
        self._record_loss_weight_diagnostics(
            model, logits, labels, loss_weights, row_diagnostics
        )
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            # DDP averages gradients. Match Trainer.compute_loss by restoring
            # the process factor after normalizing by the global denominator.
            process_count = (
                self.args.n_gpu
                if self.args.n_gpu > 1
                else self.accelerator.num_processes
            )
            loss = loss * process_count
        return (loss, outputs) if return_outputs else loss

    def _record_loss_weight_diagnostics(
        self, model, logits, labels, loss_weights, row_diagnostics
    ):
        phase = "train" if getattr(model, "training", True) else "eval"
        state = self._diagnostic_state(phase)
        shifted_labels = labels[..., 1:]
        shifted_weights = loss_weights[..., 1:].to(
            device=shifted_labels.device, dtype=torch.float32
        )
        valid = shifted_labels.ne(-100)
        effective = shifted_weights * valid
        state["rows"] += float(labels.shape[0])
        state["tokens"] += float(((shifted_weights > 0) & valid).sum().item())
        state["weight"] += float(effective.sum().item())

        if phase == "eval":
            token_loss = F.cross_entropy(
                logits[..., :-1, :].float().reshape(-1, logits.size(-1)),
                shifted_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shifted_labels)
            state["unweighted_numerator"] += float((token_loss * valid).sum().item())
            state["unweighted_denominator"] += float(valid.sum().item())

        if row_diagnostics:
            state["rule_rows"] += float(len(row_diagnostics))
            matches = state["rule_matches"]
            for diagnostics in row_diagnostics:
                for name, matched in diagnostics.get("rule_matches", {}).items():
                    matches[name] = matches.get(name, 0.0) + float(bool(matched))

    def log(self, logs, *args, **kwargs):
        phase = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        state = self._leap_weight_diagnostics.get(phase)
        if state and state["rows"]:
            rule_names = self._loss_weight_rule_names()
            values = [
                state["rows"],
                state["tokens"],
                state["weight"],
                state["unweighted_numerator"],
                state["unweighted_denominator"],
                state["rule_rows"],
                *(state["rule_matches"].get(name, 0.0) for name in rule_names),
            ]
            totals = torch.tensor(values, dtype=torch.float64, device=self.args.device)
            if self.accelerator.num_processes > 1:
                totals = self.accelerator.reduce(totals, reduction="sum")
            totals = totals.cpu().tolist()
            rows, tokens, weight, unweighted_num, unweighted_den, rule_rows = totals[:6]
            prefix = f"{phase}_loss_weight"
            logs[f"{prefix}/weighted_tokens"] = tokens
            logs[f"{prefix}/effective_weight_sum"] = weight
            logs[f"{prefix}/rows"] = rows
            if phase == "eval" and unweighted_den:
                logs["eval_unweighted_loss"] = unweighted_num / unweighted_den
            if rule_rows:
                for name, matched in zip(rule_names, totals[6:], strict=True):
                    logs[f"{prefix}/{name}_matched_rows"] = matched
                    logs[f"{prefix}/{name}_unmatched_rows"] = rule_rows - matched
            self._leap_weight_diagnostics[phase] = self._diagnostic_state_template()
        return super().log(logs, *args, **kwargs)


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
