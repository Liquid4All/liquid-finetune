import json
import os
from pathlib import Path
from typing import Any

from ray import train
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from liquid_finetune.training.utils.logging import is_rank_zero

from liquid_finetune.checkpointing.paths import (
    current_checkpoint_output_dir,
    rename_standard_checkpoint,
)

LIQUID_RAY_FINAL_METRICS_FILE = ".liquid_ray_final_metrics.json"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def persist_rank_zero_metrics(output_dir: str | os.PathLike, metrics: dict) -> None:
    """Persist final metrics for Ray Train results that omit metrics."""
    try:
        if train.get_context().get_world_rank() != 0:
            return
    except Exception:
        return

    metrics_path = Path(output_dir) / LIQUID_RAY_FINAL_METRICS_FILE
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metrics_path.with_suffix(f"{metrics_path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(metrics, default=_json_default, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, metrics_path)


def hydrate_missing_ray_metrics(result, output_dir: str):
    """Fill Ray Train results from callback metrics when Ray omits them."""
    if result is None or getattr(result, "metrics", None) is not None:
        return result

    metrics_path = Path(output_dir) / LIQUID_RAY_FINAL_METRICS_FILE
    if not metrics_path.exists():
        return result

    with metrics_path.open(encoding="utf-8") as f:
        result.metrics = json.load(f)

    return result


def report_ray_metrics(metrics: dict, output_dir: str | os.PathLike) -> None:
    persist_rank_zero_metrics(output_dir, metrics)
    try:
        train.get_context()
    except RuntimeError:
        return
    train.report(metrics=metrics, checkpoint=None)


class LiquidCheckpointCallback(TrainerCallback):
    """Integrates HuggingFace Trainer with Ray Train checkpointing and
    renames checkpoint dirs to descriptive names on every save.

    Renaming happens immediately in on_save (not on_train_end) so that
    checkpoints are properly named even if training crashes mid-run.

    Only rank 0 performs filesystem operations to avoid race conditions.

    Args:
        run_name_template: Template for checkpoint names, e.g.
            "LFM2-1.2B-sft-smoltalk-1000-lr2e005-w0p2-lora_a-20250217_143022"
    """

    def __init__(
        self,
        run_name_template: str | None = None,
        *,
        manual_sharded: bool = False,
    ) -> None:
        super().__init__()
        self.metrics: dict = {}
        self.loss_history: list[float] = []
        self.reward_history: list[float] = []
        self.reward_std_history: list[float] = []
        self.grad_norm_history: list[float] = []
        self.learning_rate_history: list[float] = []
        self.run_name_template = run_name_template
        self.manual_sharded = manual_sharded

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs:
            self.metrics.update(logs)
            if "loss" in logs:
                self.loss_history.append(logs["loss"])
            for key, history in (
                ("reward", self.reward_history),
                ("reward_std", self.reward_std_history),
                ("grad_norm", self.grad_norm_history),
                ("learning_rate", self.learning_rate_history),
            ):
                value = logs.get(key)
                if isinstance(value, (int, float)):
                    history.append(float(value))

    def _optimization_history_metrics(self) -> dict:
        """Expose optimizer/reward histories for end-to-end learning checks."""
        return {
            key: values.copy()
            for key, values in (
                ("reward_history", self.reward_history),
                ("reward_std_history", self.reward_std_history),
                ("grad_norm_history", self.grad_norm_history),
                ("learning_rate_history", self.learning_rate_history),
            )
            if values
        }

    @staticmethod
    def _latest_benchmark_metrics(state: TrainerState) -> dict:
        for entry in reversed(state.log_history):
            if any(key.startswith("benchmark/") for key in entry):
                return {
                    key: value
                    for key, value in entry.items()
                    if key.startswith("benchmark/")
                }
        return {}

    @staticmethod
    def _retrieval_improvement_metrics(state: TrainerState) -> dict:
        history: dict[str, list[float]] = {}
        for entry in state.log_history:
            for key, value in entry.items():
                if "retrieval" not in key or not isinstance(value, (int, float)):
                    continue
                history.setdefault(key, []).append(float(value))

        metrics = {}
        for key, values in history.items():
            if len(values) < 2:
                continue
            name = key.removeprefix("eval_")
            metrics[f"retrieval/baseline/{name}"] = values[0]
            metrics[f"retrieval/final/{name}"] = values[-1]
            metrics[f"retrieval/delta/{name}"] = values[-1] - values[0]
        return metrics

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        checkpoint_path = current_checkpoint_output_dir(
            output_dir=args.output_dir,
            run_name_template=self.run_name_template,
            epoch=state.epoch,
            step=state.global_step,
            manual_sharded=self.manual_sharded,
        )
        if is_rank_zero():
            print(f"Checkpoint saved: {checkpoint_path}")
            print(f"   Metrics: {self.metrics}")

            if not self.manual_sharded:
                rename_standard_checkpoint(
                    output_dir=args.output_dir,
                    run_name_template=self.run_name_template,
                    epoch=state.epoch,
                    step=state.global_step,
                    save_total_limit=args.save_total_limit,
                )

        # Include loss curve summary for test assertions
        metrics = self.metrics.copy()
        metrics.update(self._latest_benchmark_metrics(state))
        metrics.update(self._optimization_history_metrics())
        if self.loss_history:
            metrics["loss_history"] = self.loss_history.copy()

        # Report metrics only; HF Trainer already saved checkpoint to output_dir.
        # Passing checkpoint=None avoids Ray duplicating files into ray_logs/.
        report_ray_metrics(metrics, args.output_dir)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.metrics:
            metrics = self.metrics.copy()
            metrics.update(self._latest_benchmark_metrics(state))
            metrics.update(self._retrieval_improvement_metrics(state))
            metrics.update(self._optimization_history_metrics())
            if self.loss_history:
                metrics["loss_history"] = self.loss_history.copy()
            report_ray_metrics(metrics, args.output_dir)
