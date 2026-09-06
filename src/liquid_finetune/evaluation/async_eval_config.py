from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ==== Async Eval Config ====

VALID_MODES = ("sync", "sidecar", "reserved")
VALID_OVERLAP = ("skip", "queue")
VALID_WEIGHT_RELOAD = ("in_place", "respawn")


class SbatchConfig(BaseModel):
    """SLURM submission settings for ``mode: sidecar``.

    ``partition``/``account`` default to inheriting from the parent
    training job. ``time=None`` lets sbatch use the partition's default
    time limit; set explicitly to cap runaway evals.
    """

    model_config = ConfigDict(extra="forbid")

    partition: str | None = None
    account: str | None = None
    time: str | None = None
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("time", mode="before")
    @classmethod
    def _coerce_time(cls, value):
        return str(value) if value else None

    @field_validator("extra_args", mode="before")
    @classmethod
    def _none_extra_args_to_empty(cls, value):
        return [] if value is None else value


class ReservedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight_reload: Literal["in_place", "respawn"] = "respawn"
    server_port: int = 8100


class FailureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_consecutive: int = 3
    max_submit_attempts: int = 3
    # Seconds between sbatch retries; exponential ``backoff * 2**(attempt-1)``.
    # ``0.0`` is allowed (immediate burst) but only safe on quiet controllers.
    submit_retry_backoff: float = 2.0


class AsyncEvalConfig(BaseModel):
    """Resolved ``async_eval:`` YAML block."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["sync", "sidecar", "reserved"] = "sync"

    # Common (used by sidecar AND reserved; ignored when mode=sync)
    vllm_gpus: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    max_model_len: int | None = None

    sbatch: SbatchConfig = Field(default_factory=SbatchConfig)
    reserved: ReservedConfig = Field(default_factory=ReservedConfig)
    failure: FailureConfig = Field(default_factory=FailureConfig)

    on_overlap: Literal["skip", "queue"] = "skip"

    @model_validator(mode="after")
    def _validate_async_eval(self) -> "AsyncEvalConfig":
        if self.mode != "sync" and self.vllm_gpus < 1:
            raise ValueError(
                f"async_eval.vllm_gpus must be >= 1 for mode={self.mode!r}, "
                f"got {self.vllm_gpus}"
            )
        if self.tensor_parallel_size > self.vllm_gpus:
            raise ValueError(
                "async_eval.tensor_parallel_size "
                f"({self.tensor_parallel_size}) cannot exceed vllm_gpus "
                f"({self.vllm_gpus})"
            )
        if self.tensor_parallel_size < 1:
            raise ValueError(
                "async_eval.tensor_parallel_size must be >= 1, got "
                f"{self.tensor_parallel_size}"
            )
        if self.failure.max_consecutive < 1:
            raise ValueError("async_eval.failure.max_consecutive must be >= 1")
        if self.failure.max_submit_attempts < 1:
            raise ValueError("async_eval.failure.max_submit_attempts must be >= 1")
        if self.failure.submit_retry_backoff < 0:
            raise ValueError("async_eval.failure.submit_retry_backoff must be >= 0")
        return self

    @classmethod
    def from_dict(cls, raw: dict | "AsyncEvalConfig" | None) -> "AsyncEvalConfig":
        """Parse + validate. ``None`` or empty dict returns the sync default."""
        if isinstance(raw, cls):
            return raw
        if not raw:
            return cls(mode="sync")
        mode = raw.get("mode", "sync")
        if mode not in VALID_MODES:
            raise ValueError(
                f"async_eval.mode must be one of {VALID_MODES}, got {mode!r}"
            )
        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for ``train_loop_config`` (Ray workers)."""
        return self.model_dump()


def make_eval_callback(
    benchmarks: list,
    async_eval_cfg: dict | None,
    *,
    benchmark_configs: dict | None = None,
    server_url: str | None = None,
    eval_gpu_ids: str = "",
    output_dir: str | None = None,
    wandb_run_id: str | None = None,
    config_dir: str | None = None,
):
    """Return the right ``TrainerCallback`` for the configured mode.

    Mode-specific kwargs:
    - sidecar: ``benchmark_configs`` (re-constructs benchmarks inside the
      eval job), ``output_dir`` (marker + checkpoint staging),
      ``wandb_run_id`` (attach to the training run), ``config_dir``.
    - reserved: ``server_url`` (set by the driver after launching the
      vLLM server), ``eval_gpu_ids``.
    """
    cfg = AsyncEvalConfig.from_dict(async_eval_cfg)

    if cfg.mode == "sync":
        # Lazy import so consumers that don't need sync don't pay.
        from liquid_finetune.evaluation.callback import BenchmarkEvalCallback

        return BenchmarkEvalCallback(benchmarks)

    if cfg.mode == "sidecar":
        from liquid_finetune.evaluation.sidecar_callback import SidecarEvalCallback

        return SidecarEvalCallback(
            benchmarks=benchmarks,
            cfg=cfg,
            benchmark_configs=benchmark_configs,
            output_dir=output_dir,
            wandb_run_id=wandb_run_id,
            config_dir=config_dir,
        )

    if cfg.mode == "reserved":
        from liquid_finetune.evaluation.reserved_callback import ReservedEvalCallback

        if not server_url:
            raise RuntimeError(
                "async_eval mode=reserved requires a vLLM server_url; the "
                "driver must launch the server before constructing the callback."
            )
        return ReservedEvalCallback(
            benchmarks=benchmarks,
            cfg=cfg,
            server_url=server_url,
            output_dir=output_dir,
            eval_gpu_ids=eval_gpu_ids,
        )

    raise ValueError(f"Unknown async_eval.mode={cfg.mode!r}")
