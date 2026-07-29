from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from datasets import Dataset
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from leap_finetune.data_loading.dataset_loader import DatasetLoader
from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

TrainingType = Literal[
    "sft",
    "dpo",
    "kto",
    "embedding",
    "colbert",
    "vlm_sft",
    "vlm_dpo",
    "moe_sft",
    "moe_dpo",
    "grpo",
    "vlm_grpo",
]

DatasetType = Literal[
    "sft",
    "dpo",
    "kto",
    "embedding",
    "colbert",
    "vlm_sft",
    "vlm_dpo",
    "grpo",
    "vlm_grpo",
]


EmbeddingLoss = Literal[
    "multiple_negatives_ranking",
    "cached_multiple_negatives_ranking",
]
ColBERTLoss = Literal["contrastive", "cached_contrastive"]
RetrievalLoss = EmbeddingLoss | ColBERTLoss


class RetrievalPrompts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = "query: "
    positive: str = "document: "
    negative: str = "document: "


QATType = Literal[
    "gguf_q4_0",
    "gguf_q8_0",
    "mlx_q4",
    "mlx_q8",
    "vllm_fp8",
    "vllm_mxfp4",
    "noise_q4",
    "noise_q8",
]


class QATConfig(BaseModel):
    """Quantization-aware training configuration."""

    model_config = ConfigDict(extra="forbid")

    type: QATType
    quantize_reference: bool = True
    target: Literal["auto", "cuda", "rocm_mi300"] | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> QATConfig:
        if self.target is not None and self.type != "vllm_fp8":
            raise ValueError("training_config.qat.target is only valid for vllm_fp8")
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    train_path: str | None = None
    val_path: str | None = None
    type: DatasetType
    limit: int | None = None
    split: str = "train"
    train_split: str | None = None
    val_split: str | None = None
    subset: str | None = None
    train_subset: str | None = None
    val_subset: str | None = None
    test_size: float | None = None
    image_root: str | None = None
    cache_dataset: bool = False
    hf_streaming_batch_size: int = 10000

    @model_validator(mode="after")
    def _validate_dataset(self) -> DatasetConfig:
        if self.path and self.train_path:
            raise ValueError("Use either dataset.path or dataset.train_path, not both")
        if not self.path and not self.train_path:
            raise ValueError("dataset.path or dataset.train_path is required")
        if self.test_size is not None and not (0 < self.test_size < 1):
            raise ValueError(
                f"dataset.test_size must be between 0 and 1 (exclusive), got {self.test_size}"
            )
        if self.test_size is not None and (
            self.val_path is not None or self.val_split is not None
        ):
            raise ValueError(
                "dataset.test_size cannot be combined with dataset.val_path or dataset.val_split"
            )
        return self

    def has_eval_dataset(self) -> bool:
        return (
            self.test_size is not None
            or self.val_path is not None
            or self.val_split is not None
        )


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    extends: str | None = None
    base: str | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    num_train_epochs: float | None = None
    per_device_train_batch_size: int | None = None
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    eval_strategy: str | None = None
    eval_steps: int | None = None
    save_strategy: str | None = None
    save_steps: int | None = None
    logging_steps: int | None = None
    bf16: bool | None = None
    gradient_checkpointing: bool | None = None
    output_dir: str | None = None
    resume_from_checkpoint: str | None = None
    chat_template_path: str | None = None
    adapter_path: str | None = None
    completion_only_loss: bool | None = None
    loss: RetrievalLoss | None = None
    prompts: RetrievalPrompts | None = None
    temperature: float | None = Field(default=None, gt=0)
    mini_batch_size: int | None = Field(default=None, gt=0)
    gather_across_devices: bool | None = None
    qat: QATConfig | None = None

    def extends_name(self) -> str | None:
        return self.extends or self.base

    def override_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude={"extends", "base"})


class PeftConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    extends: str | None = None
    base: str | None = None
    use_peft: bool | None = None

    def extends_name(self) -> str | None:
        return self.extends or self.base

    def override_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude={"extends", "base"})


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    path: str
    metric: str
    max_new_tokens: int | None = None
    image_root: str | None = None


class EvalSuiteConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_new_tokens: int | None = None
    image_root: str | None = None
    best_checkpoint_metrics: dict[str, float] | None = None
    benchmarks: list[EvalConfig] = Field(default_factory=list)


class EvalBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["hf", "vllm", "llama_cpp"] = "hf"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    max_model_len: int | None = None
    quantization: str | None = None
    base_url: str | None = None
    model_id: str | None = None
    server_binary: str = "llama-server"
    host: str = "127.0.0.1"
    port: int = 8080
    n_gpu_layers: int = 999
    mmproj: str | None = None
    server_args: list[str] = Field(default_factory=list)
    startup_timeout: float = 300.0
    request_timeout: float = 600.0
    log_path: str | None = None

    @model_validator(mode="after")
    def _validate_backend(self) -> EvalBackendConfig:
        if self.tensor_parallel_size < 1:
            raise ValueError("backend.tensor_parallel_size must be >= 1")
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError("backend.gpu_memory_utilization must be in (0, 1]")
        if not 1 <= self.port <= 65535:
            raise ValueError("backend.port must be in [1, 65535]")
        if self.startup_timeout <= 0:
            raise ValueError("backend.startup_timeout must be > 0")
        if self.request_timeout <= 0:
            raise ValueError("backend.request_timeout must be > 0")
        return self


class EvalRunConfig(BaseModel):
    """Standalone eval config.

    Uses the same ``evals:`` suite as training configs, but deliberately has
    no dataset or training section. Backend placement/configuration is kept
    outside the eval suite so benchmark definitions remain reusable.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    project_name: str | None = None
    model_name: str | None = None
    checkpoint: str | None = None
    modality: Literal["text", "vlm"] = "text"
    evals: EvalSuiteConfig = Field(
        validation_alias=AliasChoices("evals", "benchmarks"),
        serialization_alias="evals",
    )
    backend: EvalBackendConfig = Field(default_factory=EvalBackendConfig)
    model_overrides: dict[str, Any] | None = Field(default=None, alias="model_config")
    output_path: str | None = None
    config_dir: str | None = None

    @model_validator(mode="after")
    def _validate_eval_run(self) -> EvalRunConfig:
        if not self.model_name and not self.checkpoint:
            raise ValueError("Standalone eval requires model_name or checkpoint")
        return self

    @property
    def model_ref(self) -> str:
        return self.checkpoint or self.model_name or ""


class RayConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    address: str | None = None
    num_workers: int | None = None
    resources_per_worker: dict[str, Any] | None = None


class SlurmConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    nodes: int | None = None
    ntasks_per_node: int | None = None
    gpus_per_task: int | None = None
    gpus_per_node: int | None = None
    cpus_per_gpu: int | None = None
    directives: list[str] = Field(default_factory=list)
    setup_commands: list[str] = Field(default_factory=list)


class ModalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class KubeRayConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class JobConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    project_name: str | None = None
    job_name: str | None = None
    model_name: str = "LFM2-1.2B"
    training_type: TrainingType = "sft"
    dataset: DatasetConfig
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    peft_config: PeftConfig | None = None
    evals: EvalSuiteConfig | None = Field(
        default=None,
        validation_alias=AliasChoices("evals", "benchmarks"),
        serialization_alias="evals",
    )
    model_overrides: dict[str, Any] | None = Field(default=None, alias="model_config")
    ray: RayConfig | None = None
    slurm: SlurmConfig | None = None
    modal: ModalConfig | None = None
    kuberay: KubeRayConfig | None = None
    rewards: list[Any] | dict[str, Any] | None = None
    rl_env: dict[str, Any] | None = None
    grpo_rollout: dict[str, Any] | None = None
    async_eval: AsyncEvalConfig | None = None
    config_dir: str | None = None

    @model_validator(mode="after")
    def _validate_job(self) -> JobConfig:
        resolved_job_name = self.resolved_job_name
        if not all(c.isalnum() or c in "-_" for c in resolved_job_name):
            raise ValueError(
                f"Invalid job name '{resolved_job_name}': only letters, numbers, hyphens, and underscores allowed"
            )

        retrieval_types = {"embedding", "colbert"}
        if (
            self.training_type in retrieval_types
            or self.dataset.type in retrieval_types
        ) and self.training_type != self.dataset.type:
            raise ValueError(
                "Retrieval jobs require matching training_type and dataset.type; "
                f"got {self.training_type!r} and {self.dataset.type!r}."
            )
        if (
            self.training_type in retrieval_types
            and self.peft_config is not None
            and self.peft_config.use_peft is not False
        ):
            raise ValueError("PEFT is not yet supported for retrieval training")
        if self.training_type in retrieval_types:
            train_config = self.training_config
            expected_base = (
                "DEFAULT_EMBEDDING"
                if self.training_type == "embedding"
                else "DEFAULT_COLBERT"
            )
            if (
                train_config.extends_name() is not None
                and train_config.extends_name() != expected_base
            ):
                raise ValueError(
                    f"{self.training_type} training requires "
                    f"training_config.extends={expected_base!r}"
                )
            if train_config.gradient_accumulation_steps not in (None, 1):
                raise ValueError(
                    "Retrieval contrastive losses require "
                    "gradient_accumulation_steps=1; use a cached loss and "
                    "mini_batch_size for larger effective batches."
                )
            if (
                train_config.per_device_train_batch_size is not None
                and train_config.per_device_train_batch_size < 1
            ):
                raise ValueError(
                    "per_device_train_batch_size must be positive for retrieval"
                )

            if self.training_type == "embedding":
                allowed_losses = {
                    "multiple_negatives_ranking",
                    "cached_multiple_negatives_ranking",
                }
                effective_loss = train_config.loss or "multiple_negatives_ranking"
                if train_config.temperature is not None:
                    raise ValueError(
                        "training_config.temperature is only valid for ColBERT"
                    )
            else:
                allowed_losses = {"contrastive", "cached_contrastive"}
                effective_loss = train_config.loss or "contrastive"
                if train_config.prompts is not None:
                    raise ValueError(
                        "training_config.prompts is only valid for embedding"
                    )

            if effective_loss not in allowed_losses:
                raise ValueError(
                    f"Unsupported {self.training_type} loss: {effective_loss}"
                )
            if (
                train_config.mini_batch_size is not None
                and not effective_loss.startswith("cached_")
            ):
                raise ValueError(
                    "training_config.mini_batch_size requires a cached retrieval loss"
                )

        if self.training_type not in ("grpo", "vlm_grpo"):
            for key in ("rewards", "rl_env", "grpo_rollout"):
                if getattr(self, key) is not None:
                    raise ValueError(
                        f"Config key `{key}` is only valid for training_type in "
                        f"('grpo', 'vlm_grpo'); got training_type={self.training_type!r}."
                    )
        qat = self.training_config.qat
        if (
            qat is not None
            and not qat.quantize_reference
            and self.training_type not in ("dpo", "vlm_dpo", "moe_dpo")
        ):
            raise ValueError(
                "training_config.qat.quantize_reference is only valid for DPO"
            )
        if (
            qat is not None
            and self.training_type in ("grpo", "vlm_grpo")
            and bool(getattr(self.training_config, "use_vllm", False))
        ):
            raise ValueError(
                "QAT GRPO requires training_config.use_vllm: false so rollout "
                "generation and training forwards use the same fake-quantized model."
            )
        return self

    @property
    def resolved_job_name(self) -> str:
        return self.project_name or self.job_name or "default_job"

    @property
    def benchmarks(self) -> EvalSuiteConfig | None:
        return self.evals


BenchmarkConfig = EvalConfig
BenchmarkSuiteConfig = EvalSuiteConfig


class _ResolvedConfigValue:
    def __init__(self, value: Any):
        self.value = value


@dataclass
class ResolvedJobConfig:
    job_name: str
    model_name: str
    training_type: TrainingType
    dataset: DatasetLoader | tuple[Dataset, Dataset] | None
    training_config: Any
    peft_config: Any | None
    benchmark_configs: dict[str, Any] | None
    model_config: dict[str, Any] | None
    ray_config: dict[str, Any] | None
    rewards: list[Any] | dict[str, Any] | None
    rl_env: dict[str, Any] | None
    grpo_rollout: dict[str, Any] | None
    async_eval: dict[str, Any] | None
    config_dir: str | None = None

    def to_dict(self, dataset: tuple[Dataset, Dataset] | None = None) -> dict[str, Any]:
        dataset_to_use = dataset if dataset is not None else self.dataset
        return {
            "model_name": self.model_name,
            "job_name": self.job_name,
            "training_type": self.training_type,
            "training_config": self.training_config.value,
            "dataset": dataset_to_use,
            "peft_config": self.peft_config.value if self.peft_config else None,
            "benchmark_configs": self.benchmark_configs,
            "model_config": self.model_config,
            "ray_config": self.ray_config,
            "rewards": self.rewards,
            "rl_env": self.rl_env,
            "grpo_rollout": self.grpo_rollout,
            "async_eval": self.async_eval,
            "config_dir": self.config_dir,
        }
