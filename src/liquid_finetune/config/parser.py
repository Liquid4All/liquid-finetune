from __future__ import annotations

import logging
import os
import pathlib
from datetime import datetime
from typing import Any

import yaml
from peft import LoraConfig
from pydantic import ValidationError

from liquid_finetune import LIQUID_FINETUNE_DIR
from liquid_finetune.checkpointing.model_info import is_moe_model_from_name
from liquid_finetune.config.job_config import (
    DatasetConfig,
    EvalRunConfig,
    EvalSuiteConfig,
    JobConfig,
    ResolvedJobConfig,
    _ResolvedConfigValue,
)
from liquid_finetune.data_loading.dataset_loader import DatasetLoader
from liquid_finetune.training.default_configs import PEFT_DEFAULTS, TRAINING_DEFAULTS

logger = logging.getLogger(__name__)

_REWARD_SEP = "::"
_REWARDS_DIR = LIQUID_FINETUNE_DIR / "rewards"

TRAINING_TYPE_TO_CONFIG = {
    "sft": "DEFAULT_SFT",
    "dpo": "DEFAULT_DPO",
    "kto": "DEFAULT_KTO",
    "embedding": "DEFAULT_EMBEDDING",
    "colbert": "DEFAULT_COLBERT",
    "vlm_sft": "DEFAULT_VLM_SFT",
    "vlm_dpo": "DEFAULT_VLM_DPO",
    "moe_sft": "MOE_SFT",
    "moe_dpo": "MOE_DPO",
    "grpo": "DEFAULT_GRPO",
    "vlm_grpo": "DEFAULT_VLM_GRPO",
}
DATASET_TYPE_ALIASES = {
    "moe_sft": "sft",
    "moe_dpo": "dpo",
}
VALID_DATASET_TYPES = {
    "sft",
    "dpo",
    "kto",
    "embedding",
    "colbert",
    "vlm_sft",
    "vlm_dpo",
    "grpo",
    "vlm_grpo",
}


def resolve_config_path(config_input: str | pathlib.Path) -> pathlib.Path:
    input_path = pathlib.Path(config_input)
    candidates = [str(input_path)]
    if not input_path.suffix:
        candidates.append(f"{input_path}.yaml")

    for candidate in candidates:
        candidate_path = pathlib.Path(candidate)
        if candidate_path.exists():
            return candidate_path.resolve()

        local_job_config = pathlib.Path.cwd() / "job_configs" / candidate
        if local_job_config.exists():
            return local_job_config.resolve()

        repo_job_config = LIQUID_FINETUNE_DIR / "job_configs" / candidate
        if repo_job_config.exists():
            return repo_job_config.resolve()

    raise FileNotFoundError(f"Config file not found at: {input_path}")


def _resolve_local_path(value: str | None, *, base_dir: pathlib.Path) -> str | None:
    if not value:
        return value

    expanded = pathlib.Path(value).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())

    if value.startswith(("./", "../")):
        return str((base_dir / value).resolve())

    for root in (base_dir, pathlib.Path.cwd(), LIQUID_FINETUNE_DIR):
        candidate = root / value
        if candidate.exists():
            return str(candidate.resolve())

    return value


def _load_yaml_config(path_obj: pathlib.Path) -> dict[str, Any]:
    with open(path_obj) as f:
        config_dict = yaml.safe_load(f) or {}
    if not isinstance(config_dict, dict):
        raise ValueError(f"Config must be a YAML mapping: {path_obj}")
    return config_dict


def _resolve_reward_paths_to_absolute(
    rewards_cfg: list[Any] | dict[str, Any] | None,
    config_dir: pathlib.Path,
):
    config_dir = config_dir.resolve()
    cwd = pathlib.Path.cwd().resolve()
    rewards_dir = _REWARDS_DIR.resolve()

    def _abs(spec: str) -> str:
        if _REWARD_SEP not in spec:
            return spec

        path_str, sep, name = spec.partition(_REWARD_SEP)
        raw = pathlib.Path(path_str.strip())
        variants = [raw] if raw.suffix else [raw, raw.with_suffix(".py")]
        if raw.is_absolute():
            candidates = [variant.resolve() for variant in variants]
        else:
            candidates = [
                (base / variant).resolve()
                for base in (config_dir, cwd, rewards_dir)
                for variant in variants
            ]

        for candidate in dict.fromkeys(candidates):
            if candidate.exists() and candidate.is_file():
                return f"{candidate}{sep}{name}"
        return spec

    if isinstance(rewards_cfg, list):
        return [_abs(spec) if isinstance(spec, str) else spec for spec in rewards_cfg]
    if isinstance(rewards_cfg, dict):
        out = dict(rewards_cfg)
        for key in ("funcs", "rewards"):
            if key in out and isinstance(out[key], list):
                out[key] = [
                    _abs(spec) if isinstance(spec, str) else spec for spec in out[key]
                ]
        if isinstance(out.get("recipe"), str):
            out["recipe"] = _abs(out["recipe"])
        return out
    return rewards_cfg


def _require_known_training_type(training_type: str) -> None:
    if training_type not in TRAINING_TYPE_TO_CONFIG:
        raise ValueError(
            f"Unknown training type: {training_type}. "
            f"Available: {list(TRAINING_TYPE_TO_CONFIG)}"
        )


def generate_run_name(
    model_name: str,
    training_type: str,
    dataset_path: str,
    dataset_limit: int | None,
    learning_rate: float | None,
    warmup_ratio: float | None,
    use_peft: bool,
    lora_type: str = "a",
) -> str:
    safe_model_name = model_name.split("/")[-1]
    dataset_name_full = dataset_path.strip("/").split("/")[-1]
    dataset_name = (
        dataset_name_full[:10] if len(dataset_name_full) > 10 else dataset_name_full
    )
    limit_str = str(dataset_limit) if dataset_limit else "all"

    if learning_rate:
        lr_val = str(learning_rate)
        lr_clean = lr_val.replace("e-", "em").replace("-", "")
        lr_str = f"lr{lr_clean}"
    else:
        lr_str = "lr_def"

    if warmup_ratio is not None:
        w_str = f"{warmup_ratio:.1f}".replace(".", "p")
        warmup_str = f"w{w_str}"
    else:
        warmup_str = "w_def"

    lora_str = f"lora_{lora_type}" if use_peft else "no_lora"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_id = os.environ.get("SLURM_JOB_ID", "")

    name = (
        f"{safe_model_name}-{training_type}-{dataset_name}-{limit_str}"
        f"-{lr_str}-{warmup_str}-{lora_str}-{timestamp}"
    )
    if slurm_id:
        name += f"-j{slurm_id}"
    return name


def parse_job_config(config_input: str | pathlib.Path) -> JobConfig:
    path_obj = resolve_config_path(config_input)
    config_dict = _load_yaml_config(path_obj)
    try:
        return JobConfig.model_validate(
            {
                **config_dict,
                "config_dir": str(path_obj.parent.resolve()),
            }
        )
    except ValidationError as e:
        raise ValueError(f"Invalid config: {e}") from e


def parse_eval_config(config_input: str | pathlib.Path) -> EvalRunConfig:
    """Parse a standalone eval config.

    Standalone eval configs intentionally do not require training-only fields
    like ``dataset`` or ``training_config``. They share ``EvalSuiteConfig`` with
    training through the top-level ``evals:`` section.
    """
    path_obj = resolve_config_path(config_input)
    config_dict = _load_yaml_config(path_obj)
    try:
        return EvalRunConfig.model_validate(
            {
                **config_dict,
                "config_dir": str(path_obj.parent.resolve()),
            }
        )
    except ValidationError as e:
        raise ValueError(f"Invalid eval config: {e}") from e


def _resolve_training_type(
    raw_training_type: str,
    model_name: str,
    train_config_dict: dict[str, Any],
) -> str:
    if raw_training_type not in ("sft", "dpo"):
        return raw_training_type
    if not is_moe_model_from_name(model_name):
        return raw_training_type

    base_config = train_config_dict.get("extends") or train_config_dict.get("base")
    uses_moe_config = isinstance(base_config, str) and base_config.startswith("MOE_")
    if uses_moe_config or "moe_training" in train_config_dict:
        return f"moe_{raw_training_type}"
    return raw_training_type


def _build_dataset_loader(
    dataset_cfg: DatasetConfig,
    *,
    config_dir: pathlib.Path,
    model_name: str,
) -> DatasetLoader:
    ds_type = DATASET_TYPE_ALIASES.get(dataset_cfg.type, dataset_cfg.type)
    if ds_type not in VALID_DATASET_TYPES:
        raise ValueError(
            f"Invalid dataset type: '{ds_type}'. Must be one of: {sorted(VALID_DATASET_TYPES)}"
        )

    dataset_path_env = os.getenv("DATASET_PATH")
    effective_train_path = (
        dataset_path_env or dataset_cfg.train_path or dataset_cfg.path
    )
    if not effective_train_path:
        raise ValueError("dataset.path or dataset.train_path is required")

    effective_train_path = _resolve_local_path(
        effective_train_path,
        base_dir=pathlib.Path.cwd() if dataset_path_env else config_dir,
    )
    val_path = _resolve_local_path(dataset_cfg.val_path, base_dir=config_dir)
    shared_subset = dataset_cfg.subset
    train_subset = dataset_cfg.train_subset or shared_subset
    val_subset = dataset_cfg.val_subset or shared_subset
    train_split = dataset_cfg.train_split or dataset_cfg.split
    val_split = dataset_cfg.val_split
    if val_path and val_split is None:
        val_split = "train"

    test_size = dataset_cfg.test_size
    if (
        ds_type in ("grpo", "vlm_grpo")
        and test_size is None
        and val_path is None
        and val_split is None
    ):
        test_size = 0.01

    return DatasetLoader(
        dataset_path=effective_train_path,
        dataset_type=ds_type,
        model_name=model_name,
        limit=dataset_cfg.limit,
        split=train_split,
        test_size=test_size,
        subset=train_subset,
        val_dataset_path=val_path,
        val_split=val_split,
        val_subset=val_subset,
        image_root=dataset_cfg.image_root,
        cache_dataset=dataset_cfg.cache_dataset,
        hf_streaming_batch_size=dataset_cfg.hf_streaming_batch_size,
    )


def _build_training_defaults(
    train_config_dict: dict[str, Any],
    training_type: str,
):
    _require_known_training_type(training_type)
    train_config_dict = train_config_dict.copy()
    base_config_name = train_config_dict.pop("extends", None) or train_config_dict.pop(
        "base", None
    )

    if base_config_name:
        if base_config_name not in TRAINING_DEFAULTS:
            available = list(TRAINING_DEFAULTS.keys())
            raise ValueError(
                f"Unknown base config: {base_config_name}. Available: {available}"
            )
        base_train_config = TRAINING_DEFAULTS[base_config_name]
    else:
        base_train_config = TRAINING_DEFAULTS[TRAINING_TYPE_TO_CONFIG[training_type]]

    for float_key in ("learning_rate", "weight_decay"):
        if float_key in train_config_dict and isinstance(
            train_config_dict[float_key], str
        ):
            train_config_dict[float_key] = float(train_config_dict[float_key])

    resolved_train_config = base_train_config.copy()
    resolved_train_config.update(
        {k: v for k, v in train_config_dict.items() if v is not None}
    )
    return _ResolvedConfigValue(resolved_train_config), train_config_dict


def _build_peft_defaults(peft_dict: dict[str, Any] | None):
    peft_dict = peft_dict.copy() if peft_dict else {}
    use_peft = peft_dict.get("use_peft")
    if use_peft is False:
        return None, use_peft

    base_peft_name = peft_dict.pop("extends", None) or peft_dict.pop("base", None)
    peft_dict.pop("use_peft", None)

    if not base_peft_name:
        peft_config = (
            _ResolvedConfigValue(PEFT_DEFAULTS["DEFAULT_LORA"])
            if use_peft is True
            else None
        )
        return peft_config, use_peft

    if base_peft_name not in PEFT_DEFAULTS:
        available = list(PEFT_DEFAULTS.keys())
        raise ValueError(
            f"Unknown base PEFT config: {base_peft_name}. Available: {available}"
        )

    base_config_value = PEFT_DEFAULTS[base_peft_name]
    if base_config_value is None:
        return None, use_peft

    base_dict = (
        base_config_value.to_dict()
        if hasattr(base_config_value, "to_dict")
        else dict(base_config_value)
    )
    base_dict.update({k: v for k, v in peft_dict.items() if v is not None})
    return _ResolvedConfigValue(LoraConfig(**base_dict)), use_peft


def _resolve_output_dir(
    *,
    final_train_values: dict[str, Any],
    project_name: str,
    run_name: str,
) -> pathlib.Path:
    resume_from = final_train_values.get("resume_from_checkpoint")
    base_project_dir = os.getenv("OUTPUT_DIR", f"./outputs/{project_name}")

    if resume_from and resume_from != "latest":
        resume_path = pathlib.Path(resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        return resume_path.parent

    if resume_from == "latest":
        project_path = pathlib.Path(base_project_dir).resolve()
        run_dirs = (
            [
                d
                for d in project_path.iterdir()
                if d.is_dir() and (d / "latest").exists()
            ]
            if project_path.exists()
            else []
        )
        if run_dirs:
            final_output_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
            latest_link = final_output_dir / "latest"
            final_train_values["resume_from_checkpoint"] = str(latest_link.resolve())
            return final_output_dir

        final_train_values.pop("resume_from_checkpoint", None)
        return project_path / run_name

    return pathlib.Path(base_project_dir).resolve() / run_name


def _resolve_eval_suite_paths(
    evals: EvalSuiteConfig,
    *,
    base_dir: pathlib.Path,
) -> dict[str, Any]:
    benchmark_configs = evals.model_dump(exclude_none=True)
    if isinstance(benchmark_configs.get("image_root"), str):
        benchmark_configs["image_root"] = _resolve_local_path(
            benchmark_configs["image_root"],
            base_dir=base_dir,
        )
    for bench in benchmark_configs.get("benchmarks", []):
        if isinstance(bench.get("path"), str):
            bench["path"] = _resolve_local_path(bench["path"], base_dir=base_dir)
        if isinstance(bench.get("image_root"), str):
            bench["image_root"] = _resolve_local_path(
                bench["image_root"],
                base_dir=base_dir,
            )
    return benchmark_configs


def materialize_eval_config(eval_config: EvalRunConfig) -> EvalRunConfig:
    """Resolve local paths in a standalone eval config."""
    config_dir = pathlib.Path(eval_config.config_dir or pathlib.Path.cwd()).resolve()
    payload = eval_config.model_dump(by_alias=True, exclude_none=True)

    if isinstance(payload.get("checkpoint"), str):
        payload["checkpoint"] = _resolve_local_path(
            payload["checkpoint"],
            base_dir=config_dir,
        )

    output_path = payload.get("output_path")
    if isinstance(output_path, str):
        payload["output_path"] = _resolve_local_path(output_path, base_dir=config_dir)

    if eval_config.evals:
        payload["evals"] = _resolve_eval_suite_paths(
            eval_config.evals,
            base_dir=config_dir,
        )
    payload["config_dir"] = str(config_dir)
    return EvalRunConfig.model_validate(payload)


def _create_output_dir(
    *,
    output_dir: pathlib.Path,
    project_name: str,
    run_name: str,
) -> pathlib.Path:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except PermissionError:
        logger.warning(
            "Permission denied creating %s, falling back to local ./outputs",
            output_dir,
        )
        fallback_dir = pathlib.Path.cwd() / "outputs" / project_name / run_name
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir


def materialize_job_config(job_config: JobConfig) -> ResolvedJobConfig:
    config_dir = pathlib.Path(job_config.config_dir or pathlib.Path.cwd()).resolve()
    model_name = job_config.model_name

    dataset = _build_dataset_loader(
        job_config.dataset,
        config_dir=config_dir,
        model_name=model_name,
    )

    train_config_dict = job_config.training_config.model_dump(exclude_none=True)
    training_type = _resolve_training_type(
        job_config.training_type,
        model_name,
        train_config_dict,
    )
    final_training_config, train_config_overrides = _build_training_defaults(
        train_config_dict,
        training_type,
    )
    final_train_values = final_training_config.value
    final_train_values["chat_template_path"] = _resolve_local_path(
        final_train_values.get("chat_template_path"),
        base_dir=config_dir,
    )
    final_train_values["adapter_path"] = _resolve_local_path(
        final_train_values.get("adapter_path"),
        base_dir=config_dir,
    )

    peft_dict = (
        job_config.peft_config.model_dump(exclude_none=True)
        if job_config.peft_config
        else None
    )
    peft_config, use_peft = _build_peft_defaults(peft_dict)
    project_name = job_config.resolved_job_name

    run_name = generate_run_name(
        model_name=model_name,
        training_type=training_type,
        dataset_path=dataset.dataset_path,
        dataset_limit=job_config.dataset.limit,
        learning_rate=final_train_values.get("learning_rate"),
        warmup_ratio=final_train_values.get("warmup_ratio"),
        use_peft=use_peft is not False and peft_config is not None,
        lora_type="a",
    )

    final_output_dir = _resolve_output_dir(
        final_train_values=final_train_values,
        project_name=project_name,
        run_name=run_name,
    )
    final_output_dir = _create_output_dir(
        output_dir=final_output_dir,
        project_name=project_name,
        run_name=run_name,
    )
    final_train_values["output_dir"] = str(final_output_dir)
    final_train_values["liquid_run_name_template"] = run_name
    # Default 0.2 eval split for offline types, only when eval is enabled;
    # eval_strategy="no" keeps the full dataset. (grpo gets 0.01 at load.)
    if (
        not dataset.has_eval_dataset()
        and training_type not in ("grpo", "vlm_grpo")
        and final_train_values.get("eval_strategy", "no") != "no"
    ):
        dataset.test_size = 0.2
    if not dataset.has_eval_dataset():
        raw_eval_strategy = train_config_overrides.get("eval_strategy")
        if raw_eval_strategy and raw_eval_strategy != "no":
            raise ValueError(
                "training_config.eval_strategy requires a validation dataset. "
                "Set dataset.test_size, dataset.val_split, or dataset.val_path, or disable eval."
            )
        final_train_values["eval_strategy"] = "no"

    rewards_cfg = _resolve_reward_paths_to_absolute(job_config.rewards, config_dir)
    rl_env_cfg = job_config.rl_env
    grpo_rollout_cfg = job_config.grpo_rollout
    benchmark_configs = None
    if job_config.evals:
        benchmark_configs = _resolve_eval_suite_paths(
            job_config.evals,
            base_dir=config_dir,
        )

    async_eval_cfg = (
        job_config.async_eval.to_dict() if job_config.async_eval is not None else None
    )

    _validate_parallelism_config(
        final_train_values,
        training_type,
        model_name,
    )

    return ResolvedJobConfig(
        job_name=project_name,
        model_name=model_name,
        training_type=training_type,
        dataset=dataset,
        training_config=final_training_config,
        peft_config=peft_config,
        benchmark_configs=benchmark_configs,
        model_config=job_config.model_overrides,
        ray_config=job_config.ray.model_dump(exclude_none=True)
        if job_config.ray
        else None,
        rewards=rewards_cfg,
        rl_env=rl_env_cfg,
        grpo_rollout=grpo_rollout_cfg,
        async_eval=async_eval_cfg,
        config_dir=str(config_dir),
    )


def normalized_job_config_dict(
    job_config: JobConfig,
    *,
    base_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    base_dir = (
        base_dir or pathlib.Path(job_config.config_dir or pathlib.Path.cwd())
    ).resolve()
    payload = job_config.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    dataset_cfg = payload.get("dataset", {})
    for key in ("path", "train_path", "val_path", "image_root"):
        if isinstance(dataset_cfg.get(key), str):
            dataset_cfg[key] = _resolve_local_path(dataset_cfg[key], base_dir=base_dir)

    train_cfg = payload.get("training_config", {})
    for key in ("chat_template_path", "adapter_path"):
        if isinstance(train_cfg.get(key), str):
            train_cfg[key] = _resolve_local_path(train_cfg[key], base_dir=base_dir)

    benchmark_cfg = payload.get("evals", {})
    if isinstance(benchmark_cfg.get("image_root"), str):
        benchmark_cfg["image_root"] = _resolve_local_path(
            benchmark_cfg["image_root"], base_dir=base_dir
        )
    for bench in benchmark_cfg.get("benchmarks", []):
        if isinstance(bench.get("path"), str):
            bench["path"] = _resolve_local_path(bench["path"], base_dir=base_dir)
        if isinstance(bench.get("image_root"), str):
            bench["image_root"] = _resolve_local_path(
                bench["image_root"], base_dir=base_dir
            )

    if "rewards" in payload:
        payload["rewards"] = _resolve_reward_paths_to_absolute(
            payload["rewards"], base_dir
        )
    payload["config_dir"] = str(base_dir)
    return payload


def _validate_parallelism_config(
    training_config: dict[str, Any],
    training_type: str,
    model_name: str,
) -> None:
    if (training_config.get("context_parallel_size", 1) or 1) > 1:
        raise ValueError("context_parallel_size is not supported in the EP-only branch")

    moe_config = training_config.get("moe_training", {})
    if not moe_config:
        return

    effective_training_type = training_type
    if training_type in ("sft", "dpo") and is_moe_model_from_name(model_name):
        effective_training_type = f"moe_{training_type}"

    if effective_training_type in ("moe_sft", "moe_dpo"):
        # EP forward path ignores capacity dropping; reject for both so it isn't a silent no-op.
        capacity_factor = moe_config.get("capacity_factor")
        token_drop_policy = moe_config.get("token_drop_policy")
        if capacity_factor is not None or token_drop_policy not in (None, "probs"):
            raise ValueError(
                "MoE training currently supports uncapped routing only. "
                "Remove capacity_factor/token_drop_policy from moe_training."
            )

    ep_size = moe_config.get("expert_parallel_size", 1) or 1
    if ep_size <= 1:
        return

    if effective_training_type not in ("moe_sft", "moe_dpo"):
        raise ValueError(
            f"expert_parallel_size={ep_size} requires training_type 'moe_sft' or 'moe_dpo', "
            f"got '{training_type}'"
        )

    if ep_size & (ep_size - 1) != 0:
        raise ValueError(f"expert_parallel_size must be a power of 2, got {ep_size}")


def print_job_config_summary(job_config: ResolvedJobConfig) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    config_value = job_config.training_config.value
    peft_value = job_config.peft_config.value if job_config.peft_config else None

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="bold cyan", min_width=18)
    table.add_column("Value", style="green")

    table.add_row("Project", job_config.job_name)
    table.add_row("Model", job_config.model_name)
    table.add_row("Training Type", job_config.training_type)
    table.add_row(
        "Dataset", job_config.dataset.dataset_path if job_config.dataset else "None"
    )
    table.add_row(
        "Dataset Type",
        job_config.dataset.dataset_type if job_config.dataset else "None",
    )
    table.add_row("Output Dir", str(config_value.get("output_dir", "")))
    if peft_value is not None:
        table.add_row("PEFT", peft_value.__class__.__name__)
    else:
        table.add_row("PEFT", "disabled")

    for key in (
        "num_train_epochs",
        "per_device_train_batch_size",
        "learning_rate",
        "gradient_accumulation_steps",
        "eval_strategy",
        "save_strategy",
    ):
        if key in config_value:
            table.add_row(key, str(config_value[key]))

    Console().print(Panel(table, title="Liquid Job Config", border_style="blue"))
