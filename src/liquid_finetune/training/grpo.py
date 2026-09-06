from __future__ import annotations

# ruff: noqa: E402

import logging
from typing import cast

from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from liquid_finetune.distribution.ray_runtime import normalize_visible_devices

normalize_visible_devices()

from trl import GRPOConfig, GRPOTrainer

from liquid_finetune.checkpointing.callback import LiquidCheckpointCallback
from liquid_finetune.checkpointing.model_info import is_moe_model_from_name
from liquid_finetune.checkpointing.model_loading import load_model
from liquid_finetune.evaluation import (
    create_llm_benchmarks_from_config,
    make_eval_callback,
)
from liquid_finetune.rl.rewards import resolve_reward_specs
from liquid_finetune.training.default_configs.grpo_configs import GRPO_EXCLUDED_KEYS
from liquid_finetune.training.peft.peft import (
    apply_peft_to_model,
    load_peft_adapter,
    merge_and_save_peft_model,
)
from liquid_finetune.training.utils.logging import (
    finish_tracker,
    get_wandb_run_id,
    is_rank_zero,
)
from liquid_finetune.training.utils.trainer_lifecycle import run_training_safely
from liquid_finetune.training.utils.worker_setup import (
    init_tracking_from_config,
    resolve_train_eval_datasets,
)
from liquid_finetune.training.utils.config_filter import filter_runtime_config_kwargs

logger = logging.getLogger(__name__)


def _apply_grpo_peft(
    model: PreTrainedModel,
    *,
    peft_config: LoraConfig | None,
    adapter_path: str | None,
) -> PreTrainedModel:
    """Load a trainable adapter continuation or create a fresh adapter."""
    if adapter_path:
        return load_peft_adapter(model, adapter_path)
    if peft_config:
        return apply_peft_to_model(model, peft_config)
    return model


# === Text GRPO loop ===
#
# GRPO generates completions online, so it must use TRL's native
# RepeatSampler/accelerate path. The Ray driver gives every worker the full
# dataset; TRL then distributes repeated prompt groups across ranks.


def grpo_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "liquid-ft-run")

    is_moe = is_moe_model_from_name(model_name)
    if is_moe:
        raise ValueError("GRPO for MoE models is not supported in this EP branch")

    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("liquid_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    adapter_path = train_config.get("adapter_path")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    excluded_keys = GRPO_EXCLUDED_KEYS
    train_config_filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=excluded_keys,
        config_cls=GRPOConfig,
    )

    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    training_args = GRPOConfig(**config_kwargs)

    model, tokenizer = load_model(model_name)
    # GRPO requires left-padded prompts so generated completions append cleanly.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _apply_grpo_peft(
        model,
        peft_config=peft_config,
        adapter_path=adapter_path,
    )

    # Resolve reward functions from the driver-side config_dir. Loaders are
    # deterministic, so each worker re-runs the resolution independently
    # rather than shipping closures across processes.
    reward_funcs, reward_weights = resolve_reward_specs(
        training_config.get("rewards"),
        training_config.get("config_dir") or ".",
    )

    # Deferred import keeps OpenEnv optional for plain reward-function GRPO.
    rl_env_cfg = training_config.get("rl_env")
    rollout_func = None
    if rl_env_cfg is not None:
        try:
            from liquid_finetune.rl.environments import (  # noqa: PLC0415
                build_openenv_rollout_func,
                connect_openenv,
                env_reward,
            )
        except ImportError as e:
            raise ImportError(
                "`rl_env:` requires the optional OpenEnv extra. "
                "Install with: uv sync --extra rl-env"
            ) from e

        env_client = connect_openenv(rl_env_cfg)
        rollout_func = build_openenv_rollout_func(
            env_client,
            max_turns=int(rl_env_cfg.get("max_turns", 1)),
            reset_kwargs=rl_env_cfg.get("reset_kwargs") or {},
            action_key=rl_env_cfg.get("action_key", "message"),
        )
        reward_funcs = [env_reward, *reward_funcs]
        if reward_weights is not None:
            reward_weights = [1.0, *reward_weights]

    if not reward_funcs:
        raise ValueError(
            "GRPO requires at least one reward function. Add a `rewards:` block "
            "with a list of './rewards/<file>.py::<fn>' specs, or set `rl_env:` "
            "to use an OpenEnv environment's reward."
        )

    if reward_weights is not None:
        training_args.reward_weights = reward_weights

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=cast(PreTrainedTokenizerBase, tokenizer),
        rollout_func=rollout_func,
    )

    trainer.add_callback(LiquidCheckpointCallback(run_name_template=run_name_template))

    # Reuse benchmark callback (same pattern as SFT/DPO)
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

    # Save PEFT adapter if applicable
    if (peft_config or adapter_path) and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)
