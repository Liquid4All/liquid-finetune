from __future__ import annotations

# ruff: noqa: E402

import logging
from typing import Any

import torch

from leap_finetune.distribution.ray_runtime import normalize_visible_devices

normalize_visible_devices()

from trl import GRPOConfig, GRPOTrainer

from leap_finetune.checkpointing.callback import LeapCheckpointCallback
from leap_finetune.checkpointing.model_loading import load_vlm_model
from leap_finetune.data_loading.image_loader import load_image
from leap_finetune.evaluation import (
    create_vlm_benchmarks_from_config,
    make_eval_callback,
)
from leap_finetune.rl.rewards import resolve_reward_specs
from leap_finetune.training.default_configs.grpo_configs import VLM_GRPO_EXCLUDED_KEYS
from leap_finetune.training.default_configs.vlm_sft_configs import (
    DEFAULT_LR_MULTIPLIERS,
)
from leap_finetune.training.peft.peft import (
    apply_peft_to_model,
    merge_and_save_peft_model,
)
from leap_finetune.training.utils.logging import (
    finish_tracker,
    get_wandb_run_id,
    is_rank_zero,
)
from leap_finetune.training.utils.trainer_lifecycle import run_training_safely
from leap_finetune.training.utils.vlm_optimizer import (
    build_vlm_param_groups,
    log_per_group_lrs,
)
from leap_finetune.training.utils.worker_setup import (
    init_tracking_from_config,
    resolve_train_eval_datasets,
)
from leap_finetune.training.utils.config_filter import filter_runtime_config_kwargs

logger = logging.getLogger(__name__)


# === VLM GRPO loop ===


class LFMVLMGRPOTrainer(GRPOTrainer):
    """Leap integration around TRL native LFM2-VL GRPO support.

    TRL 1.7+ owns multimodal image discovery, tile-aware buffering,
    ``spatial_shapes`` propagation, chunked log-probabilities, and the fused
    Liger hidden-state path. Leap retains per-component optimizer groups and
    local image-path normalization. GRPO must continue using TRL native
    RepeatSampler/Accelerate distribution rather than RayDataLoaderMixin.
    """

    def __init__(self, lr_multipliers: dict[str, float] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.lr_multipliers = lr_multipliers or DEFAULT_LR_MULTIPLIERS
        self._optimizer_group_names: list[str] = []
        self._vlm_grpo_loaded_images = []

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        optimizer_groups, self._optimizer_group_names = build_vlm_param_groups(
            self.model,
            self.lr_multipliers,
            base_lr=self.args.learning_rate,
            weight_decay=float(self.args.weight_decay),
        )

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.AdamW(
            optimizer_groups, betas=betas, fused=torch.cuda.is_available()
        )
        return self.optimizer

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        log_per_group_lrs(self.optimizer, self._optimizer_group_names, logs)
        super().log(logs, *args, **kwargs)

    def _tokenize_prompts(self, prompts: list):
        """Resolve any remaining image path strings to PIL objects.

        TRL 1.7+ discovers images directly in conversational prompt blocks.
        Leap datasets may store local paths in those blocks, while image
        processors and vLLM expect decoded image objects, so normalize paths
        before delegating to TRL.
        """
        patched_prompts = []
        for prompt in prompts:
            new_prompt = []
            for message in prompt:
                content = message.get("content")
                if isinstance(content, list):
                    new_content = []
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "image"
                            and isinstance(part.get("image"), str)
                        ):
                            img = load_image(part["image"])
                            self._vlm_grpo_loaded_images.append(img)
                            new_content.append({"type": "image", "image": img})
                        else:
                            new_content.append(part)
                    new_prompt.append({**message, "content": new_content})
                else:
                    new_prompt.append(message)
            patched_prompts.append(new_prompt)
        return super()._tokenize_prompts(patched_prompts)

    def _generate_and_score_completions(self, inputs):
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            for image in self._vlm_grpo_loaded_images:
                image.close()
            self._vlm_grpo_loaded_images.clear()


def vlm_grpo_run(training_config: dict, train_dataset=None, eval_dataset=None) -> None:
    train_dataset, eval_dataset, prepare_trainer = resolve_train_eval_datasets(
        train_dataset, eval_dataset
    )

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    train_config = training_config.get("train_config", {})
    max_image_tokens = train_config.get("max_image_tokens")
    do_image_splitting = train_config.get("do_image_splitting", True)
    run_name_template = train_config.get("leap_run_name_template")

    lr_multipliers = dict(DEFAULT_LR_MULTIPLIERS)
    if "lr_multipliers" in train_config:
        lr_multipliers.update(train_config["lr_multipliers"])
    if "vision_encoder_lr_multiplier" in train_config:
        lr_multipliers["model.vision_tower"] = train_config[
            "vision_encoder_lr_multiplier"
        ]

    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    excluded_keys = VLM_GRPO_EXCLUDED_KEYS | {"leap_run_name_template"}
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

    model, processor = load_vlm_model(
        model_name,
        max_image_tokens=max_image_tokens,
        do_image_splitting=do_image_splitting,
    )
    # GRPO appends completions to prompts, so left padding keeps positions sane.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    reward_funcs, reward_weights = resolve_reward_specs(
        training_config.get("rewards"),
        training_config.get("config_dir") or ".",
    )

    # Deferred import keeps OpenEnv optional for plain reward-function GRPO.
    rl_env_cfg = training_config.get("rl_env")
    rollout_func = None
    if rl_env_cfg is not None:
        try:
            from leap_finetune.rl.environments import (  # noqa: PLC0415
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
            "VLM GRPO requires at least one reward function. Add a `rewards:` "
            "block with './rewards/<file>.py::<fn>' specs, or set `rl_env:` to "
            "use an OpenEnv environment's reward."
        )

    if reward_weights is not None:
        training_args.reward_weights = reward_weights

    trainer = LFMVLMGRPOTrainer(
        lr_multipliers=lr_multipliers,
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        rollout_func=rollout_func,
    )

    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))

    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_vlm_benchmarks_from_config(benchmark_configs, processor)
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

    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, processor, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)
