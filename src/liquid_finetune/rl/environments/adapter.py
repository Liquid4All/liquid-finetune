from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

ENV_REWARD_KEY = "env_reward"
_DEFAULT_ACTION_KEY = "message"


# === OpenEnv adapter ===
#
# openenv-core is optional. Keep imports inside the functions below so regular
# GRPO jobs do not need the rl-env extra installed.


def connect_openenv(env_cfg: dict) -> Any:
    """Resolve a YAML ``rl_env`` block to a synchronous OpenEnv client.

    One of ``source`` (HF Hub repo-id), ``base_url`` (running server), or
    ``docker_image`` is required. See ``rl/environments/README.md`` for the full
    YAML schema.
    """
    source = env_cfg.get("source")
    base_url = env_cfg.get("base_url")
    docker_image = env_cfg.get("docker_image")

    if not any([source, base_url, docker_image]):
        raise ValueError(
            "`rl_env` must specify one of: source, base_url, or docker_image."
        )

    try:
        from openenv import AutoEnv
    except ImportError as e:
        raise ImportError(
            "openenv-core is required. Install with: uv sync --extra rl-env"
        ) from e

    # Training is non-interactive: default trust_remote_code to True so the
    # AutoEnv security prompt doesn't hang the job.
    trust_remote_code = bool(env_cfg.get("trust_remote_code", True))
    if trust_remote_code:
        os.environ.setdefault("OPENENV_TRUST_REMOTE_CODE", "1")

    logger.info(
        "Connecting to OpenEnv: source=%r base_url=%r docker_image=%r",
        source,
        base_url,
        docker_image,
    )

    name = source or docker_image or base_url

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if base_url:
        kwargs["base_url"] = base_url
    if docker_image:
        kwargs["docker_image"] = docker_image
    if "env_vars" in env_cfg:
        kwargs["env_vars"] = env_cfg["env_vars"]
    if "wait_timeout" in env_cfg:
        kwargs["wait_timeout"] = float(env_cfg["wait_timeout"])
    kwargs["skip_install"] = bool(env_cfg.get("skip_install", False))

    async_client = AutoEnv.from_env(name, **kwargs)
    sync_client = async_client.sync()
    sync_client.connect()
    return sync_client


def build_openenv_rollout_func(
    env: Any,
    *,
    max_turns: int = 1,
    reset_kwargs: dict | None = None,
    action_key: str = _DEFAULT_ACTION_KEY,
) -> Callable:
    """Return a TRL ``rollout_func(prompts, trainer)`` that drives an OpenEnv env.

    Generates completions through the trainer's native generation path, then
    steps each completion through the env and collects ``env_reward`` per
    prompt. This works with Transformers generation and both vLLM modes.
    Only ``max_turns=1`` is supported; multi-turn envs have env-specific
    prompt construction and should ship a custom rollout_func.
    """
    reset_kwargs = reset_kwargs or {}
    if max_turns != 1:
        raise NotImplementedError(
            "max_turns > 1 requires a custom rollout_func; see "
            "rl/environments/README.md."
        )

    def rollout_func(prompts: list, trainer) -> dict:
        prompt_ids, images, multimodal_fields = trainer._tokenize_prompts(prompts)
        completion_ids, logprobs = trainer._generate_single_turn(
            prompt_ids, images, multimodal_fields
        )
        texts = trainer.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        env_rewards: list[float] = []
        for text in texts:
            env.reset(**reset_kwargs)
            result = env.step({action_key: text})
            reward = float(result.reward) if result.reward is not None else 0.0
            env_rewards.append(reward)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            ENV_REWARD_KEY: env_rewards,
        }

    return rollout_func
