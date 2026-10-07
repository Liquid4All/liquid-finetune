from __future__ import annotations


# === OpenEnv reward bridge ===


def env_reward(completions, **kwargs) -> list[float]:
    """Read rewards forwarded by the OpenEnv rollout function."""
    env_rewards = kwargs.get("env_reward")
    if env_rewards is None:
        return [0.0] * len(completions)
    return [float(r) if r is not None else 0.0 for r in env_rewards]
