# === Judge LLM reward ===
#
# Configured through the YAML `rewards.judge:` block. The driver exports the
# resolved model/server config to Ray workers, and this primitive reads it at
# reward-call time.

from liquid_finetune.rl.judge import judge_llm_reward

__all__ = ["judge_llm_reward"]
