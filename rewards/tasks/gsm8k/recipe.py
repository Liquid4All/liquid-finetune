from __future__ import annotations

import re

from liquid_finetune.rl.rewards import Recipe

_GSM8K_MARKER = re.compile(r"####\s*([\-\+]?[\d,\.]+)")
_ANY_NUMBER = re.compile(r"([\-\+]?[\d,\.]+)")


# === GSM8K numeric exact match ===
#
# Required columns: prompt, solution. The solution may be a bare number or a
# chain-of-thought string ending in "#### <answer>".


def _normalize_number(s) -> str | None:
    if s is None:
        return None
    s = str(s).strip().rstrip(".").replace(",", "").replace("$", "").replace(" ", "")
    if not s or not re.search(r"\d", s):
        return None
    try:
        return str(float(s))
    except (ValueError, TypeError):
        return s or None


def _extract_answer(text: str) -> str | None:
    if not text:
        return None
    tail = text[-1024:]
    matches = _GSM8K_MARKER.findall(tail)
    if matches:
        return _normalize_number(matches[-1])
    matches = _ANY_NUMBER.findall(tail)
    if matches:
        return _normalize_number(matches[-1])
    return None


def _completion_text(completion) -> str:
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "") or ""
        return str(first)
    return str(completion)


def _parse_gt(gt_raw) -> str | None:
    if gt_raw is None:
        return None
    s = str(gt_raw)
    matches = _GSM8K_MARKER.findall(s)
    if matches:
        return _normalize_number(matches[-1])
    return _normalize_number(s)


def gsm8k_reward(completions, solution=None, **kwargs) -> list[float | None]:
    """Return 1.0 on numeric match, 0.0 on mismatch, None if gold missing."""
    n = len(completions)
    if solution is None:
        return [None] * n
    rewards: list[float | None] = []
    for i, completion in enumerate(completions):
        gt = _parse_gt(solution[i] if i < len(solution) else None)
        if gt is None:
            rewards.append(None)
            continue
        pred = _extract_answer(_completion_text(completion))
        rewards.append(1.0 if pred == gt else 0.0)
    return rewards


class GSM8KRecipe(Recipe):
    description = "GSM8K - exact match on the final numeric answer."

    required_columns = ("prompt", "solution")

    system_prompt = (
        "Solve the math problem step by step. Put your final numeric "
        "answer at the end of your response, preceded by '#### '. "
        "Example: '#### 42'."
    )

    def rewards(self):
        return [(gsm8k_reward, 1.0)]
