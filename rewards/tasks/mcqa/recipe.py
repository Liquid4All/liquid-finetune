from __future__ import annotations

import re

from liquid_finetune.rl.rewards import Recipe

_ANSWER_COLON = re.compile(
    r"[Aa]nswer(?:\s+is)?\**\s*[:\-]?\s*\**\s*\(?([A-J])\)?",
)
_BOXED = re.compile(r"\\boxed\{\s*([A-J])\s*\}")
_TRAILING_LETTER = re.compile(r"(?:^|[^A-Za-z])([A-J])[\.\)\*]?\s*$")


# === MCQA letter match ===
#
# Required columns: prompt, solution. The solution may be a bare letter A-J or
# a sentence that contains the answer letter.


def _extract_letter(text: str) -> str | None:
    if not text:
        return None
    tail = text[-512:]
    for pattern in (_ANSWER_COLON, _BOXED):
        matches = pattern.findall(tail)
        if matches:
            return matches[-1].upper()
    m = _TRAILING_LETTER.search(tail.strip())
    if m:
        return m.group(1).upper()
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


def _parse_gt_letter(gt) -> str | None:
    if not isinstance(gt, str) or not gt.strip():
        return None
    stripped = gt.strip().upper()
    if len(stripped) == 1 and "A" <= stripped <= "J":
        return stripped
    return _extract_letter(gt)


def mcqa_reward(completions, solution=None, **kwargs) -> list[float | None]:
    """Return 1.0 on letter match, 0.0 on mismatch, None if gold unparseable."""
    n = len(completions)
    if solution is None:
        return [None] * n
    rewards: list[float | None] = []
    for i, completion in enumerate(completions):
        gt_letter = _parse_gt_letter(solution[i] if i < len(solution) else None)
        if gt_letter is None:
            rewards.append(None)
            continue
        pred = _extract_letter(_completion_text(completion))
        rewards.append(1.0 if pred == gt_letter else 0.0)
    return rewards


class MCQARecipe(Recipe):
    description = "MCQA - letter match on A..J; last match in the completion wins."

    required_columns = ("prompt", "solution")

    system_prompt = (
        "Answer the multiple-choice question. Explain your reasoning "
        "briefly, then end your response with 'Answer: <letter>' "
        "where <letter> is one of the provided options."
    )

    def rewards(self):
        return [(mcqa_reward, 1.0)]
