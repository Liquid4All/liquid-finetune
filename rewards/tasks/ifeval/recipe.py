from __future__ import annotations

import json
import re

from liquid_finetune.rl.rewards import Recipe

_HIGHLIGHT_PATTERN = re.compile(r"\*[^*\n]+\*")
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")


# === IFEval constraint scoring ===
#
# Required columns: prompt, solution. The solution is a JSON constraint spec.
# Unsupported instruction IDs are skipped; samples with zero supported
# constraints return None so GRPO drops them from advantage computation.


def _completion_text(completion) -> str:
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "") or ""
        return str(first)
    return str(completion)


def _check_no_comma(text, kwargs):
    return "," not in text


def _check_num_words(text, kwargs):
    if not isinstance(kwargs, dict):
        return None
    target = kwargs.get("num_words")
    if target is None:
        return None
    relation = (kwargs.get("relation") or "at least").lower().strip()
    n_words = len(_WORD_PATTERN.findall(text))
    if relation in ("at least", "at_least", ">=", "atleast"):
        return n_words >= target
    if relation in ("at most", "at_most", "<=", "atmost"):
        return n_words <= target
    if relation in ("less than", "less_than", "<"):
        return n_words < target
    if relation in ("more than", "more_than", ">"):
        return n_words > target
    if relation in ("exactly", "equal", "equal to", "=="):
        return n_words == target
    return None


def _check_highlighted_sections(text, kwargs):
    if not isinstance(kwargs, dict):
        return None
    target = kwargs.get("num_highlights")
    if target is None:
        return None
    return len(_HIGHLIGHT_PATTERN.findall(text)) >= target


def _check_keywords_multiple(text, kwargs):
    if not isinstance(kwargs, dict):
        return None
    keywords = [
        v for k, v in kwargs.items() if k.startswith("keyword") and isinstance(v, str)
    ]
    if not keywords:
        return None
    lower = text.lower()
    return all(kw.lower() in lower for kw in keywords)


def _check_letter_count_in_word(text, kwargs):
    if not isinstance(kwargs, dict):
        return None
    letter = kwargs.get("letter")
    word = kwargs.get("word")
    if not isinstance(letter, str) or not isinstance(word, str):
        return None
    correct = str(word.lower().count(letter.lower()))
    return re.search(rf"\b{re.escape(correct)}\b", text) is not None


_CHECKERS = {
    "punctuation:no_comma": _check_no_comma,
    "length_constraints:number_words": _check_num_words,
    "detectable_format:number_highlighted_sections": _check_highlighted_sections,
    "count:keywords_multiple": _check_keywords_multiple,
    "counting:letter_count_in_word": _check_letter_count_in_word,
}


def _parse_constraints(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    else:
        parsed = raw
    if not isinstance(parsed, list) or not parsed:
        return None
    out: list[tuple[str, dict]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        ids = item.get("instruction_id") or []
        kwargs_list = item.get("kwargs") or []
        for i, inst_id in enumerate(ids):
            kw = kwargs_list[i] if i < len(kwargs_list) else None
            out.append((inst_id, kw or {}))
    return out or None


def ifeval_reward(completions, solution=None, **kwargs) -> list[float | None]:
    """Return the fraction of supported constraints satisfied, in ``[0, 1]``."""
    n = len(completions)
    if solution is None:
        return [None] * n
    rewards: list[float | None] = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        constraints = _parse_constraints(solution[i] if i < len(solution) else None)
        if not constraints:
            rewards.append(None)
            continue
        results: list[float] = []
        for inst_id, inst_kwargs in constraints:
            checker = _CHECKERS.get(inst_id)
            if checker is None:
                continue
            try:
                ok = checker(text, inst_kwargs)
            except Exception:
                ok = False
            if ok is None:
                continue
            results.append(1.0 if ok else 0.0)
        if not results:
            rewards.append(None)
            continue
        rewards.append(sum(results) / len(results))
    return rewards


class IFEvalRecipe(Recipe):
    description = "IFEval - fraction of instruction constraints satisfied."

    required_columns = ("prompt", "solution")

    system_prompt = (
        "Follow the instruction in the user turn exactly. Satisfy "
        "every constraint mentioned (word count, formatting, "
        "punctuation, required keywords). Do not add explanations "
        "outside the requested response."
    )

    def rewards(self):
        return [(ifeval_reward, 1.0)]
