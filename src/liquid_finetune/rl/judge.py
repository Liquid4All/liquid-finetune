from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

JUDGE_LLM_CONFIG_ENV = "LIQUID_JUDGE_LLM_CONFIG"

_DEFAULT_SYSTEM_PROMPT = (
    "You are a strict grading judge. Score the assistant response for the "
    'requested task. Return only a JSON object like {"score": 0.0}, where '
    "score is a number in the configured range."
)

_DEFAULT_PROMPT_TEMPLATE = """Prompt:
{prompt}

Assistant response:
{completion}

Reference answer or rubric:
{solution}

Return only JSON with one numeric field named "score"."""

_SCORE_FIELD_RE = re.compile(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


# === Judge LLM reward runtime ===
#
# The driver exports one JSON config to Ray workers. The reward callable stays
# a normal shipped primitive (`judge_llm_reward`) while the HTTP endpoint/model
# can be configured per YAML run.


@dataclass(frozen=True)
class JudgeLLMConfig:
    model: str
    base_url: str
    tokenizer: str | None = None
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    prompt_template: str = _DEFAULT_PROMPT_TEMPLATE
    min_score: float = 0.0
    max_score: float = 1.0
    failure_score: float = 0.0
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 8
    timeout_s: float = 120.0
    use_chat_template: bool = True
    clip_score: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, default_model: str | None = None):
        model = str(raw.get("model") or default_model or "").strip()
        base_url = str(raw.get("base_url") or "").strip().rstrip("/")
        if not model:
            raise ValueError("`rewards.judge.model` is required.")
        if not base_url:
            raise ValueError("Judge LLM base_url is required at runtime.")

        min_score = float(raw.get("min_score", 0.0))
        max_score = float(raw.get("max_score", 1.0))
        if max_score <= min_score:
            raise ValueError(
                "`rewards.judge.max_score` must be greater than min_score."
            )

        return cls(
            model=model,
            base_url=base_url,
            tokenizer=raw.get("tokenizer"),
            system_prompt=str(raw.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)),
            prompt_template=str(raw.get("prompt_template", _DEFAULT_PROMPT_TEMPLATE)),
            min_score=min_score,
            max_score=max_score,
            failure_score=float(raw.get("failure_score", 0.0)),
            max_tokens=int(raw.get("max_tokens", 32)),
            temperature=float(raw.get("temperature", 0.0)),
            top_p=float(raw.get("top_p", 1.0)),
            batch_size=int(raw.get("batch_size", 8)),
            timeout_s=float(raw.get("timeout_s", raw.get("timeout", 120.0))),
            use_chat_template=bool(raw.get("use_chat_template", True)),
            clip_score=bool(raw.get("clip_score", True)),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class JudgeLLM:
    """Scores completions by calling a TRL vLLM server and parsing a score."""

    def __init__(
        self,
        config: JudgeLLMConfig,
        *,
        session: requests.Session | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self._tokenizer = tokenizer

    def score(self, completions, **kwargs) -> list[float]:
        prompts = [
            self._render_prompt(i, completion, kwargs)
            for i, completion in enumerate(completions)
        ]
        judge_outputs = self._generate(prompts)
        return [self._parse_score(text) for text in judge_outputs]

    def _render_prompt(self, index: int, completion, kwargs: dict[str, Any]) -> str:
        fields = {
            key: _format_value(_item_at(value, index))
            for key, value in kwargs.items()
            if key not in {"completion_ids", "trainer_state"}
        }
        fields["completion"] = _completion_text(completion)
        fields.setdefault(
            "prompt",
            _format_value(_item_at(kwargs.get("prompts", kwargs.get("prompt")), index)),
        )
        fields.setdefault(
            "solution", _format_value(_item_at(kwargs.get("solution"), index))
        )
        user_prompt = self.config.prompt_template.format_map(_SafeFormatDict(fields))

        tokenizer = self.tokenizer
        if (
            self.config.use_chat_template
            and getattr(tokenizer, "chat_template", None)
            and hasattr(tokenizer, "apply_chat_template")
        ):
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{self.config.system_prompt}\n\n{user_prompt}\n"

    def _generate(self, prompts: list[str]) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), self.config.batch_size):
            batch = prompts[start : start + self.config.batch_size]
            response = self.session.post(
                f"{self.config.base_url}/generate/",
                json={
                    "prompts": batch,
                    "n": 1,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens,
                    "logprobs": None,
                },
                timeout=self.config.timeout_s,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Judge LLM request failed: {response.status_code} {response.text}"
                )
            completion_ids = response.json()["completion_ids"]
            outputs.extend(
                self.tokenizer.batch_decode(
                    completion_ids,
                    skip_special_tokens=True,
                )
            )
        return outputs

    def _parse_score(self, text: str) -> float:
        raw_score = _extract_score(text)
        if raw_score is None:
            logger.warning(
                "Judge LLM output did not contain a numeric score; using %s. Output: %r",
                self.config.failure_score,
                text,
            )
            return self.config.failure_score

        score = (raw_score - self.config.min_score) / (
            self.config.max_score - self.config.min_score
        )
        if self.config.clip_score:
            score = max(0.0, min(1.0, score))
        return float(score)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer or self.config.model,
                trust_remote_code=True,
            )
        return self._tokenizer


def get_judge_config(rewards_cfg: list | dict | None) -> dict[str, Any] | None:
    if not isinstance(rewards_cfg, dict) or "judge" not in rewards_cfg:
        return None
    raw = rewards_cfg["judge"]
    if raw is None or raw is False:
        return None
    if raw is True:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"`rewards.judge` must be a dict, got {type(raw).__name__}.")
    return dict(raw)


def judge_needs_local_server(rewards_cfg: list | dict | None) -> bool:
    judge_cfg = get_judge_config(rewards_cfg)
    return bool(judge_cfg is not None and not judge_cfg.get("base_url"))


def build_judge_runtime_config(
    rewards_cfg: list | dict | None,
    *,
    default_model: str,
    base_url: str | None = None,
) -> JudgeLLMConfig | None:
    judge_cfg = get_judge_config(rewards_cfg)
    if judge_cfg is None:
        return None
    raw = dict(judge_cfg)
    if base_url is not None:
        raw["base_url"] = base_url
    return JudgeLLMConfig.from_dict(raw, default_model=default_model)


def export_judge_runtime_config(config: JudgeLLMConfig | None) -> None:
    if config is None:
        os.environ.pop(JUDGE_LLM_CONFIG_ENV, None)
        return
    os.environ[JUDGE_LLM_CONFIG_ENV] = config.to_json()


def judge_llm_reward(completions, **kwargs) -> list[float]:
    """Reward callable that scores completions with the configured judge LLM."""
    return _judge_from_env().score(completions, **kwargs)


_JUDGE_CACHE: JudgeLLM | None = None
_JUDGE_CACHE_RAW: str | None = None


def _judge_from_env() -> JudgeLLM:
    global _JUDGE_CACHE, _JUDGE_CACHE_RAW
    raw = os.environ.get(JUDGE_LLM_CONFIG_ENV)
    if not raw:
        raise RuntimeError(
            f"{JUDGE_LLM_CONFIG_ENV} is not set. Add `rewards.judge` to the "
            "GRPO YAML or export a judge config before using judge_llm_reward."
        )
    if _JUDGE_CACHE is None or raw != _JUDGE_CACHE_RAW:
        _JUDGE_CACHE = JudgeLLM(JudgeLLMConfig.from_dict(json.loads(raw)))
        _JUDGE_CACHE_RAW = raw
    return _JUDGE_CACHE


def _extract_score(text: str) -> float | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("score"), (int, float)):
        return float(parsed["score"])

    match = _SCORE_FIELD_RE.search(text)
    if match:
        return float(match.group(1))

    match = _NUMBER_RE.search(text)
    return float(match.group(0)) if match else None


def _completion_text(completion) -> str:
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion)


def _item_at(value, index: int):
    if value is None:
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if index < len(value):
            return value[index]
    return value


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in value
        )
    return str(value)


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return ""
