from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SelectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["key_value_line", "json_pointer", "regex", "callable"]
    key: str | None = None
    pointer: str | None = None
    pattern: str | None = None
    group: int | str = 0
    callable: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)
    include_key: bool = False
    include_value: bool = True
    include_delimiter: bool = True
    ignore_case: bool = False
    multiline: bool = True
    dotall: bool = False

    @model_validator(mode="after")
    def _validate_selector(self) -> SelectorConfig:
        required = {
            "key_value_line": ("key", self.key),
            "json_pointer": ("pointer", self.pointer),
            "regex": ("pattern", self.pattern),
            "callable": ("callable", self.callable),
        }
        field_name, value = required[self.type]
        if not value:
            raise ValueError(f"selector type {self.type!r} requires {field_name!r}")
        if not self.include_key and not self.include_value:
            raise ValueError("selector must include its key, value, or both")
        if self.type == "json_pointer" and not self.pointer.startswith("/"):
            raise ValueError("JSON Pointer selectors must start with '/'")
        return self


class LossWeightRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    selector: SelectorConfig
    weight: float = Field(ge=0)

    @model_validator(mode="after")
    def _finite_weight(self) -> LossWeightRule:
        if not math.isfinite(self.weight):
            raise ValueError("loss weights must be finite")
        return self


class LossWeightingConfig(BaseModel):
    """Ordered semantic weighting rules; overlapping rules use last-wins."""

    model_config = ConfigDict(extra="forbid")

    default_weight: float = Field(default=1.0, ge=0)
    rules: list[LossWeightRule] = Field(default_factory=list)
    token_boundary: Literal["overlap", "contained"] = "overlap"
    overlap_strategy: Literal["last_wins"] = "last_wins"
    zero_weight_action: Literal["error", "warn", "allow"] = "error"

    @model_validator(mode="after")
    def _validate_weights(self) -> LossWeightingConfig:
        if not math.isfinite(self.default_weight):
            raise ValueError("default_weight must be finite")
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("loss weighting rule names must be unique")
        return self

    def runtime_dict(self) -> dict[str, Any]:
        return self.model_dump()
