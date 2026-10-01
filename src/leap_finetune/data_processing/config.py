from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PreprocessingStage = Literal[
    "raw_pre_normalize",
    "normalized_pre_validate",
    "post_validate_pre_tokenize",
]


class PreprocessingOperation(BaseModel):
    """One lazy Ray Data transform loaded from an importable callable."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["map", "map_batches", "filter", "flat_map"]
    callable: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    fit_callable: str | None = None
    fit_kwargs: dict[str, Any] = Field(default_factory=dict)
    stage: PreprocessingStage = "raw_pre_normalize"
    batch_size: int | None = Field(default=None, gt=0)
    num_cpus: float | None = Field(default=None, gt=0)
    batch_format: Literal["pyarrow", "pandas", "numpy"] = "pyarrow"
    fingerprint: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_operation(self) -> PreprocessingOperation:
        if ":" not in self.callable:
            raise ValueError(
                "preprocessing callable must be 'module:function' or "
                "'/path/to/file.py:function'"
            )
        if self.fit_callable is not None and ":" not in self.fit_callable:
            raise ValueError(
                "preprocessing fit_callable must be 'module:function' or "
                "'/path/to/file.py:function'"
            )
        if self.fit_callable is None and self.fit_kwargs:
            raise ValueError("fit_kwargs requires fit_callable")
        if self.fit_callable is not None and "artifact" in self.kwargs:
            raise ValueError("kwargs.artifact is reserved for fit_callable output")
        if self.op != "map_batches" and self.batch_size is not None:
            raise ValueError("batch_size is only valid for op='map_batches'")
        if self.op != "map_batches" and self.batch_format != "pyarrow":
            raise ValueError("batch_format is only valid for op='map_batches'")
        return self

    def runtime_dict(self) -> dict[str, Any]:
        return self.model_dump()
