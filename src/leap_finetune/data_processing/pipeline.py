from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from .config import PreprocessingOperation, PreprocessingStage
from .importing import load_callable


@dataclass(frozen=True)
class RecipeContext:
    """Stable metadata available to preprocessing callables on every worker."""

    split: str
    seed: int
    stage: PreprocessingStage
    operation_index: int


def _accepts_context(function) -> bool:
    signature = inspect.signature(function)
    return "context" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


class _LazyCallable:
    """Resolve the callable on each Ray worker, not only on the driver."""

    def __init__(
        self,
        reference: str,
        kwargs: dict[str, Any],
        context: RecipeContext,
    ) -> None:
        self.reference = reference
        self.kwargs = kwargs
        self.context = context
        self._function = None
        self._pass_context = False

    def __call__(self, value):
        if self._function is None:
            self._function = load_callable(self.reference)
            self._pass_context = _accepts_context(self._function)
        kwargs = dict(self.kwargs)
        if self._pass_context:
            kwargs["context"] = self.context
        return self._function(value, **kwargs)


def _runtime_operation(value: PreprocessingOperation | dict) -> PreprocessingOperation:
    if isinstance(value, PreprocessingOperation):
        return value
    return PreprocessingOperation.model_validate(value)


def validate_preprocessing(operations: list[PreprocessingOperation | dict]) -> None:
    """Fail on the driver before Ray schedules work, then re-import on workers."""
    for raw_operation in operations:
        operation = _runtime_operation(raw_operation)
        load_callable(operation.callable)
        if operation.fit_callable:
            load_callable(operation.fit_callable)


def apply_preprocessing(
    dataset,
    operations: list[PreprocessingOperation | dict],
    *,
    stage: PreprocessingStage,
    split: str,
    seed: int,
    report: list[dict[str, Any]] | None = None,
):
    """Apply one stage of a preprocessing graph lazily through Ray Data."""
    for index, raw_operation in enumerate(operations):
        operation = _runtime_operation(raw_operation)
        if operation.stage != stage:
            continue

        before_count = dataset.count() if report is not None else None
        before_schema = str(dataset.schema()) if report is not None else None

        context = RecipeContext(
            split=split,
            seed=seed,
            stage=stage,
            operation_index=index,
        )
        transform_kwargs = dict(operation.kwargs)
        if operation.fit_callable:
            fit_function = load_callable(operation.fit_callable)
            fit_kwargs = dict(operation.fit_kwargs)
            if _accepts_context(fit_function):
                fit_kwargs["context"] = context
            transform_kwargs["artifact"] = fit_function(dataset, **fit_kwargs)
        transform = _LazyCallable(
            operation.callable,
            transform_kwargs,
            context,
        )
        remote_args = {}
        if operation.num_cpus is not None:
            remote_args["num_cpus"] = operation.num_cpus

        if operation.op == "map":
            dataset = dataset.map(transform, **remote_args)
        elif operation.op == "filter":
            dataset = dataset.filter(transform, **remote_args)
        elif operation.op == "flat_map":
            dataset = dataset.flat_map(transform, **remote_args)
        elif operation.op == "map_batches":
            batch_args = {
                "batch_format": operation.batch_format,
                **remote_args,
            }
            if operation.batch_size is not None:
                batch_args["batch_size"] = operation.batch_size
            dataset = dataset.map_batches(transform, **batch_args)
        else:  # pragma: no cover - Pydantic rejects this before execution.
            raise ValueError(f"Unsupported preprocessing operation: {operation.op}")

        if report is not None:
            after_count = dataset.count()
            report.append(
                {
                    "split": split,
                    "operation_index": index,
                    "stage": stage,
                    "op": operation.op,
                    "callable": operation.callable,
                    "fit_callable": operation.fit_callable,
                    "input_schema": before_schema,
                    "output_schema": str(dataset.schema()),
                    "input_rows": before_count,
                    "output_rows": after_count,
                    "row_delta": after_count - before_count,
                }
            )
    return dataset
