import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from liquid_finetune.evaluation.backend import InferenceBackend

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Aggregated evaluation result. The callback all-reduces each metric across ranks.

    ``metrics`` is a dict of accumulated scores (e.g. {"accuracy": 42.0, "f1": 38.5}).
    The callback divides each by ``count`` to get averages.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    count: int = 0


class Benchmark(ABC):
    """Base class for all benchmarks.

    Subclass and implement:
      - ``load_samples()`` — return evaluation data (called once, cached).
      - ``score_sample(model, sample, device)`` — score a single sample (sync path).
      - Optionally ``evaluate_with_backend(backend, samples)`` for async paths.
    """

    def __init__(self, name: str):
        self.name = name
        self._samples: list | None = None

    @abstractmethod
    def load_samples(self) -> list:
        """Return all evaluation samples. Called once, then cached."""
        ...

    @abstractmethod
    def score_sample(self, model, sample: dict, device) -> float:
        """Score a single sample. Return a float (e.g. 0.0 or 1.0)."""
        ...

    def evaluate(self, model, samples: list, device) -> BenchmarkResult:
        """Score the model on ``samples`` (already sharded to this rank).

        Skips failed samples and excludes them from the count so they
        don't deflate the average.
        """
        total_score = 0.0
        count = 0
        for sample in samples:
            try:
                total_score += self.score_sample(model, sample, device)
                count += 1
            except Exception:
                logger.warning(
                    "[%s] Failed on sample %s",
                    self.name,
                    sample.get("id", count),
                    exc_info=True,
                )
        return BenchmarkResult(metrics={"score": total_score}, count=count)

    def evaluate_with_backend(
        self, backend: "InferenceBackend", samples: list
    ) -> BenchmarkResult:
        """Score ``samples`` via an :class:`InferenceBackend`. Subclasses
        opt in by overriding: build batched requests, dispatch to the
        backend, score the responses with their existing scorer.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support evaluate_with_backend; "
            "implement it or use mode: sync."
        )

    def get_samples(self) -> list:
        if self._samples is None:
            self._samples = self.load_samples()
        return self._samples


EvalResult = BenchmarkResult
Eval = Benchmark
