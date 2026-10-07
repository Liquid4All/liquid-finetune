import logging

from liquid_finetune.evaluation.base import Benchmark
from liquid_finetune.evaluation.vlm_benchmarks import (
    VLMGenerationBenchmark,
    VLMLogprobBenchmark,
)

logger = logging.getLogger(__name__)

GENERATION_METRICS = {
    "grounding_iou",
    "grounding_iou_f1",
    "short_answer",
    "mcq_gen",
    "bleu",
    "rouge_l",
}
LOGPROB_METRICS = {"logprob_zero_shot"}

# Keys consumed by the factory — not forwarded to benchmark constructors
_FACTORY_KEYS = {"name", "path", "metric"}


def create_vlm_benchmarks_from_config(
    benchmark_configs: dict,
    processor,
) -> list[Benchmark]:
    """Convert the ``benchmarks:`` YAML section into VLM Benchmark instances."""
    benchmarks_list = benchmark_configs.get("benchmarks", [])
    default_max_new_tokens = benchmark_configs.get("max_new_tokens", 128)
    default_image_root = benchmark_configs.get("image_root")

    result: list[Benchmark] = []
    for bench in benchmarks_list:
        name = bench["name"]
        path = bench["path"]
        metric = bench.get("metric")
        if not metric:
            raise ValueError(f"Benchmark {name!r} is missing required 'metric' field")

        kwargs = {k: v for k, v in bench.items() if k not in _FACTORY_KEYS}
        kwargs.setdefault("image_root", default_image_root)

        if metric in LOGPROB_METRICS:
            # Logprob benchmarks don't generate text — drop max_new_tokens
            logprob_kwargs = {k: v for k, v in kwargs.items() if k != "max_new_tokens"}
            result.append(
                VLMLogprobBenchmark(
                    name=name, path=path, processor=processor, **logprob_kwargs
                )
            )
        elif metric in GENERATION_METRICS:
            kwargs.setdefault("max_new_tokens", default_max_new_tokens)
            result.append(
                VLMGenerationBenchmark(
                    name=name, path=path, processor=processor, metric=metric, **kwargs
                )
            )
        else:
            raise ValueError(
                f"Unknown metric {metric!r} for benchmark {name!r}. "
                f"Available: {sorted(GENERATION_METRICS | LOGPROB_METRICS)}"
            )

    return result
