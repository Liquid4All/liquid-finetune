import logging

from leap_finetune.evaluation.base import Benchmark
from leap_finetune.evaluation.llm_benchmarks import (
    LLMGenerationBenchmark,
    LLMLogprobBenchmark,
)

logger = logging.getLogger(__name__)

GENERATION_METRICS = {"short_answer", "mcq_gen", "gsm8k", "bleu", "rouge_l"}
LOGPROB_METRICS = {"logprob_zero_shot"}

# Keys consumed by the factory — not forwarded to benchmark constructors
_FACTORY_KEYS = {"name", "path", "metric"}


def create_llm_benchmarks_from_config(
    benchmark_configs: dict,
    tokenizer,
) -> list[Benchmark]:
    """Convert the ``benchmarks:`` YAML section into LLM Benchmark instances."""
    benchmarks_list = benchmark_configs.get("benchmarks", [])
    default_max_new_tokens = benchmark_configs.get("max_new_tokens", 128)

    result: list[Benchmark] = []
    for bench in benchmarks_list:
        name = bench["name"]
        path = bench["path"]
        metric = bench.get("metric")
        if not metric:
            raise ValueError(f"Benchmark {name!r} is missing required 'metric' field")

        kwargs = {k: v for k, v in bench.items() if k not in _FACTORY_KEYS}

        if metric in LOGPROB_METRICS:
            # Logprob benchmarks don't generate text — drop max_new_tokens
            logprob_kwargs = {k: v for k, v in kwargs.items() if k != "max_new_tokens"}
            result.append(
                LLMLogprobBenchmark(
                    name=name, path=path, tokenizer=tokenizer, **logprob_kwargs
                )
            )
        elif metric in GENERATION_METRICS:
            kwargs.setdefault("max_new_tokens", default_max_new_tokens)
            result.append(
                LLMGenerationBenchmark(
                    name=name,
                    path=path,
                    tokenizer=tokenizer,
                    metric=metric,
                    **kwargs,
                )
            )
        else:
            raise ValueError(
                f"Unknown metric {metric!r} for benchmark {name!r}. "
                f"Available: {sorted(GENERATION_METRICS | LOGPROB_METRICS)}"
            )

    return result
