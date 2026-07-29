from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from leap_finetune.evaluation.backend import HFBackend, InferenceBackend
from leap_finetune.evaluation.base import Benchmark

logger = logging.getLogger(__name__)

# ==== Benchmark Construction ====


def create_benchmarks_from_eval_config(
    evals: Any,
    processor,
    *,
    modality: str,
) -> list[Benchmark]:
    evals_dict = (
        evals.model_dump(exclude_none=True)
        if hasattr(evals, "model_dump")
        else dict(evals or {})
    )
    if modality == "text":
        from leap_finetune.evaluation.llm_config import (
            create_llm_benchmarks_from_config,
        )

        return create_llm_benchmarks_from_config(evals_dict, processor)
    if modality == "vlm":
        from leap_finetune.evaluation.vlm_config import (
            create_vlm_benchmarks_from_config,
        )

        return create_vlm_benchmarks_from_config(evals_dict, processor)
    raise ValueError(f"Unsupported eval modality: {modality!r}")


# ==== Backend Construction ====


def load_eval_processor(model_ref: str, *, modality: str):
    from leap_finetune.checkpointing.model_loading import _resolve_model_id

    resolved_model_ref = _resolve_model_id(model_ref)
    if modality == "text":
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(resolved_model_ref, trust_remote_code=True)
    if modality == "vlm":
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(resolved_model_ref, trust_remote_code=True)
    raise ValueError(f"Unsupported eval modality: {modality!r}")


def create_hf_backend(
    model_ref: str,
    *,
    modality: str,
    model_config: dict[str, Any] | None = None,
) -> tuple[HFBackend, Any]:
    import torch

    from leap_finetune.checkpointing.model_loading import load_model, load_vlm_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if modality == "text":
        model, processor = load_model(
            model_ref,
            model_config=model_config,
            install_memory_efficient_loss=False,
        )
    elif modality == "vlm":
        model, processor = load_vlm_model(model_ref)
    else:
        raise ValueError(f"Unsupported eval modality: {modality!r}")

    model.to(device)
    model.eval()
    return HFBackend(model, processor, device, modality=modality), processor


def create_vllm_backend(model_ref: str, backend_config):
    from leap_finetune.evaluation.backend import VLLMInProcessBackend
    from leap_finetune.checkpointing.model_loading import _resolve_model_id

    cfg = (
        backend_config.model_dump(exclude_none=True)
        if hasattr(backend_config, "model_dump")
        else dict(backend_config or {})
    )
    return VLLMInProcessBackend(
        model_path=_resolve_model_id(model_ref),
        tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
        dtype=str(cfg.get("dtype", "bfloat16")),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.9)),
        max_model_len=cfg.get("max_model_len"),
        quantization=cfg.get("quantization"),
    )


def create_llama_cpp_backend(model_ref: str, backend_config):
    from leap_finetune.evaluation.backend import LlamaCppServerBackend

    cfg = (
        backend_config.model_dump(exclude_none=True)
        if hasattr(backend_config, "model_dump")
        else dict(backend_config or {})
    )
    return LlamaCppServerBackend(
        model_path=model_ref,
        base_url=cfg.get("base_url"),
        model_id=cfg.get("model_id"),
        server_binary=str(cfg.get("server_binary", "llama-server")),
        host=str(cfg.get("host", "127.0.0.1")),
        port=int(cfg.get("port", 8080)),
        n_gpu_layers=int(cfg.get("n_gpu_layers", 999)),
        mmproj=cfg.get("mmproj"),
        server_args=list(cfg.get("server_args", [])),
        startup_timeout=float(cfg.get("startup_timeout", 300.0)),
        request_timeout=float(cfg.get("request_timeout", 600.0)),
        log_path=cfg.get("log_path"),
    )


# ==== Benchmark Execution ====


def _run_one(bench: Benchmark, backend: InferenceBackend) -> dict[str, float]:
    samples = bench.get_samples()
    if not samples:
        logger.warning("[%s] no samples loaded; skipping", bench.name)
        return {}
    result = bench.evaluate_with_backend(backend, samples)
    return {
        f"benchmark/{bench.name}/{metric}": (
            total / result.count if result.count > 0 else 0.0
        )
        for metric, total in result.metrics.items()
    }


def run_benchmarks_with_backend(
    benchmarks: list[Benchmark],
    backend: InferenceBackend,
    *,
    fallback_backend_factory: Callable[[], InferenceBackend] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    fallback_backend: InferenceBackend | None = None

    for bench in benchmarks:
        try:
            bench_metrics = _run_one(bench, backend)
        except NotImplementedError as e:
            if fallback_backend_factory is None:
                logger.warning(
                    "[%s] backend %s does not support this benchmark: %s",
                    bench.name,
                    getattr(backend, "name", type(backend).__name__),
                    e,
                )
                continue

            try:
                if fallback_backend is None:
                    fallback_backend = fallback_backend_factory()
                bench_metrics = _run_one(bench, fallback_backend)
                logger.info("[%s] completed via HF fallback", bench.name)
            except Exception:
                logger.exception("[%s] HF fallback failed", bench.name)
                continue
        except Exception:
            logger.exception("[%s] failed; other benchmarks continue", bench.name)
            continue

        metrics.update(bench_metrics)
        logger.info("[%s] OK: %s", bench.name, bench_metrics)

    return metrics


# ==== Checkpoint Staging ====


def stage_checkpoint_for_eval(
    *,
    model,
    benchmarks: list[Benchmark],
    ckpt_root: Path,
    step: int,
    keep: int,
) -> Path:
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_root / f"step_{step}"

    if ckpt_path.exists():
        shutil.rmtree(ckpt_path, ignore_errors=True)

    existing = sorted(
        ckpt_root.glob("step_*"),
        key=lambda p: int(p.name.removeprefix("step_")) if p.name[5:].isdigit() else 0,
    )
    for stale in existing[: max(0, len(existing) - keep + 1)]:
        shutil.rmtree(stale, ignore_errors=True)

    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(str(ckpt_path))

    for bench in benchmarks:
        processor = getattr(bench, "tokenizer", None) or getattr(
            bench, "processor", None
        )
        if processor is None:
            continue
        try:
            processor.save_pretrained(str(ckpt_path))
        except Exception:
            logger.debug("eval processor save_pretrained failed", exc_info=True)
        break

    return ckpt_path


# ==== Standalone Eval ====


def _load_config(config):
    from leap_finetune.config import EvalRunConfig
    from leap_finetune.config.parser import materialize_eval_config, parse_eval_config

    if isinstance(config, EvalRunConfig):
        return materialize_eval_config(config)
    return materialize_eval_config(parse_eval_config(config))


def run_eval_config(
    config,
    *,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Run configured benchmarks without starting a training job."""
    cfg = _load_config(config)
    model_ref = cfg.model_ref

    if cfg.backend.type == "hf":
        backend, processor = create_hf_backend(
            model_ref,
            modality=cfg.modality,
            model_config=cfg.model_overrides,
        )
    elif cfg.backend.type == "vllm":
        processor = load_eval_processor(model_ref, modality=cfg.modality)
        backend = create_vllm_backend(model_ref, cfg.backend)
    elif cfg.backend.type == "llama_cpp":
        # Backend-driven benchmarks do not need a Transformers processor;
        # llama-server owns chat templating and vision preprocessing.
        processor = None
        backend = create_llama_cpp_backend(model_ref, cfg.backend)
    else:
        raise ValueError(f"Unsupported eval backend: {cfg.backend.type!r}")

    benchmarks = create_benchmarks_from_eval_config(
        cfg.evals,
        processor,
        modality=cfg.modality,
    )
    try:
        metrics = run_benchmarks_with_backend(benchmarks, backend)
    finally:
        try:
            backend.close()
        except Exception:
            pass

    resolved_output_path = (
        Path(output_path or cfg.output_path).expanduser()
        if (output_path or cfg.output_path)
        else None
    )
    if resolved_output_path is not None:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(json.dumps(metrics, indent=2) + "\n")

    return metrics
