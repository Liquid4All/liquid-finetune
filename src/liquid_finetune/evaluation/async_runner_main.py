from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("liquid_finetune.async_eval")

# ==== Async Sidecar Runner ====


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async eval sidecar runner.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--benchmark-configs", required=True, type=Path)
    p.add_argument("--modality", required=True, choices=["text", "vlm"])
    p.add_argument("--trigger-step", required=True, type=int)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--vllm-gpus", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--wandb-run-id", default=None)
    p.add_argument("--wandb-project", default=None)
    return p.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _load_benchmarks(args: argparse.Namespace):
    """Load tokenizer/processor from the checkpoint dir, then build benchmarks."""
    with args.benchmark_configs.open() as f:
        bench_configs = json.load(f)

    from liquid_finetune.evaluation.runner import (
        create_benchmarks_from_eval_config,
        load_eval_processor,
    )

    processor = load_eval_processor(str(args.checkpoint), modality=args.modality)
    return create_benchmarks_from_eval_config(
        bench_configs,
        processor,
        modality=args.modality,
    )


def _log_to_wandb(args: argparse.Namespace, results: dict[str, float]) -> None:
    """Attach to the training run and append the benchmark point.

    We do NOT pass ``step=trigger_step``. The sidecar resumes the run from a
    separate process minutes after the trigger fired, by which point the live
    run's internal ``_step`` has advanced well past trigger_step — often far
    past it, since GRPO commits many times per training step. A ``step=`` in
    the past is a backwards write and wandb silently drops it. Logging without
    ``step=`` appends at the current (forward) ``_step``, so the point always
    lands. The originating training step rides along as plain data fields —
    ``train/global_step`` (what training dashboards use) and ``benchmark/step``
    (a clean alias) — so benchmark panels align on the trainer's step axis.
    """
    if not args.wandb_run_id:
        logger.info("--wandb-run-id not set; skipping wandb log")
        return

    try:
        import wandb

        init_kwargs = {
            "id": args.wandb_run_id,
            "resume": "allow",
            # 90s default times out on busy clusters; bump for robustness.
            "settings": wandb.Settings(init_timeout=300),
        }
        if args.wandb_project:
            init_kwargs["project"] = args.wandb_project

        wandb.init(**init_kwargs)
        # Pin every benchmark key to benchmark/step BEFORE first log;
        # define_metric called after a key is logged doesn't rebind it.
        # The trailing benchmark/* glob covers keys we didn't enumerate
        # (wandb only accepts suffix globs — benchmark/*/* is rejected).
        try:
            wandb.define_metric("benchmark/step")
            for key in sorted(results.keys() if results else ()):
                if key.startswith("benchmark/"):
                    wandb.define_metric(key, step_metric="benchmark/step")
            wandb.define_metric("benchmark/*", step_metric="benchmark/step")
        except Exception as e:
            logger.warning("wandb.define_metric failed; axis-pin skipped (%s)", e)
        if not results:
            logger.info("no benchmark results to log")
        else:
            payload = dict(results)
            payload["train/global_step"] = args.trigger_step
            payload["benchmark/step"] = args.trigger_step
            wandb.log(payload)
            logger.info(
                "logged %d metrics to wandb (benchmark/step=%d)",
                len(results),
                args.trigger_step,
            )
        wandb.finish()
    except ImportError:
        logger.warning("wandb not installed; skipping log")
    except Exception:
        logger.exception("wandb logging failed")


def main() -> int:
    _setup_logging()
    args = _parse_args()

    logger.info(
        "async eval starting: step=%d ckpt=%s modality=%s vllm_gpus=%d",
        args.trigger_step,
        args.checkpoint,
        args.modality,
        args.vllm_gpus,
    )

    try:
        benchmarks = _load_benchmarks(args)
        logger.info("loaded %d benchmarks", len(benchmarks))
    except Exception:
        logger.exception("benchmark construction failed; aborting")
        return 2

    if not benchmarks:
        logger.warning("no benchmarks to run; exiting cleanly")
        return 0

    # vLLM EngineCore CUDA init occasionally races with the parent training
    # job's CUDA state propagation (especially at on_train_begin, when Ray
    # workers are still settling). Retry the spawn a few times with backoff
    # before giving up — the eval node is healthy, we just hit a transient.
    from liquid_finetune.evaluation.runner import (
        create_hf_backend,
        create_vllm_backend,
        run_benchmarks_with_backend,
    )
    import time

    _MAX_VLLM_RETRIES = 3
    _RETRY_BACKOFF_S = 30
    backend = None
    for attempt in range(1, _MAX_VLLM_RETRIES + 1):
        try:
            backend = create_vllm_backend(
                str(args.checkpoint),
                {
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "dtype": args.dtype,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_model_len": args.max_model_len,
                },
            )
            break
        except Exception as e:
            transient = "CUDA unknown error" in str(
                e
            ) or "Engine core initialization failed" in str(e)
            if not transient or attempt == _MAX_VLLM_RETRIES:
                logger.exception(
                    "vLLM startup failed (attempt %d/%d, transient=%s)",
                    attempt,
                    _MAX_VLLM_RETRIES,
                    transient,
                )
                return 3
            logger.warning(
                "vLLM startup transient failure (attempt %d/%d); "
                "sleeping %ds then retrying. Error: %s",
                attempt,
                _MAX_VLLM_RETRIES,
                _RETRY_BACKOFF_S,
                str(e)[:200],
            )
            time.sleep(_RETRY_BACKOFF_S)

    fallback_backend = None

    def fallback_factory():
        nonlocal fallback_backend
        if fallback_backend is None:
            fallback_backend, _ = create_hf_backend(
                str(args.checkpoint),
                modality=args.modality,
            )
        return fallback_backend

    try:
        results = run_benchmarks_with_backend(
            benchmarks,
            backend,
            fallback_backend_factory=fallback_factory,
        )
    finally:
        try:
            backend.close()
        except Exception:
            pass
        if fallback_backend is not None:
            try:
                fallback_backend.close()
            except Exception:
                pass

    _log_to_wandb(args, results)
    logger.info("async eval done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
