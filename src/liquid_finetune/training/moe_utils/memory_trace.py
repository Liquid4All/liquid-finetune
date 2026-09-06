import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import torch

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    dist = None


DEFAULT_TRACE_STEPS = "0,1,2"
TRACE_ENV_VAR = "LIQUID_MEMORY_TRACE"
TRACE_DIR_ENV_VAR = "LIQUID_MEMORY_TRACE_DIR"
TRACE_STEPS_ENV_VAR = "LIQUID_MEMORY_TRACE_STEPS"
TRACE_SYNC_ENV_VAR = "LIQUID_MEMORY_TRACE_SYNCHRONIZE"

_TRACE_FILE: Path | None = None
_TRACE_FRAMEWORK: str | None = None


def memory_trace_enabled() -> bool:
    return os.getenv(TRACE_ENV_VAR, "0") == "1"


def _parse_trace_steps() -> set[int]:
    raw = os.getenv(TRACE_STEPS_ENV_VAR, DEFAULT_TRACE_STEPS).strip()
    if not raw:
        return set()

    steps = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        steps.add(int(piece))
    return steps


def _should_trace_step(step: int | None) -> bool:
    if step is None:
        return True
    return step in _parse_trace_steps()


def _get_rank() -> int:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.getenv("RANK", "0"))


def _get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def init_memory_trace(output_dir: str, framework: str) -> None:
    global _TRACE_FILE, _TRACE_FRAMEWORK

    if not memory_trace_enabled() or _TRACE_FILE is not None:
        return

    trace_root = Path(os.getenv(TRACE_DIR_ENV_VAR, output_dir)) / "memory_traces"
    trace_root.mkdir(parents=True, exist_ok=True)

    rank = _get_rank()
    _TRACE_FILE = trace_root / f"{framework}_rank{rank}.jsonl"
    _TRACE_FRAMEWORK = framework

    write_memory_trace_event(
        "trace_initialized",
        always=True,
        extra={
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "trace_file": str(_TRACE_FILE),
            "trace_steps": sorted(_parse_trace_steps()),
        },
    )


def write_memory_trace_event(
    tag: str,
    step: int | None = None,
    *,
    always: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    if not memory_trace_enabled():
        return
    if _TRACE_FILE is None:
        return
    if not always and not _should_trace_step(step):
        return

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        if os.getenv(TRACE_SYNC_ENV_VAR, "1") == "1":
            torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
    else:  # pragma: no cover
        device = None
        free_bytes = total_bytes = 0
        allocated = reserved = 0
        max_allocated = max_reserved = 0

    payload = {
        "ts": time.time(),
        "framework": _TRACE_FRAMEWORK,
        "tag": tag,
        "step": step,
        "rank": _get_rank(),
        "local_rank": _get_local_rank(),
        "device": device,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "max_allocated_bytes": max_allocated,
        "max_reserved_bytes": max_reserved,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }
    if extra:
        payload.update(extra)

    with _TRACE_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def wrap_optimizer_step(optimizer: Any, get_step) -> Any:
    if not memory_trace_enabled() or optimizer is None:
        return optimizer
    if getattr(optimizer, "_liquid_memory_trace_wrapped", False):
        return optimizer

    original_step = optimizer.step
    empty_cache_after_step = os.getenv(
        "LIQUID_EMPTY_CACHE_AFTER_OPTIMIZER_STEP", ""
    ).lower() in {"1", "true", "yes"}

    def traced_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        write_memory_trace_event("after_optimizer_step", step=get_step())
        if empty_cache_after_step and torch.cuda.is_available():
            torch.cuda.empty_cache()
            write_memory_trace_event(
                "after_optimizer_empty_cache",
                step=get_step(),
                extra={"empty_cache_enabled": True},
            )
        return result

    optimizer.step = traced_step
    optimizer._liquid_memory_trace_wrapped = True
    return optimizer
