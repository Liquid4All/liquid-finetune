from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


# === Local vLLM server lifecycle and resource planning ===
#
# The driver resolves local GPU splits before Ray starts, launches vLLM
# server processes on reserved GPUs, waits for /health, then passes endpoints
# to the relevant runtime config. The caller owns the server protocol; this
# module handles local process lifecycle and resource planning.


@dataclass
class VLLMServerHandle:
    """Handle to a running ``trl vllm-serve`` subprocess.

    Carries enough info to (a) plumb the server endpoint through to
    GRPOConfig on the training workers, and (b) cleanly tear down the
    server at the end of training.
    """

    process: subprocess.Popen
    host: str
    port: int
    tensor_parallel_size: int
    server_gpu_ids: list[int]
    log_path: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        """Terminate the subprocess if it is still running."""
        _terminate_server(self.process)


@dataclass(frozen=True)
class VLLMRolloutResourcePlan:
    """GPU/resource plan for local GRPO server-mode rollouts.

    Users configure counts, not ids. The ids here are an internal contiguous
    assignment relative to the driver's current ``CUDA_VISIBLE_DEVICES``:
    vLLM gets the first ``server_gpus`` devices, the optional judge gets the
    next ``judge_gpus`` devices, and Ray training gets the next
    ``training_gpus`` devices.
    """

    total_gpus: int
    server_gpu_ids: list[int]
    judge_gpu_ids: list[int]
    training_gpu_ids: list[int]
    visible_device_tokens: list[str]
    num_training_workers: int

    @property
    def launches_local_server(self) -> bool:
        return bool(self.server_gpu_ids)

    @property
    def launches_local_judge(self) -> bool:
        return bool(self.judge_gpu_ids)

    @property
    def server_cuda_visible_devices(self) -> str:
        return ",".join(
            self.visible_device_tokens[gpu_id] for gpu_id in self.server_gpu_ids
        )

    @property
    def judge_cuda_visible_devices(self) -> str:
        return ",".join(
            self.visible_device_tokens[gpu_id] for gpu_id in self.judge_gpu_ids
        )

    @property
    def training_cuda_visible_devices(self) -> str:
        return ",".join(
            self.visible_device_tokens[gpu_id] for gpu_id in self.training_gpu_ids
        )

    @property
    def uses_custom_training_visibility(self) -> bool:
        return self.training_gpu_ids != list(range(self.total_gpus))

    @property
    def resources_per_worker(self) -> dict[str, float]:
        return {"GPU": 1.0}


def resolve_vllm_rollout_plan(
    total_gpus: int,
    grpo_rollout_cfg: dict | None,
    *,
    vllm_mode: str,
    is_multi_node: bool,
    reserve_judge: bool = False,
) -> VLLMRolloutResourcePlan:
    """Resolve a local vLLM server/training GPU split.

    Supported ``grpo_rollout`` keys:

    * ``server_gpus``: number of local GPUs to reserve for ``trl vllm-serve``.
    * ``dedicated_gpus``: legacy alias for ``server_gpus``.
    * ``training_gpus``: optional number of GPUs for Ray training. Defaults to
      all GPUs not reserved for vLLM. When set without ``server_gpus``, the
      remaining GPUs are reserved for the local vLLM server.
    * ``judge_gpus``: number of local GPUs to reserve for the judge LLM server.
      Defaults to 1 when a local judge is enabled.
    """
    cfg = grpo_rollout_cfg or {}
    if total_gpus < 1:
        return VLLMRolloutResourcePlan(
            total_gpus=total_gpus,
            server_gpu_ids=[],
            judge_gpu_ids=[],
            training_gpu_ids=[],
            visible_device_tokens=[],
            num_training_workers=0,
        )

    all_gpu_ids = list(range(total_gpus))
    visible_device_tokens = _visible_device_tokens(total_gpus)
    if vllm_mode != "server" and not reserve_judge:
        return VLLMRolloutResourcePlan(
            total_gpus=total_gpus,
            server_gpu_ids=[],
            judge_gpu_ids=[],
            training_gpu_ids=all_gpu_ids,
            visible_device_tokens=visible_device_tokens,
            num_training_workers=total_gpus,
        )

    if vllm_mode != "server":
        server_gpus = 0
        judge_gpus = _resolve_judge_gpu_count(total_gpus, cfg, reserve_judge)
        training_gpus = _resolve_training_gpu_count(
            total_gpus=total_gpus,
            server_gpus=server_gpus,
            judge_gpus=judge_gpus,
            cfg=cfg,
        )
    else:
        server_gpus, judge_gpus, training_gpus = _resolve_server_training_gpu_counts(
            total_gpus,
            cfg,
            reserve_judge=reserve_judge,
        )

    if is_multi_node and (server_gpus or judge_gpus or "training_gpus" in cfg):
        raise NotImplementedError(
            "Local GRPO GPU partitioning is single-node only. For "
            "multi-node GRPO, use vllm_mode: colocate or point "
            "vllm_server_base_url and judge base_url at externally managed "
            "servers without setting server_gpus/judge_gpus/training_gpus."
        )

    server_gpu_ids = all_gpu_ids[:server_gpus]
    judge_start = server_gpus
    judge_gpu_ids = all_gpu_ids[judge_start : judge_start + judge_gpus]
    train_start = server_gpus + judge_gpus
    training_gpu_ids = all_gpu_ids[train_start : train_start + training_gpus]

    return VLLMRolloutResourcePlan(
        total_gpus=total_gpus,
        server_gpu_ids=server_gpu_ids,
        judge_gpu_ids=judge_gpu_ids,
        training_gpu_ids=training_gpu_ids,
        visible_device_tokens=visible_device_tokens,
        num_training_workers=training_gpus,
    )


def plan_gpu_split(
    total_gpus: int, grpo_rollout_cfg: dict
) -> tuple[list[int], list[int]]:
    """Decide which GPUs go to training vs vLLM.

    Training uses the *trailing* GPUs so the lowest-indexed devices stay
    with vLLM, a common convention that keeps the training process on
    a contiguous device range after we set ``CUDA_VISIBLE_DEVICES``.

    Returns ``(vllm_gpu_ids, train_gpu_ids)`` as local visible-device ordinals.
    """
    plan = resolve_vllm_rollout_plan(
        total_gpus,
        grpo_rollout_cfg,
        vllm_mode="server",
        is_multi_node=False,
    )
    return plan.server_gpu_ids, plan.training_gpu_ids


def _resolve_server_training_gpu_counts(
    total_gpus: int,
    cfg: dict,
    *,
    reserve_judge: bool,
) -> tuple[int, int, int]:
    judge_gpus = _resolve_judge_gpu_count(total_gpus, cfg, reserve_judge)
    server_gpus = _resolve_explicit_server_gpu_count(total_gpus, cfg)
    if server_gpus is None and "training_gpus" in cfg:
        training_gpus = _coerce_gpu_count(cfg["training_gpus"], "training_gpus")
        if training_gpus < 1:
            raise ValueError(
                "GRPO rollout plan leaves no GPUs for training. Set "
                "grpo_rollout.training_gpus >= 1."
            )
        if training_gpus > total_gpus:
            raise ValueError(
                f"grpo_rollout.training_gpus={training_gpus} exceeds available "
                f"GPUs ({total_gpus})."
            )
        if training_gpus + judge_gpus >= total_gpus:
            raise ValueError(
                "grpo_rollout.training_gpus leaves no GPUs for the local vLLM "
                "server after reserving judge_gpus. Reserve fewer training GPUs, "
                "or set server_gpus: 0 and provide an externally managed vLLM "
                "server."
            )
        return total_gpus - training_gpus - judge_gpus, judge_gpus, training_gpus

    server_gpus = server_gpus or 0
    training_gpus = _resolve_training_gpu_count(
        total_gpus=total_gpus,
        server_gpus=server_gpus,
        judge_gpus=judge_gpus,
        cfg=cfg,
    )
    return server_gpus, judge_gpus, training_gpus


def _resolve_explicit_server_gpu_count(total_gpus: int, cfg: dict) -> int | None:
    if "server_gpus" in cfg:
        key = "server_gpus"
        raw = cfg[key]
    elif "dedicated_gpus" in cfg:
        key = "dedicated_gpus"
        raw = cfg[key]
    else:
        return None

    server_gpus = _coerce_gpu_count(raw, key)
    if server_gpus > total_gpus:
        raise ValueError(
            f"grpo_rollout.{key}={server_gpus} exceeds available GPUs ({total_gpus})."
        )
    return server_gpus


def _resolve_training_gpu_count(
    *,
    total_gpus: int,
    server_gpus: int,
    judge_gpus: int,
    cfg: dict,
) -> int:
    remaining = total_gpus - server_gpus - judge_gpus
    raw = cfg.get("training_gpus")
    training_gpus = (
        remaining if raw is None else _coerce_gpu_count(raw, "training_gpus")
    )
    if training_gpus < 1:
        raise ValueError(
            "GRPO rollout plan leaves no GPUs for training. Set "
            "grpo_rollout.training_gpus >= 1 or reserve fewer server_gpus/judge_gpus."
        )
    if server_gpus + judge_gpus + training_gpus > total_gpus:
        raise ValueError(
            "GRPO rollout plan requests "
            f"{server_gpus} server GPU(s) + {judge_gpus} judge GPU(s) + "
            f"{training_gpus} training GPU(s), "
            f"but only {total_gpus} GPU(s) are visible."
        )
    return training_gpus


def _resolve_judge_gpu_count(total_gpus: int, cfg: dict, reserve_judge: bool) -> int:
    if not reserve_judge:
        return 0

    raw = cfg.get("judge_gpus")
    judge_gpus = _coerce_gpu_count(raw, "judge_gpus") if raw is not None else 1
    if reserve_judge and judge_gpus < 1:
        raise ValueError(
            "rewards.judge requires at least one local judge GPU unless "
            "`rewards.judge.base_url` points at an external judge server."
        )
    if judge_gpus > total_gpus:
        raise ValueError(
            f"grpo_rollout.judge_gpus={judge_gpus} exceeds available GPUs "
            f"({total_gpus})."
        )
    return judge_gpus


def _coerce_gpu_count(value, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"grpo_rollout.{key} must be an integer, got {value!r}.")
    count = int(value)
    if count < 0:
        raise ValueError(f"grpo_rollout.{key} must be >= 0, got {count}.")
    return count


def _visible_device_tokens(total_gpus: int) -> list[str]:
    raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible_devices:
        tokens = [
            token.strip() for token in raw_visible_devices.split(",") if token.strip()
        ]
        if len(tokens) >= total_gpus:
            return tokens[:total_gpus]
    return [str(gpu_id) for gpu_id in range(total_gpus)]


def resolve_server_host(host: str | None) -> str:
    """Resolve a ``vllm_server_host`` YAML value.

    Special values:

    * ``None`` / ``"auto"``: the current node's hostname, resolved via
      ``SLURMD_NODENAME`` (if running under SLURM) or ``socket.gethostname()``.
      On a single-node dev box this resolves to the machine's own hostname,
      which the workers can reach over loopback.
    * Any other string: used verbatim.
    """
    if not host or host == "auto":
        return os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    return host


def launch_vllm_server(
    model_id: str,
    vllm_gpu_ids: list[int],
    grpo_rollout_cfg: dict,
    host: str = "0.0.0.0",
    port: int = 8000,
    startup_timeout: float = 300.0,
    vllm_cuda_visible_devices: str | None = None,
    log_path: str | os.PathLike | None = None,
) -> VLLMServerHandle:
    """Launch ``trl vllm-serve`` on the reserved GPUs and wait for /health.

    Args:
        model_id: The model path or HF id to pass to vLLM. Should match the
            model the trainer is about to load.
        vllm_gpu_ids: Local visible-device ordinals reserved for vLLM.
        grpo_rollout_cfg: The ``grpo_rollout:`` YAML block. Recognized keys:
            ``tensor_parallel_size`` (default = len(vllm_gpu_ids)), ``dtype``
            (default "bfloat16"), ``gpu_memory_utilization`` (default 0.9),
            ``max_model_len`` (default None, vLLM infers).
        host: Bind address for the vLLM server. "0.0.0.0" makes it reachable
            from other training processes on the same host.
        port: TCP port.
        startup_timeout: Seconds to wait for /health to return 200 before
            giving up.
        vllm_cuda_visible_devices: Explicit CUDA_VISIBLE_DEVICES value for the
            server subprocess. This preserves an existing parent
            CUDA_VISIBLE_DEVICES mapping while still letting users configure
            only GPU counts.
        log_path: Optional path for vLLM server stdout/stderr. If omitted,
            server output is discarded.

    Returns:
        A ``VLLMServerHandle`` with the running subprocess. An ``atexit`` hook
        is registered to terminate the process on interpreter shutdown.
    """
    if not vllm_gpu_ids:
        raise ValueError("launch_vllm_server called with empty vllm_gpu_ids")

    if shutil.which("trl") is None:
        raise RuntimeError(
            "`trl` CLI not found on PATH. vLLM server mode requires `trl vllm-serve`. "
            "Make sure trl is installed: `uv sync`."
        )

    tensor_parallel_size = int(
        grpo_rollout_cfg.get("tensor_parallel_size", len(vllm_gpu_ids))
    )
    if tensor_parallel_size > len(vllm_gpu_ids):
        raise ValueError(
            f"grpo_rollout.tensor_parallel_size={tensor_parallel_size} exceeds "
            f"grpo_rollout.server_gpus={len(vllm_gpu_ids)}."
        )
    dtype = str(grpo_rollout_cfg.get("dtype", "bfloat16"))
    gpu_memory_utilization = float(grpo_rollout_cfg.get("gpu_memory_utilization", 0.9))
    max_model_len = grpo_rollout_cfg.get("max_model_len")

    cmd: list[Any] = [
        "trl",
        "vllm-serve",
        "--model",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--dtype",
        dtype,
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        cmd += ["--max_model_len", str(max_model_len)]

    # Isolate the server from training GPUs via CUDA_VISIBLE_DEVICES. The
    # training process gets the complementary set from the driver
    # (see trainer.py::ray_trainer).
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = vllm_cuda_visible_devices or ",".join(
        str(g) for g in vllm_gpu_ids
    )

    logger.info(
        "Launching vLLM server on GPU(s) %s with CUDA_VISIBLE_DEVICES=%s%s: %s",
        vllm_gpu_ids,
        env["CUDA_VISIBLE_DEVICES"],
        f" (logs: {log_path})" if log_path else "",
        " ".join(str(x) for x in cmd),
    )

    log_file = None
    stdout = subprocess.DEVNULL
    resolved_log_path = os.fspath(log_path) if log_path else None
    if resolved_log_path:
        os.makedirs(os.path.dirname(resolved_log_path), exist_ok=True)
        log_file = open(resolved_log_path, "a", buffering=1)
        stdout = log_file

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
    )
    if log_file is not None:
        log_file.close()

    handle = VLLMServerHandle(
        process=process,
        host=host if host not in ("0.0.0.0", "") else socket.gethostname(),
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        server_gpu_ids=vllm_gpu_ids,
        log_path=resolved_log_path,
    )

    # Register cleanup BEFORE waiting for health so a failed startup still
    # gets its process terminated.
    atexit.register(_terminate_server, process)

    wait_for_vllm_health(
        handle.base_url,
        timeout=startup_timeout,
        process=process,
        log_path=resolved_log_path,
    )
    logger.info("vLLM server healthy at %s (pid=%d)", handle.base_url, process.pid)
    return handle


def wait_for_vllm_health(
    base_url: str,
    timeout: float,
    process: subprocess.Popen,
    log_path: str | None = None,
) -> None:
    """Poll ``<base_url>/health`` until 200 or timeout."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        # If the subprocess died, bail out with its exit code.
        if process.poll() is not None:
            msg = (
                "vLLM server process exited before becoming healthy "
                f"(exit code {process.returncode})."
            )
            raise RuntimeError(_with_server_log_tail(msg, log_path))
        try:
            resp = requests.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code == 200:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(2.0)

    # Timed out; clean up the process so we don't leak it.
    _terminate_server(process)
    msg = f"vLLM server at {base_url} did not become healthy within {timeout}s"
    if last_err is not None:
        msg += f" (last error: {last_err})"
    raise RuntimeError(_with_server_log_tail(msg, log_path))


def _with_server_log_tail(msg: str, log_path: str | None, max_lines: int = 80) -> str:
    """Append the tail of the vLLM server log to startup errors."""
    if not log_path:
        return msg
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except OSError as e:
        return f"{msg}\nCould not read vLLM server log {log_path}: {e}"

    if not lines:
        return f"{msg}\nvLLM server log is empty: {log_path}"

    tail = "".join(lines[-max_lines:]).rstrip()
    return f"{msg}\nLast vLLM server log lines from {log_path}:\n{tail}"


def _terminate_server(process: subprocess.Popen) -> None:
    """Best-effort shutdown of the vLLM server subprocess."""
    if process.poll() is not None:
        return  # already exited
    try:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not exit on SIGTERM, sending SIGKILL")
            process.kill()
            process.wait(timeout=5)
    except Exception as e:  # pragma: no cover - best effort cleanup
        logger.warning("Error terminating vLLM server: %s", e)
