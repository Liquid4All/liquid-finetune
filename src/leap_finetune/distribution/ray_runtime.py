import os
import shutil
from pathlib import Path

import psutil


def _is_rocm_torch() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.version, "hip", None))


def normalize_visible_devices() -> None:
    """Normalize accelerator visibility for the active Torch backend."""
    if not _is_rocm_torch():
        # Some mixed clusters expose ROCR_VISIBLE_DEVICES even for CUDA jobs.
        # Do not let that make Ray workers look like ROCm processes.
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        return

    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    if "HIP_VISIBLE_DEVICES" not in os.environ and rocr_visible is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = rocr_visible

    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def normalize_uv_project_path(base_dir: str | os.PathLike | None = None) -> None:
    """Make UV_PROJECT absolute before Ray changes process working dirs."""
    uv_project = os.environ.get("UV_PROJECT")
    if not uv_project:
        return

    project_path = Path(uv_project).expanduser()
    if project_path.is_absolute():
        return

    base_path = Path(base_dir) if base_dir is not None else Path.cwd()
    absolute_path = (base_path / project_path).resolve()
    if absolute_path.exists():
        os.environ["UV_PROJECT"] = str(absolute_path)


def worker_process_setup_hook() -> None:
    """Configure logging and cache directories when Ray starts a worker process."""
    import logging
    import warnings

    normalize_uv_project_path()
    normalize_visible_devices()
    patch_ray_rocm_torch_device_helpers()

    logging.getLogger("ray.data").setLevel(logging.ERROR)
    logging.getLogger("ray.train").setLevel(logging.ERROR)
    logging.getLogger("ray").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    for key in ("TMPDIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
        path = os.environ.get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def patch_ray_rocm_torch_device_helpers() -> None:
    """Make Ray Train's device helper compatible with per-worker ROCm visibility.

    Ray 2.51 narrows ``HIP_VISIBLE_DEVICES`` to one physical GPU per worker, but
    Ray Train v2's torch context still selects ``cuda:<local_rank>``. For ranks
    greater than zero that is out of range inside a single-visible-GPU process.
    Within the narrowed ROCm process, the assigned accelerator is always
    addressable as ``cuda:0``.
    """
    if not _is_rocm_torch():
        return

    if not is_single_visible_rocm_worker():
        return

    import torch

    def _get_rocm_worker_devices():
        if torch.cuda.is_available():
            return [torch.device("cuda:0")]
        return [torch.device("cpu")]

    _get_rocm_worker_devices._leap_rocm_patch = True  # type: ignore[attr-defined]

    def _get_rocm_worker_device():
        return _get_rocm_worker_devices()[0]

    _get_rocm_worker_device._leap_rocm_patch = True  # type: ignore[attr-defined]

    try:
        import ray.air._internal.torch_utils as torch_utils

        torch_utils.get_devices = _get_rocm_worker_devices
    except ImportError:
        pass

    try:
        import ray.train.torch as ray_train_torch

        ray_train_torch.get_devices = _get_rocm_worker_devices
        ray_train_torch.get_device = _get_rocm_worker_device
    except ImportError:
        pass

    try:
        import ray.train.torch.train_loop_utils as train_loop_utils

        train_loop_utils.get_devices = _get_rocm_worker_devices
        train_loop_utils.get_device = _get_rocm_worker_device
    except ImportError:
        pass

    try:
        import ray.train.v2.torch.train_loop_utils as v2_train_loop_utils

        v2_train_loop_utils.get_devices = _get_rocm_worker_devices
        v2_train_loop_utils.get_device = _get_rocm_worker_device
        v2_train_loop_utils.get_devices_distributed = _get_rocm_worker_devices
    except ImportError:
        pass


def get_ray_env_vars(ray_temp_dir: str) -> dict[str, str]:
    """Environment variables passed to Ray workers."""
    normalize_uv_project_path()
    normalize_visible_devices()

    torch_extensions_dir = os.path.join(ray_temp_dir, "torch_extensions")
    triton_cache_dir = os.path.join(ray_temp_dir, "triton_cache")
    env_vars = {
        "TMPDIR": ray_temp_dir,
        "TEMP": ray_temp_dir,
        "TMP": ray_temp_dir,
        "TORCH_EXTENSIONS_DIR": torch_extensions_dir,
        "TRITON_CACHE_DIR": triton_cache_dir,
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "RAY_DISABLE_IMPORT_WARNING": "1",
        "RAY_memory_monitor_refresh_ms": "0",
        "RAY_DATA_DISABLE_PROGRESS_BARS": "1",
        "RAY_IGNORE_UNHANDLED_ERRORS": "1",
        "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": os.environ.get(
            "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1"
        ),
        "RAY_LOG_TO_DRIVER": "0",
        "RAY_DEDUP_LOGS": "1",
        "RAY_DEDUP_LOGS_SKIP_REGEX": r"SplitCoordinator|ProcessGroupNCCL|object.store",
    }

    for key in (
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "NCCL_SOCKET_FAMILY",
        "HIP_VISIBLE_DEVICES",
        "LEAP_JUDGE_LLM_CONFIG",
    ):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key
        if wandb_mode:
            env_vars["WANDB_MODE"] = wandb_mode
    else:
        env_vars["WANDB_MODE"] = wandb_mode if wandb_mode else "offline"

    return env_vars


def is_single_visible_rocm_worker() -> bool:
    """Return true when a ROCm worker process can address only cuda:0."""
    if not _is_rocm_torch():
        return False
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if not hip_visible:
        return False
    return len([token for token in hip_visible.split(",") if token]) == 1


def select_ray_temp_dir(preferred: str | None = None) -> str:
    """Pick a temp directory on a filesystem with >10% free space when possible."""
    candidates: list[str] = []
    env_tmp = os.environ.get("RAY_TMPDIR")
    if env_tmp:
        Path(env_tmp).mkdir(parents=True, exist_ok=True)
        return env_tmp
    slurm_tmp = _slurm_ray_temp_candidate()
    if slurm_tmp:
        candidates.append(slurm_tmp)
    if preferred:
        candidates.append(preferred)
    home_default = str(Path.home() / "tmp-ray")
    user = os.environ.get("USER", "default")
    candidates.extend(
        [
            f"/tmp/{user}/ray",
            home_default,
            "/tmp/ray",
        ]
    )

    best_path = home_default
    best_ratio = -1.0
    for path in candidates:
        try:
            base = Path(path)
            base.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(base))
            ratio = usage.free / usage.total if usage.total else 0.0
            if ratio > 0.10:
                return str(base)
            if ratio > best_ratio:
                best_ratio = ratio
                best_path = str(base)
        except OSError:
            continue

    return best_path


def _slurm_ray_temp_candidate() -> str | None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    safe_job_id = "".join(ch for ch in job_id if ch.isalnum() or ch in ("_", "-"))
    return f"/tmp/r{safe_job_id}" if safe_job_id else None


def _paths_with_free_space(
    candidates: list[str], min_free_ratio: float = 0.10
) -> list[str]:
    qualified: list[str] = []
    for path in candidates:
        try:
            base = Path(path)
            base.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(base))
            ratio = usage.free / usage.total if usage.total else 0.0
            if ratio >= min_free_ratio:
                qualified.append(str(base))
        except OSError:
            continue
    return qualified


def select_object_spilling_dir(ray_temp_dir: str | None = None) -> str:
    """Choose a directory with enough free space for Ray object spilling."""
    home = str(Path.home())
    temp_root = ray_temp_dir or str(Path.home() / "tmp-ray")
    candidates = [
        os.path.join(temp_root, "spill"),
        os.path.join(home, "tmp-ray", "spill"),
        os.path.join(home, "ray_spill"),
    ]
    good = _paths_with_free_space(candidates, min_free_ratio=0.10)
    target = good[0] if good else candidates[-1]
    Path(target).mkdir(parents=True, exist_ok=True)
    return target


def get_requested_ray_address(ray_config: dict | None = None) -> str | None:
    """Resolve external Ray cluster address from env, then YAML config."""
    for key in ("LEAP_RAY_ADDRESS", "RAY_ADDRESS"):
        value = os.environ.get(key)
        if value:
            return value

    if ray_config:
        value = ray_config.get("address")
        if value:
            return str(value)

    return None


def resolve_num_workers(
    ray_config: dict | None,
    *,
    local_num_gpus: int,
    connected_to_existing_cluster: bool,
) -> int:
    """Resolve the Ray worker count for this training job."""
    for key in ("LEAP_RAY_NUM_WORKERS", "LEAP_NUM_WORKERS"):
        raw = os.environ.get(key)
        if raw:
            return int(raw)

    if ray_config and ray_config.get("num_workers") is not None:
        return int(ray_config["num_workers"])

    if connected_to_existing_cluster:
        import ray

        cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
        if cluster_gpus < 1:
            raise ValueError(
                "Connected to external Ray cluster but no GPU resources were detected. "
                "Set LEAP_RAY_NUM_WORKERS if cluster resources are still registering."
            )
        return cluster_gpus

    return local_num_gpus


def resolve_local_ray_num_cpus() -> int | None:
    """Return the CPU count Ray should advertise for a local Slurm allocation."""
    raw = os.environ.get("LEAP_RAY_NUM_CPUS")
    if raw:
        return int(raw)

    for key in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        raw = os.environ.get(key)
        if raw:
            return int(raw)

    cpus_per_gpu = os.environ.get("SLURM_CPUS_PER_GPU")
    gpus_per_task = os.environ.get("SLURM_GPUS_PER_TASK")
    if cpus_per_gpu and gpus_per_task:
        return int(cpus_per_gpu) * int(gpus_per_task)

    return None


def build_scaling_config(
    ray_config: dict | None,
    *,
    num_workers: int,
    resources_per_worker: dict | None = None,
):
    from ray.train import ScalingConfig

    resources_per_worker = resources_per_worker or {"GPU": 1.0}
    if ray_config and isinstance(ray_config.get("resources_per_worker"), dict):
        resources_per_worker = dict(ray_config["resources_per_worker"])
        resources_per_worker.setdefault("GPU", 1.0)

    return ScalingConfig(
        num_workers=num_workers,
        use_gpu=resources_per_worker.get("GPU", 0) > 0,
        resources_per_worker=resources_per_worker,
    )


def resolve_local_object_store_memory() -> int:
    """Choose a local Ray object store size that respects tiny /dev/shm setups."""
    available_mem = int(psutil.virtual_memory().available * 0.4)
    default_cap = 8 * 1024**3
    target = min(available_mem, default_cap)

    try:
        shm_total = shutil.disk_usage("/dev/shm").total
    except OSError:
        shm_total = 0

    if shm_total >= 1 * 1024**3:
        return min(target, int(shm_total * 0.8))

    os.environ.setdefault("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1")
    return min(target, 2 * 1024**3)


# Ray's AMD accelerator manager rejects ROCR_VISIBLE_DEVICES during import.
# Normalize it as soon as this dependency-free runtime module is imported so
# all Ray-backed modules can safely import Ray afterwards.
normalize_visible_devices()
