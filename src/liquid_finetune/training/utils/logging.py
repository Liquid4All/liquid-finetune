import os
import warnings
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path

_ENV_DONE = False


def _is_ray_worker() -> bool:
    """Check if we're running inside a Ray worker process."""
    # Ray sets these env vars in worker processes
    return any(
        os.environ.get(var)
        for var in ["RAY_WORKER_MODE", "RAY_RAYLET_PID", "RAY_JOB_ID"]
    )


def is_rank_zero() -> bool:
    """Check if this is the main (rank 0) process for Ray Train.

    Returns True if rank 0 or if not running in a Ray context (single process).
    """
    try:
        from ray.train import get_context

        ctx = get_context()
        return ctx is None or ctx.get_world_rank() == 0
    except Exception:
        return True


def init_tracker(
    job_name: str,
    tracker: str,
    space_id: str | None = None,
    output_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    """Initialize experiment tracker. Must be called BEFORE creating the trainer.

    Args:
        job_name: Name for the run
        tracker: "wandb", "trackio", or "none"
        space_id: HF Space ID for trackio (required when tracker is "trackio")
        output_dir: Training output directory — used to persist/restore wandb run ID
        resume_from_checkpoint: If set, restore the saved wandb run ID for continuity
    """
    if tracker == "none":
        return

    try:
        if tracker == "trackio":
            import trackio as wandb

            if not space_id:
                raise ValueError(
                    "trackio requires 'trackio_space_id' in training_config"
                )
        else:
            import wandb

        if is_rank_zero():
            project = os.environ.get("WANDB_PROJECT", "liquid-finetune")

            # Auto-read saved run ID from previous run if resuming.
            # Stored per job_name to avoid collisions between different runs.
            run_id = None
            run_id_file = (
                Path(output_dir) / f".wandb_run_id_{job_name}" if output_dir else None
            )
            if resume_from_checkpoint and run_id_file and run_id_file.exists():
                run_id = run_id_file.read_text().strip() or None

            init_kwargs = {
                "project": project,
                "name": job_name,
                "resume": "allow",
            }
            if run_id:
                init_kwargs["id"] = run_id
            if tracker == "wandb":
                init_kwargs["settings"] = wandb.Settings(_disable_stats=False)
            if space_id:
                init_kwargs["space_id"] = space_id

            run = wandb.init(**init_kwargs)
            if hasattr(run, "url") and run.url:
                print(f"\nTracker URL: {run.url}")

            # Persist run ID so resumed runs can continue logging to the same run
            if wandb.run and run_id_file:
                run_id_file.write_text(wandb.run.id)
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(f"Failed to initialize {tracker}: {e}", UserWarning)


def finish_tracker(tracker: str) -> None:
    """Cleanly finish the tracker run so it shows as 'Completed'.

    Must only be called after all training steps have finished successfully.
    Only acts on rank 0 (the process that owns the run).
    """
    if tracker == "none" or not is_rank_zero():
        return
    try:
        if tracker == "trackio":
            import trackio as wandb
        else:
            import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def get_wandb_run_id() -> str | None:
    """Best-effort fetch of the active wandb run id from rank 0."""
    if not is_rank_zero():
        return None
    try:
        import wandb

        return wandb.run.id if wandb.run is not None else None
    except Exception:
        return None


def setup_training_environment() -> None:
    """Configure training environment. Only prints messages on driver process."""
    global _ENV_DONE
    if _ENV_DONE:
        return

    is_worker = _is_ray_worker()

    os.environ.setdefault("DS_DISABLE_CONFIG_PRINT", "1")
    os.environ.setdefault("DEEPSPEED_LOG_LEVEL", "ERROR")
    os.environ.setdefault("RAY_DATA_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1")
    # Skip noisy duplicate logs from workers (NCCL warnings, object store warnings, SplitCoordinator)
    os.environ.setdefault(
        "RAY_DEDUP_LOGS_SKIP_REGEX",
        r"ProcessGroupNCCL|object.store.is.configured|SplitCoordinator",
    )
    warnings.filterwarnings("ignore")  # keep only tracebacks

    if "WANDB_API_KEY" not in os.environ and "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"

    # Keep JIT/compiler caches under the worker temp root instead of /tmp.
    pid = os.getpid()
    temp_root = os.environ.get("TMPDIR", str(Path.home() / "tmp-ray"))
    cache = Path(temp_root) / f"triton_cache_{pid}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)

    # Disable DeepSpeed Triton autotune to prevent /dev/shm permission errors
    os.environ.setdefault("DS_TRITON_AUTOTUNE", "0")
    os.environ.setdefault("TRITON_DISABLE_AUTOTUNE", "1")

    try:
        import deepspeed
        from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

        ds_log = deepspeed.utils.logging.logger
        ds_log.setLevel(logging.ERROR)
        for h in ds_log.handlers:
            h.setLevel(logging.ERROR)

        _orig_ds_destroy = BF16_Optimizer.destroy

        def _safe_ds_destroy(self, *a, **kw):
            try:
                _orig_ds_destroy(self, *a, **kw)
            except IndexError:
                pass

        BF16_Optimizer.destroy = _safe_ds_destroy

    except ImportError:
        pass  # Silent - deepspeed not required

    if not is_worker:
        print("Training environment configured ✅")
    _ENV_DONE = True


def print_next_steps_panel(output_dir: str) -> None:
    """Render the Next Steps panel spanning full console width."""
    console = Console()
    quick_start_url = (
        "https://liquid.liquid.ai/docs/liquid-bundle/quick-start?utm_source=github"
        "&utm_medium=link&utm_campaign=LIQUID&utm_content=general"
    )

    next_steps_table = Table(show_header=False, box=None, padding=(0, 2))
    next_steps_table.add_column("Property", style="bold cyan", min_width=15)
    next_steps_table.add_column("Value", style="green")

    next_steps_table.add_row("Status", "[bold green]Training complete![/bold green]")
    next_steps_table.add_row("Checkpoint Directory", f"[cyan]{output_dir}[/cyan]")
    next_steps_table.add_row(
        "Bundle",
        f"[dim]liquid-bundle create {output_dir}/[CHECKPOINT_NAME][/dim]",
    )
    next_steps_table.add_row(
        "Quick Start",
        f"[link={quick_start_url}]{quick_start_url}[/link]",
    )

    console.print(
        Panel(
            next_steps_table,
            title="Next Step: Bundle for LIQUID",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )
