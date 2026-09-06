from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from liquid_finetune.evaluation.async_eval_config import AsyncEvalConfig
from liquid_finetune.evaluation.runner import stage_checkpoint_for_eval
from liquid_finetune.training.utils.logging import is_rank_zero

logger = logging.getLogger(__name__)

# ==== Reserved Async Eval Callback ====


@dataclass
class _EvalRequest:
    step: int
    ckpt_path: Path


@dataclass
class _EvalResult:
    step: int
    metrics: dict
    # False ONLY when the helper-thread cycle raised. ``metrics`` can be
    # empty in healthy cycles too (e.g. benchmarks with no samples, or
    # NotImplementedError-skipped benchmarks for the current backend), so
    # we must distinguish "cycle failed" from "cycle ran clean and had
    # nothing to log" for the auto-disable counter.
    ok: bool = True


class ReservedEvalCallback(TrainerCallback):
    """Long-running vLLM server on dedicated GPUs; helper thread drives it."""

    def __init__(
        self,
        *,
        benchmarks: list,
        cfg: AsyncEvalConfig,
        server_url: str,
        output_dir: str | None,
        eval_gpu_ids: str = "",
    ):
        super().__init__()
        if cfg.reserved.weight_reload != "respawn":
            raise NotImplementedError(
                f"async_eval.reserved.weight_reload="
                f"{cfg.reserved.weight_reload!r} not supported; use 'respawn'."
            )

        self.benchmarks = benchmarks
        self.cfg = cfg
        self.server_url = server_url
        self.output_dir = Path(output_dir) if output_dir else None
        self.eval_gpu_ids = eval_gpu_ids

        self._input_q: queue.Queue[_EvalRequest | None] = queue.Queue()
        self._output_q: queue.Queue[_EvalResult] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self._disabled = False
        self._in_flight_lock = threading.Lock()
        self._in_flight_count = 0

    @property
    def _eval_dir(self) -> Path:
        assert self.output_dir is not None
        d = self.output_dir / "_async_eval"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run_loop,
                name="liquid-async-eval",
                daemon=True,
            )
            self._thread.start()

    def _run_loop(self) -> None:
        """Helper thread main loop. Runs eval cycles serially."""
        from liquid_finetune.evaluation.backend import VLLMServerBackend

        backend = VLLMServerBackend(self.server_url)
        while True:
            req = self._input_q.get()
            if req is None:
                break
            try:
                metrics, ok = self._run_one_cycle(backend, req)
                self._output_q.put(_EvalResult(step=req.step, metrics=metrics, ok=ok))
            except Exception:
                logger.exception(
                    "[async_eval/reserved] cycle failed at step %d", req.step
                )
                self._output_q.put(_EvalResult(step=req.step, metrics={}, ok=False))
            finally:
                self._release_eval_slot()

    def _run_one_cycle(self, backend, req: _EvalRequest) -> tuple[dict, bool]:
        """Respawn the server with the new checkpoint, wait for /health,
        then run all benchmarks through ``backend``.

        Returns ``(results, ok)``. ``ok=False`` only when at least one
        benchmark raised a real error AND no benchmark produced any
        metrics — i.e. the cycle yielded nothing despite a real failure.
        NotImplementedError and empty-samples paths don't count as
        failures; they're healthy ways for a cycle to be a no-op.
        """
        from liquid_finetune.distribution.vllm_server import wait_for_vllm_health

        self._respawn_server(req.ckpt_path)
        wait_for_vllm_health(
            self.server_url,
            timeout=600.0,
            process=self._server_process,
        )

        results: dict[str, float] = {}
        real_errors = 0
        for bench in self.benchmarks:
            samples = bench.get_samples()
            if not samples:
                continue
            try:
                r = bench.evaluate_with_backend(backend, samples)
            except NotImplementedError as e:
                logger.warning(
                    "[async_eval/reserved] [%s] backend doesn't support: %s",
                    bench.name,
                    e,
                )
                continue
            except Exception:
                logger.exception(
                    "[async_eval/reserved] [%s] failed; other benchmarks continue",
                    bench.name,
                )
                real_errors += 1
                continue
            for metric, total in r.metrics.items():
                avg = total / r.count if r.count > 0 else 0.0
                results[f"benchmark/{bench.name}/{metric}"] = avg

        # Partial data (some succeeded) → ok. No real errors → ok (healthy
        # no-op). Real errors with zero salvaged metrics → not ok.
        ok = bool(results) or real_errors == 0
        return results, ok

    # === Subprocess management (helper thread owns it) ===

    _server_process = None  # subprocess.Popen of vLLM OpenAI API server

    def _respawn_server(self, ckpt_path: Path) -> None:
        import shlex
        import subprocess
        import sys
        import time
        from urllib.parse import urlparse

        # Tear down existing — then SLEEP to let CUDA fully release the GPU's
        # memory. Without this pause, the next vLLM startup can race the
        # kernel's allocator cleanup and fail engine-core init.
        if self._server_process is not None and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait(timeout=5)
            time.sleep(5)
        self._server_process = None

        port = urlparse(self.server_url).port or self.cfg.reserved.server_port
        if not self.eval_gpu_ids:
            raise RuntimeError(
                "eval_gpu_ids not set on the callback; the driver must populate "
                "train_loop_config['async_eval_gpu_ids'] when mode=reserved."
            )

        # Launch via the venv's own Python to guarantee the right interpreter
        # (a system-wide ``vllm`` shim can point at a different Python). The
        # OpenAI-compatible api_server exposes /v1/chat/completions.
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(ckpt_path),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(self.cfg.tensor_parallel_size),
            "--dtype",
            self.cfg.dtype,
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            # Stable served-model-name so clients can address respawned
            # servers by the same string across cycles.
            "--served-model-name",
            "default",
        ]
        if self.cfg.max_model_len is not None:
            cmd += ["--max-model-len", str(self.cfg.max_model_len)]

        # Strip distributed env vars leaking from the parent Ray worker
        # (LOCAL_RANK, RANK, etc.) before pinning CUDA_VISIBLE_DEVICES to
        # the carved-out eval GPUs.
        from liquid_finetune.evaluation.sidecar_callback import _clean_subprocess_env

        env = _clean_subprocess_env()
        env["CUDA_VISIBLE_DEVICES"] = self.eval_gpu_ids

        # Append server stdout+stderr across respawns so failures stay
        # debuggable. Close the parent's fd as soon as Popen duplicates it
        # into the child — otherwise we leak a fd per respawn over the
        # lifetime of training.
        log_dir = self._eval_dir / "vllm_server"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "server.log"
        logger.info(
            "[async_eval/reserved] launching vLLM (log=%s): %s",
            log_path,
            " ".join(shlex.quote(c) for c in cmd),
        )
        with open(log_path, "ab") as server_log:
            self._server_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )

    # === TrainerCallback hooks ===

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if self._disabled or not is_rank_zero() or not self.benchmarks:
            return
        if not self.output_dir:
            logger.warning(
                "[async_eval/reserved] output_dir not set; skipping step %d",
                state.global_step,
            )
            return

        if not self._reserve_eval_slot(state.global_step):
            return

        try:
            self._ensure_thread()
            ckpt_path = self._save_checkpoint(model, state)
            self._input_q.put(_EvalRequest(step=state.global_step, ckpt_path=ckpt_path))
            # NOTE: do NOT reset _consecutive_failures here. The counter is
            # shared with helper-thread cycle failures drained in
            # _account_result; a successful submit followed by N failed
            # cycles would otherwise keep wiping the running count and
            # auto-disable could never fire. Only an ok=True cycle drain
            # resets the counter — see _account_result.
        except Exception:
            self._release_eval_slot()
            self._consecutive_failures += 1
            logger.exception(
                "[async_eval/reserved] submission failed at step %d (%d consecutive)",
                state.global_step,
                self._consecutive_failures,
            )
            if self._consecutive_failures >= self.cfg.failure.max_consecutive:
                self._disabled = True
                logger.error(
                    "[async_eval/reserved] disabling after %d consecutive failures",
                    self._consecutive_failures,
                )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if not is_rank_zero():
            return
        # Drain whatever's ready and log to wandb at the originating step
        while True:
            try:
                result = self._output_q.get_nowait()
            except queue.Empty:
                break
            self._account_result(result)
            self._log_to_wandb(result)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if not is_rank_zero():
            return
        # First drain — pick up anything that finished while training was ending
        while True:
            try:
                result = self._output_q.get_nowait()
            except queue.Empty:
                break
            self._account_result(result)
            self._log_to_wandb(result)

        # Send poison pill AFTER any in-flight requests; wait long enough for
        # one full eval cycle to complete (server respawn ~60s + eval ~30s).
        # This is the user-visible contract: if your eval was in-flight at
        # train-end, we'll wait for it instead of dropping it on the floor.
        if self._thread is not None and self._thread.is_alive():
            logger.info(
                "[async_eval/reserved] draining in-flight eval cycles (up to %ds)...",
                600,
            )
            self._input_q.put(None)
            self._thread.join(timeout=600)
            if self._thread.is_alive():
                logger.warning(
                    "[async_eval/reserved] helper thread did not exit within "
                    "drain window; proceeding to teardown"
                )

        # Second drain — pick up the results that arrived while we were
        # waiting for the helper thread to finish.
        while True:
            try:
                result = self._output_q.get_nowait()
            except queue.Empty:
                break
            self._account_result(result)
            self._log_to_wandb(result)

        if self._server_process is not None and self._server_process.poll() is None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=10)
            except Exception:
                pass

    # === Helpers ===

    def _reserve_eval_slot(self, step: int) -> bool:
        with self._in_flight_lock:
            if self.cfg.on_overlap == "skip" and self._in_flight_count > 0:
                logger.warning(
                    "[async_eval/reserved] previous eval still in flight; "
                    "skipping step %d",
                    step,
                )
                return False
            self._in_flight_count += 1
            return True

    def _release_eval_slot(self) -> None:
        with self._in_flight_lock:
            self._in_flight_count = max(0, self._in_flight_count - 1)

    def _save_checkpoint(self, model, state: TrainerState) -> Path:
        return stage_checkpoint_for_eval(
            model=model,
            benchmarks=self.benchmarks,
            ckpt_root=self._eval_dir / "checkpoints",
            step=state.global_step,
            keep=2,
        )

    def _account_result(self, result: _EvalResult) -> None:
        """Track helper-thread eval-cycle success/failure for auto-disable.

        Only ``ok=False`` (a raised exception inside ``_run_one_cycle``)
        counts as a failure. An empty ``metrics`` dict with ``ok=True``
        is healthy — happens when benchmarks legitimately produce no
        rows (no samples loaded, or all benchmarks ``NotImplementedError``
        on the current backend and got skipped). Treating those as
        failures would auto-disable a working setup.
        """
        if not result.ok:
            self._consecutive_failures += 1
            logger.warning(
                "[async_eval/reserved] eval cycle failed at step %d (%d consecutive)",
                result.step,
                self._consecutive_failures,
            )
            if self._consecutive_failures >= self.cfg.failure.max_consecutive:
                self._disabled = True
                logger.error(
                    "[async_eval/reserved] disabling after %d consecutive failures",
                    self._consecutive_failures,
                )
            return
        self._consecutive_failures = 0

    def _log_to_wandb(self, result: _EvalResult) -> None:
        if not result.metrics:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            # Axis-pin benchmark/* to benchmark/step. train/global_step and
            # benchmark/step are payload fields, NOT step=result.step — the
            # trainer has already advanced past result.step so step= would
            # be a backwards write wandb drops.
            try:
                wandb.define_metric("benchmark/step")
                wandb.define_metric("benchmark/*", step_metric="benchmark/step")
            except Exception as e:
                logger.warning("wandb.define_metric failed; axis-pin skipped (%s)", e)
            payload = {
                **result.metrics,
                "train/global_step": result.step,
                "benchmark/step": result.step,
            }
            wandb.log(payload)
            logger.info(
                "[async_eval/reserved] logged %d metrics at step %d",
                len(result.metrics),
                result.step,
            )
        except ImportError:
            pass
        except Exception:
            logger.warning("[async_eval/reserved] wandb.log failed", exc_info=True)
