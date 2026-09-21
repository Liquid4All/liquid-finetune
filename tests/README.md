# Tests

Keep the suite intentionally small. New tests should land in one of three buckets:

- `config/` — Pydantic config parsing and validation.
- `numerics/` — deterministic correctness for losses, metrics, masks, routing, sharding, and adapters.
- `e2e/` — full-service training tests, fixtures, and SLURM launchers.

## Local Checks

```bash
uv run pytest tests/config tests/numerics
```

On AMD / ROCm environments, use the ROCm project so CUDA and ROCm locks remain
separate. Set this once in your shell, module, or direnv config:

```bash
export UV_PROJECT=envs/rocm
uv run python -m pytest tests/config tests/numerics
```

## GPU Smoke Tests

The complete E2E suite is SLURM-only. The launcher submits two jobs: a four-GPU
Ray job for the normal distributed cases and a one-GPU job for explicit native
local cases. Each job uses one self-contained, job-scoped temporary
root, requests an early TERM signal for cleanup, and removes its own temporary
root on normal termination. On AMD/ROCm, select the project before submission:

```bash
export UV_PROJECT=envs/rocm
tests/e2e/slurm/submit_e2e_tests.sh
```

Inspect both generated batch scripts without submitting them:

```bash
tests/e2e/slurm/submit_e2e_tests.sh --dry-run
```

Individual tests may be run directly for debugging. A direct multi-test E2E
invocation is rejected; a multi-GPU test with only one visible GPU fails with a
message directing you to SLURM.

```bash
uv run pytest tests/e2e/test_grpo_e2e.py::TestDenseGRPO::test_text_grpo_optimizes -v
```

To submit only one side of the matrix:

```bash
tests/e2e/slurm/submit_e2e_tests.sh --mode=ray
tests/e2e/slurm/submit_e2e_tests.sh --mode=local
```

## FA2 Validation

Normal tests may fall back to SDPA. FA2 validation should inspect the active
environment and can require runtime selection:

```bash
uv sync
uv run leap-finetune env fa2-status --require
```

For ROCm:

```bash
UV_PROJECT=envs/rocm uv sync
UV_PROJECT=envs/rocm uv run leap-finetune env fa2-status --require
```

## SLURM

```bash
tests/e2e/slurm/submit_e2e_tests.sh --dry-run
tests/e2e/slurm/submit_e2e_tests.sh
sbatch tests/e2e/fixtures/toy_async_eval_sidecar.sh
sbatch tests/e2e/fixtures/toy_async_eval_reserved.sh
```
