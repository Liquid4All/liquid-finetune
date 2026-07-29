# Tests

Keep the suite intentionally small. New tests should land in one of four buckets:

- `config/` — Pydantic config parsing and validation.
- `numerics/` — deterministic correctness for losses, metrics, masks, routing, sharding, and adapters.
- `quantization/` — QAT kernels, model preparation, and deployment backends.
- `e2e/` — full-service training tests, fixtures, and SLURM launchers.

## Local Checks

```bash
uv run pytest tests/config tests/numerics tests/quantization
```

On AMD / ROCm environments, use the ROCm project so CUDA and ROCm locks remain
separate. Set this once in your shell, module, or direnv config:

```bash
export UV_PROJECT=envs/rocm
uv run python -m pytest tests/config tests/numerics tests/quantization
```

## GPU Smoke Tests

```bash
uv run pytest tests/e2e --dense --moe --vlm --retrieval
```

For ROCm:

```bash
UV_PROJECT=envs/rocm uv run python -m pytest tests/e2e --dense --moe --vlm --retrieval
```

To run only the native local-path cases on one GPU:

```bash
GPUS_PER_TASK=1 PYTEST_ARGS="tests/e2e -m single_gpu" tests/e2e/slurm/submit_e2e_tests.sh
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

QAT deployment and evaluation jobs are under `manifests/slurm`. They
require explicit persistent paths and never place models or environments in
`/tmp`.
