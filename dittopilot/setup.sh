#!/usr/bin/env bash
set -euo pipefail

# The allocation can inherit an unrelated active virtualenv from the submitter.
# Pin the ROCm lockfile's supported interpreter instead of letting uv reuse it.
unset VIRTUAL_ENV
UV_PROJECT=envs/rocm uv sync --frozen --python 3.12 --no-group rocm-vllm
