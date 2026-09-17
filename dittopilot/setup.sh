#!/usr/bin/env bash
set -euo pipefail
UV_PROJECT=envs/rocm uv sync --frozen --no-group rocm-vllm
