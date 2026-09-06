#!/bin/bash
# Toy slurm launcher for async_eval mode=reserved.
# Allocates 3 GPUs total: 2 for training, 1 carved off at job start for the
# long-running trl vllm-serve subprocess that the helper thread manages.
#SBATCH --job-name=eval_toy_reserved
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-gpu=14
#SBATCH --time=00:45:00
#SBATCH --output=logs/async_eval_toy/OUT_reserved_%j.log
#SBATCH --error=logs/async_eval_toy/ERR_reserved_%j.log

set -euo pipefail

if [ -n "${LIQUID_CUDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module load "$LIQUID_CUDA_MODULE" 2>/dev/null || true
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_ACTIVATE="${VIRTUAL_ENV:-$(pwd)/.venv}/bin/activate"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
    VENV_ACTIVATE="$(pwd)/.venv/bin/activate"
fi
source "${VENV_ACTIVATE}"
[ -f "$HOME/.env" ] && source "$HOME/.env"

export TMPDIR="${HOME}/tmp"
export TRITON_CACHE_DIR="${HOME}/.triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" logs/async_eval_toy

liquid-finetune tests/e2e/fixtures/toy_async_eval_reserved.yaml
