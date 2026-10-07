#!/bin/bash
# Toy slurm launcher for async_eval mode=sidecar.
# Training takes 2 GPUs on one node. The sidecar callback will sbatch ITS
# OWN 1-GPU eval job per eval_step, separate from this allocation.
#SBATCH --job-name=eval_toy_sidecar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=14
#SBATCH --time=00:45:00
#SBATCH --output=logs/async_eval_toy/OUT_sidecar_%j.log
#SBATCH --error=logs/async_eval_toy/ERR_sidecar_%j.log

set -euo pipefail

# Optional: load your cluster's CUDA module if `module load` is in use.
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

# Make sure the sidecar runner picks up the same caches
export TMPDIR="${HOME}/tmp"
export TRITON_CACHE_DIR="${HOME}/.triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" logs/async_eval_toy

liquid-finetune tests/e2e/fixtures/toy_async_eval_sidecar.yaml
