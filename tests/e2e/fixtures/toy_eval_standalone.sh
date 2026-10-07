#!/bin/bash
# Toy slurm launcher for standalone eval only.
# Uses the explicit eval subcommand so the fixture can pass `--output`.
#SBATCH --job-name=eval_standalone_toy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=00:20:00
#SBATCH --output=logs/eval_standalone_toy/OUT_%j.log
#SBATCH --error=logs/eval_standalone_toy/ERR_%j.log

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

export TMPDIR="${HOME}/tmp/eval-standalone-${SLURM_JOB_ID:-manual}"
export TRITON_CACHE_DIR="${TMPDIR}/triton_cache"
export TORCH_EXTENSIONS_DIR="${TMPDIR}/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" logs/eval_standalone_toy outputs/eval_standalone_toy

liquid-finetune eval \
    tests/e2e/fixtures/toy_eval_standalone.yaml \
    --output "outputs/eval_standalone_toy/results_${SLURM_JOB_ID:-manual}.json"
