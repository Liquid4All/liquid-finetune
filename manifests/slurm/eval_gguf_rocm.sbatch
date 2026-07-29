#!/bin/bash
#SBATCH --job-name=qat-gguf-eval
#SBATCH --partition=gpuworker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/OUT_%x.%j
#SBATCH --error=logs/ERR_%x.%j

set -euo pipefail

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi

: "${EVAL_CONFIG:?Set EVAL_CONFIG to a llama_cpp standalone eval YAML}"
REPO_ROOT="${LEAP_FINETUNE_ROOT:-$SLURM_SUBMIT_DIR}"

mkdir -p logs
cd "$REPO_ROOT"
"$REPO_ROOT/.venv/bin/leap-finetune" eval "$EVAL_CONFIG"
