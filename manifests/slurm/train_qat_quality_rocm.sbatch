#!/bin/bash
#SBATCH --job-name=qat-quality-train
#SBATCH --partition=gpuworker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/OUT_%x.%j
#SBATCH --error=logs/ERR_%x.%j

set -euo pipefail

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi

: "${ROCM_VENV:?Set ROCM_VENV to the ROCm leap-finetune environment}"
: "${TRAIN_CONFIG:?Set TRAIN_CONFIG to a local training YAML}"
: "${QUALITY_OUTPUT_DIR:?Set QUALITY_OUTPUT_DIR to a persistent output directory}"
REPO_ROOT="${LEAP_FINETUNE_ROOT:-$SLURM_SUBMIT_DIR}"

mkdir -p logs "$QUALITY_OUTPUT_DIR"
export OUTPUT_DIR="$QUALITY_OUTPUT_DIR"
export HF_HOME="${HF_HOME:-${QUALITY_OUTPUT_DIR}/hf-cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TMPDIR="/tmp/leap-qat-quality-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

cd "$REPO_ROOT"
"$ROCM_VENV/bin/leap-finetune" run "$TRAIN_CONFIG"
