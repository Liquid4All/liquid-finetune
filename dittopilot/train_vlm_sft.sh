#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=""
DATASET_DIR=""
OUTPUT_DIR=""
PROJECT_NAME=""
RUN_NAME=""
WANDB_PROJECT=""
LIMIT=""
MAX_STEPS=""
BATCH_SIZE=""
LEARNING_RATE=""
WARMUP_RATIO=""
LORA_R=""
LORA_ALPHA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --project-name) PROJECT_NAME="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --warmup-ratio) WARMUP_RATIO="$2"; shift 2 ;;
    --lora-r) LORA_R="$2"; shift 2 ;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for required in MODEL_PATH DATASET_DIR OUTPUT_DIR PROJECT_NAME RUN_NAME WANDB_PROJECT LIMIT MAX_STEPS BATCH_SIZE LEARNING_RATE WARMUP_RATIO LORA_R LORA_ALPHA; do
  if [[ -z "${!required}" ]]; then
    echo "missing required value: ${required}" >&2
    exit 2
  fi
done

TRAIN_PATH="${DATASET_DIR}/train.jsonl"
EVAL_PATH="${DATASET_DIR}/eval.jsonl"
[[ -f "${TRAIN_PATH}" ]] || { echo "missing ${TRAIN_PATH}" >&2; exit 1; }
[[ -f "${EVAL_PATH}" ]] || { echo "missing ${EVAL_PATH}" >&2; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "missing model directory ${MODEL_PATH}" >&2; exit 1; }

CONFIG_PATH="$(mktemp --suffix=.yaml)"
trap 'rm -f "${CONFIG_PATH}"' EXIT
cat >"${CONFIG_PATH}" <<EOF
project_name: "${PROJECT_NAME}"
model_name: "${MODEL_PATH}"
training_type: "vlm_sft"

dataset:
  train_path: "${TRAIN_PATH}"
  val_path: "${EVAL_PATH}"
  type: "vlm_sft"
  limit: ${LIMIT}

training_config:
  extends: "DEFAULT_VLM_SFT"
  max_steps: ${MAX_STEPS}
  per_device_train_batch_size: ${BATCH_SIZE}
  per_device_eval_batch_size: ${BATCH_SIZE}
  gradient_accumulation_steps: 1
  learning_rate: ${LEARNING_RATE}
  lr_scheduler_type: "cosine"
  warmup_ratio: ${WARMUP_RATIO}
  max_length: 1024
  do_image_splitting: false
  max_image_tokens: 64
  assistant_only_loss: true
  skip_peft_merge: true
  tracker: "wandb"
  eval_strategy: "steps"
  eval_steps: 10
  save_strategy: "steps"
  save_steps: ${MAX_STEPS}
  save_total_limit: 1
  logging_steps: 1
  seed: 20260916
  output_dir: "${OUTPUT_DIR}"

peft_config:
  extends: "DEFAULT_VLM_LORA"
  use_peft: true
  r: ${LORA_R}
  lora_alpha: ${LORA_ALPHA}
  lora_dropout: 0.05
EOF

cat "${CONFIG_PATH}"
if [[ "${LEAP_TRAINER_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

export LEAP_FINETUNE_FROM_SLURM=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_NAME="${RUN_NAME}"

# AMD allocations expose one cgroup-isolated GPU. Ray and torch both address it
# as local ordinal zero; avoid leaking the physical ROCR ordinal into workers.
if envs/rocm/.venv/bin/python - <<'PY' >/dev/null 2>&1
import sys
import torch
sys.exit(0 if getattr(torch.version, "hip", None) else 1)
PY
then
  unset ROCR_VISIBLE_DEVICES
  export HIP_VISIBLE_DEVICES=0
  export CUDA_VISIBLE_DEVICES=0
fi

envs/rocm/.venv/bin/leap-finetune run "${CONFIG_PATH}"

if [[ ! -d "${OUTPUT_DIR}" ]] || [[ -z "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 2 -print -quit)" ]]; then
  echo "training produced no checkpoint files under ${OUTPUT_DIR}" >&2
  exit 1
fi
