#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=""
DATASET=""
OUTPUT=""
RUN_NAME=""
LEARNING_RATE="1e-4"
BATCH_SIZE="32"
GRADIENT_ACCUMULATION_STEPS="1"
MAX_STEPS="3000"
TEMPERATURE="0.02"
MAX_IMAGE_TOKENS="256"
LORA_RANK="64"
LORA_ALPHA="64"
WARMUP_RATIO="0.05"
WEIGHT_DECAY="0.01"
SAVE_STEPS="500"
SEED="20261019"

while (($#)); do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --gradient-accumulation-steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --max-image-tokens) MAX_IMAGE_TOKENS="$2"; shift 2 ;;
    --lora-rank) LORA_RANK="$2"; shift 2 ;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
    --warmup-ratio) WARMUP_RATIO="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --save-steps) SAVE_STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in CHECKPOINT DATASET OUTPUT RUN_NAME; do
  if [[ -z "${!value}" ]]; then
    echo "Missing required value: ${value}" >&2
    exit 2
  fi
done
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -e "${DATASET}" ]]; then
  echo "Dataset path does not exist: ${DATASET}" >&2
  exit 2
fi
if [[ -d "${DATASET}" ]]; then
  DATASET="${DATASET}/train.jsonl"
fi
if [[ ! -f "${DATASET}" ]]; then
  echo "Training JSONL does not exist: ${DATASET}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite output path: ${OUTPUT}" >&2
  exit 2
fi

STAGING_ROOT="${OUTPUT}.staging"
CONFIG_PATH="${TMPDIR:-/tmp}/vlm-embedding-${SLURM_JOB_ID:-local}.yaml"
if [[ -e "${STAGING_ROOT}" ]]; then
  echo "Refusing to overwrite staging path: ${STAGING_ROOT}" >&2
  exit 2
fi
mkdir -p "${STAGING_ROOT}"

cat >"${CONFIG_PATH}" <<EOF
project_name: "retrieval_lfm_vlm_embedding"
job_name: "${RUN_NAME}"
model_name: "${CHECKPOINT}"
training_type: "vlm_embedding"
dataset:
  path: "${DATASET}"
  type: "vlm_embedding"
training_config:
  extends: "DEFAULT_VLM_EMBEDDING"
  max_steps: ${MAX_STEPS}
  per_device_train_batch_size: ${BATCH_SIZE}
  per_device_eval_batch_size: ${BATCH_SIZE}
  gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
  learning_rate: ${LEARNING_RATE}
  warmup_ratio: ${WARMUP_RATIO}
  weight_decay: ${WEIGHT_DECAY}
  seed: ${SEED}
  data_seed: ${SEED}
  temperature: ${TEMPERATURE}
  max_image_tokens: ${MAX_IMAGE_TOKENS}
  eval_strategy: "no"
  eval_on_start: false
  save_strategy: "steps"
  save_steps: ${SAVE_STEPS}
  save_total_limit: 2
  logging_steps: 10
  dataloader_num_workers: 8
  dataloader_prefetch_factor: 4
  leap_run_name_template: "vlm-embedding-lora_a"
  tracker: "none"
peft_config:
  extends: "DEFAULT_VLM_LORA"
  use_peft: true
  r: ${LORA_RANK}
  lora_alpha: ${LORA_ALPHA}
  lora_dropout: 0.0
EOF

UV_PROJECT_ARGS=()
if [[ "${DITTOPILOT_CLUSTER:-local}" == "amd" ]]; then
  UV_PROJECT_ARGS=(--project envs/rocm)
fi
OUTPUT_DIR="${STAGING_ROOT}" uv run "${UV_PROJECT_ARGS[@]}" leap-finetune "${CONFIG_PATH}"

MERGED_DIR="$(find "${STAGING_ROOT}" -type f -name config.json -printf '%h\n' | while read -r candidate; do
  if compgen -G "${candidate}/*.safetensors" >/dev/null; then
    printf '%s\n' "${candidate}"
  fi
done | sort | tail -1)"
if [[ -z "${MERGED_DIR}" ]]; then
  echo "Training completed without a merged safetensors checkpoint" >&2
  exit 1
fi
mv "${MERGED_DIR}" "${OUTPUT}"
printf '%s\n' "${MERGED_DIR}" >"${OUTPUT}/source_merged_path.txt"
rm -rf -- "${STAGING_ROOT}"

test -f "${OUTPUT}/config.json"
compgen -G "${OUTPUT}/*.safetensors" >/dev/null
