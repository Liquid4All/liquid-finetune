#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=""
DATASET=""
OUTPUT=""
RUN_NAME=""
LEARNING_RATE="1e-4"
BATCH_SIZE="8192"
MICRO_BATCH_SIZE="32"
COLLATE_WORKERS="0"
COLLATE_BACKEND="thread"
USE_PEFT="true"
TEXT_ATTN_IMPLEMENTATION="sdpa"
CACHED_LOSS="true"
SORT_ROWS="true"
RETAIN_GRAPHS="false"
MAX_TEXT_TOKENS="512"
GRADIENT_ACCUMULATION_STEPS="1"
GRADIENT_CHECKPOINTING="false"
GRADIENT_CHECKPOINTING_SCOPE="auto"
MAX_STEPS="80"
TEMPERATURE="0.02"
MAX_IMAGE_TOKENS="256"
MIN_IMAGE_TOKENS="64"
POOLING="last"
LR_SCHEDULER="linear"
TEACHER_EMBEDDINGS=""
DISTILL_WEIGHT="0"
FEATURE_DISTILL_WEIGHT="0"
FEATURE_HEAD_LR="1e-3"
FEATURE_PROJECTION="learned"
SAME_TOWER_NEGATIVES="false"
SPREAD_OUT_WEIGHT="0"
TEACHER_TEMPERATURE="0.02"
LORA_RANK="64"
LORA_ALPHA="64"
WARMUP_RATIO="0.05"
WEIGHT_DECAY="0.01"
SAVE_STEPS="10"
TRACKER="wandb"
SEED="20261019"
RESUME_FROM="auto"

while (($#)); do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --micro-batch-size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --collate-workers) COLLATE_WORKERS="$2"; shift 2 ;;
    --collate-backend) COLLATE_BACKEND="$2"; shift 2 ;;
    --use-peft) USE_PEFT="$2"; shift 2 ;;
    --text-attn-implementation) TEXT_ATTN_IMPLEMENTATION="$2"; shift 2 ;;
    --cached-loss) CACHED_LOSS="$2"; shift 2 ;;
    --sort-rows) SORT_ROWS="$2"; shift 2 ;;
    --retain-graphs) RETAIN_GRAPHS="$2"; shift 2 ;;
    --max-text-tokens) MAX_TEXT_TOKENS="$2"; shift 2 ;;
    --gradient-accumulation-steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --gradient-checkpointing) GRADIENT_CHECKPOINTING="$2"; shift 2 ;;
    --gradient-checkpointing-scope) GRADIENT_CHECKPOINTING_SCOPE="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --max-image-tokens) MAX_IMAGE_TOKENS="$2"; shift 2 ;;
    --min-image-tokens) MIN_IMAGE_TOKENS="$2"; shift 2 ;;
    --pooling) POOLING="$2"; shift 2 ;;
    --lr-scheduler) LR_SCHEDULER="$2"; shift 2 ;;
    --teacher-embeddings) TEACHER_EMBEDDINGS="$2"; shift 2 ;;
    --distill-weight) DISTILL_WEIGHT="$2"; shift 2 ;;
    --feature-distill-weight) FEATURE_DISTILL_WEIGHT="$2"; shift 2 ;;
    --feature-head-lr) FEATURE_HEAD_LR="$2"; shift 2 ;;
    --feature-projection) FEATURE_PROJECTION="$2"; shift 2 ;;
    --same-tower-negatives) SAME_TOWER_NEGATIVES="$2"; shift 2 ;;
    --spread-out-weight) SPREAD_OUT_WEIGHT="$2"; shift 2 ;;
    --teacher-temperature) TEACHER_TEMPERATURE="$2"; shift 2 ;;
    --lora-rank) LORA_RANK="$2"; shift 2 ;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
    --warmup-ratio) WARMUP_RATIO="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --save-steps) SAVE_STEPS="$2"; shift 2 ;;
    --tracker) TRACKER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${GRADIENT_CHECKPOINTING}" in
  true|false) ;;
  *) echo "gradient-checkpointing must be true or false" >&2; exit 2 ;;
esac
case "${POOLING}" in
  last|mean) ;;
  *) echo "pooling must be last or mean" >&2; exit 2 ;;
esac
case "${LR_SCHEDULER}" in
  linear|cosine|constant_with_warmup) ;;
  *) echo "lr-scheduler must be linear, cosine, or constant_with_warmup" >&2; exit 2 ;;
esac
if [[ "${TEACHER_EMBEDDINGS}" == "none" ]]; then
  TEACHER_EMBEDDINGS=""
fi
if [[ -n "${TEACHER_EMBEDDINGS}" && ! -f "${TEACHER_EMBEDDINGS}" ]]; then
  echo "teacher-embeddings file not found: ${TEACHER_EMBEDDINGS}" >&2
  exit 2
fi
if ! [[ "${MIN_IMAGE_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "min-image-tokens must be a positive integer" >&2
  exit 2
fi
if ! [[ "${MAX_TEXT_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "max-text-tokens must be a non-negative integer (0 disables truncation)" >&2
  exit 2
fi
if ! [[ "${COLLATE_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "collate-workers must be a non-negative integer" >&2
  exit 2
fi
case "${COLLATE_BACKEND}" in
  thread|process) ;;
  *) echo "collate-backend must be thread or process" >&2; exit 2 ;;
esac
case "${USE_PEFT}" in
  true|false) ;;
  *) echo "use-peft must be true or false" >&2; exit 2 ;;
esac
case "${TEXT_ATTN_IMPLEMENTATION}" in
  sdpa|flash_attention_2) ;;
  *) echo "text-attn-implementation must be sdpa or flash_attention_2" >&2; exit 2 ;;
esac
case "${CACHED_LOSS}" in
  true|false) ;;
  *) echo "cached-loss must be true or false" >&2; exit 2 ;;
esac
case "${SORT_ROWS}" in
  true|false) ;;
  *) echo "sort-rows must be true or false" >&2; exit 2 ;;
esac
case "${FEATURE_PROJECTION}" in
  learned|truncate) ;;
  *) echo "feature-projection must be learned or truncate" >&2; exit 2 ;;
esac
case "${SAME_TOWER_NEGATIVES}" in
  true|false) ;;
  *) echo "same-tower-negatives must be true or false" >&2; exit 2 ;;
esac
case "${RETAIN_GRAPHS}" in
  true|false) ;;
  *) echo "retain-graphs must be true or false" >&2; exit 2 ;;
esac
case "${GRADIENT_CHECKPOINTING_SCOPE}" in
  auto|all|language|vision|none) ;;
  *) echo "gradient-checkpointing-scope must be auto, all, language, vision, or none" >&2; exit 2 ;;
esac
case "${TRACKER}" in
  wandb|none) ;;
  *) echo "tracker must be wandb or none" >&2; exit 2 ;;
esac

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
# auto: a requeued job finds its own staging root and resumes from the newest
# `latest` checkpoint in it. A path resumes that checkpoint; the trainer then
# writes into the checkpoint's run directory. none: always start fresh.
RUN_ROOT="${STAGING_ROOT}"
RESUME_LINE=""
case "${RESUME_FROM}" in
  none)
    if [[ -e "${STAGING_ROOT}" ]]; then
      echo "Refusing to overwrite staging path: ${STAGING_ROOT}" >&2
      exit 2
    fi ;;
  auto)
    if [[ -e "${STAGING_ROOT}" ]]; then
      echo "Resuming from the latest checkpoint under ${STAGING_ROOT}"
      RESUME_LINE='  resume_from_checkpoint: "latest"'
    fi ;;
  *)
    if [[ ! -f "${RESUME_FROM}/trainer_state.json" ]]; then
      echo "resume-from is not a Trainer checkpoint: ${RESUME_FROM}" >&2
      exit 2
    fi
    RESUME_FROM="$(realpath "${RESUME_FROM}")"
    RUN_ROOT="$(dirname "${RESUME_FROM}")"
    RESUME_LINE="  resume_from_checkpoint: \"${RESUME_FROM}\"" ;;
esac
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
  # No in-run eval: otherwise the parser holds out 20% of the training file,
  # and every Ray worker re-executes the full read/shuffle to build that split.
  eval_strategy: "no"
  max_steps: ${MAX_STEPS}
  global_batch_size: ${BATCH_SIZE}
  per_device_train_batch_size: ${BATCH_SIZE}
  per_device_eval_batch_size: ${BATCH_SIZE}
  micro_batch_size: ${MICRO_BATCH_SIZE}
  collate_workers: ${COLLATE_WORKERS}
  collate_backend: "${COLLATE_BACKEND}"
  text_attn_implementation: "${TEXT_ATTN_IMPLEMENTATION}"
  cached_loss: ${CACHED_LOSS}
  sort_rows: ${SORT_ROWS}
  retain_graphs: ${RETAIN_GRAPHS}
  max_text_tokens: ${MAX_TEXT_TOKENS}
  gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
  gradient_checkpointing: ${GRADIENT_CHECKPOINTING}
  gradient_checkpointing_scope: "${GRADIENT_CHECKPOINTING_SCOPE}"
  learning_rate: ${LEARNING_RATE}
  lr_scheduler_type: "${LR_SCHEDULER}"
  warmup_ratio: ${WARMUP_RATIO}
  weight_decay: ${WEIGHT_DECAY}
  seed: ${SEED}
  data_seed: ${SEED}
  temperature: ${TEMPERATURE}
  teacher_embeddings: "${TEACHER_EMBEDDINGS}"
  distill_weight: ${DISTILL_WEIGHT}
  feature_distill_weight: ${FEATURE_DISTILL_WEIGHT}
  feature_head_lr: ${FEATURE_HEAD_LR}
  feature_projection: "${FEATURE_PROJECTION}"
  same_tower_negatives: ${SAME_TOWER_NEGATIVES}
  spread_out_weight: ${SPREAD_OUT_WEIGHT}
  teacher_temperature: ${TEACHER_TEMPERATURE}
  max_image_tokens: ${MAX_IMAGE_TOKENS}
  min_image_tokens: ${MIN_IMAGE_TOKENS}
  pooling: "${POOLING}"
  eval_strategy: "no"
  eval_on_start: false
  dataloader_drop_last: false
  save_strategy: "steps"
  save_steps: ${SAVE_STEPS}
  save_total_limit: 2
  logging_steps: 1
  dataloader_num_workers: 8
  dataloader_prefetch_factor: 4
  leap_run_name_template: "vlm-embedding-lora_a"
  tracker: "${TRACKER}"
${RESUME_LINE}
peft_config:
  extends: "DEFAULT_VLM_LORA"
  use_peft: ${USE_PEFT}
  r: ${LORA_RANK}
  lora_alpha: ${LORA_ALPHA}
  lora_dropout: 0.0
EOF

if [[ "${TRACKER}" == "wandb" ]]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_ENTITY="${WANDB_ENTITY:-liquid-ai}"
  export WANDB_PROJECT="${WANDB_PROJECT:-retrieval-lfm}"
  export WANDB_DIR="${WANDB_DIR:-$(dirname "${OUTPUT}")/wandb}"
  mkdir -p "${WANDB_DIR}"
fi

# Ray's AF_UNIX socket paths must stay under 107 bytes; the per-job TMPDIR
# (which embeds the user name and job id twice) exceeds that on this cluster.
# Job-local scratch: Ray socket paths must stay under the AF_UNIX limit, and
# MIOpen's per-user kernel database in $HOME races across concurrent jobs.
JOB_SCRATCH="/tmp/vlm-${SLURM_JOB_ID:-$$}"
mkdir -p "${JOB_SCRATCH}"
trap 'rm -rf -- "${JOB_SCRATCH}"' EXIT
export RAY_TMPDIR="${RAY_TMPDIR:-${JOB_SCRATCH}/ray}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-${JOB_SCRATCH}/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-${JOB_SCRATCH}/miopen}"
mkdir -p "${RAY_TMPDIR}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

UV_PROJECT_ARGS=()
if [[ "${DITTOPILOT_CLUSTER:-local}" == "amd" ]]; then
  UV_PROJECT_ARGS=(--project envs/rocm)
fi
OUTPUT_DIR="${STAGING_ROOT}" uv run "${UV_PROJECT_ARGS[@]}" leap-finetune "${CONFIG_PATH}"

# Full fine-tuning saves final/; intermediate checkpoint-N dirs also carry
# config.json, so never fall back to sorting them.
MERGED_DIR="$(find "${RUN_ROOT}" -type f -path '*/final/config.json' -printf '%h\n' | head -1)"
[[ "${USE_PEFT}" == "false" ]] || MERGED_DIR="$(find "${RUN_ROOT}" -type f -name config.json -printf '%h\n' | while read -r candidate; do
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
