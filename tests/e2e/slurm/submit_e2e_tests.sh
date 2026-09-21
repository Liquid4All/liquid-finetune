#!/usr/bin/env bash
set -euo pipefail

# Submit the complete E2E matrix as two SLURM jobs:
#   ray   - four GPUs, normal distributed/Ray cases
#   local - one GPU, explicit native local single-GPU cases
# Individual pytest cases may still be run directly for debugging.

usage() {
    cat <<'EOF'
Usage: tests/e2e/slurm/submit_e2e_tests.sh [--mode all|ray|local] [--dry-run]

With no mode, submit both the four-GPU Ray job and the one-GPU local job.

Environment overrides:
  JOB_NAME                 Base SLURM job name (default: lft_e2e_tests)
  PARTITION                Optional SLURM partition
  NODES                    Number of nodes (default: 1)
  RAY_GPUS_PER_TASK        Ray job GPUs (default: 4)
  LOCAL_GPUS_PER_TASK      Local job GPUs (default: 1)
  CPUS_PER_GPU             CPUs per GPU (default: 14)
  TIME_LIMIT               SLURM time limit (default: 06:00:00)
  OUTPUT_DIR               Root test result directory (default: <repo>/.test-results/e2e)
  TMP_ROOT                 Node temp root (default: /tmp)
  RAY_PYTEST_ARGS          Ray job pytest arguments
  LOCAL_PYTEST_ARGS        Local job pytest arguments; empty uses isolated per-file runs
  EXTRA_SBATCH_DIRECTIVES  Newline-separated extra #SBATCH directives
  VIRTUAL_ENV              Active environment to source in the batch jobs
  UV_PROJECT               If set to envs/rocm, use the ROCm project venv when available
EOF
}

MODE=all
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN=1
            ;;
        --mode=ray|--mode=local|--mode=all)
            MODE="${arg#--mode=}"
            ;;
        --ray)
            MODE=ray
            ;;
        --local)
            MODE=local
            ;;
        --all)
            MODE=all
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${MODE}" == "all" && -n "${PYTEST_ARGS:-}" ]]; then
    echo "PYTEST_ARGS is only valid with --mode=ray or --mode=local; use " \
        "RAY_PYTEST_ARGS/LOCAL_PYTEST_ARGS for --mode=all." >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SLURM_DIR="${ROOT_DIR}/tests/e2e/slurm/generated"
JOB_NAME="${JOB_NAME:-lft_e2e_tests}"
PARTITION="${PARTITION:-}"
NODES="${NODES:-1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-14}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
TMP_ROOT="${TMP_ROOT:-/tmp}"
OUTPUT_ROOT="${OUTPUT_DIR:-${ROOT_DIR}/.test-results/e2e}"
RAY_GPUS_PER_TASK="${RAY_GPUS_PER_TASK:-4}"
LOCAL_GPUS_PER_TASK="${LOCAL_GPUS_PER_TASK:-1}"
RAY_PYTEST_ARGS="${RAY_PYTEST_ARGS:-tests/e2e --dense --moe --vlm --retrieval}"
LOCAL_PYTEST_ARGS="${LOCAL_PYTEST_ARGS:-}"
EXTRA_SBATCH_DIRECTIVES="${EXTRA_SBATCH_DIRECTIVES:-}"

if [[ "${MODE}" != "all" && -n "${GPUS_PER_TASK:-}" ]]; then
    if [[ "${MODE}" == "ray" ]]; then
        RAY_GPUS_PER_TASK="${GPUS_PER_TASK}"
    else
        LOCAL_GPUS_PER_TASK="${GPUS_PER_TASK}"
    fi
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    VENV_ACTIVATE="${VIRTUAL_ENV}/bin/activate"
elif [[ "${UV_PROJECT:-}" == "envs/rocm" ]]; then
    VENV_ACTIVATE="${ROOT_DIR}/envs/rocm/.venv/bin/activate"
else
    VENV_ACTIVATE="${ROOT_DIR}/.venv/bin/activate"
fi

mkdir -p "${SLURM_DIR}" "${ROOT_DIR}/logs" "${OUTPUT_ROOT}" "${TMP_ROOT}"

write_script() {
    local mode="$1"
    local gpus="$2"
    local script_path="${SLURM_DIR}/e2e_${mode}.sh"
    local job_name="${JOB_NAME}_${mode}"
    local output_dir="${OUTPUT_ROOT}/${mode}"
    local pytest_args
    if [[ "${mode}" == "ray" ]]; then
        pytest_args="${RAY_PYTEST_ARGS}"
    else
        pytest_args="${LOCAL_PYTEST_ARGS}"
    fi

    mkdir -p "${output_dir}"
    {
        cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=${gpus}
#SBATCH --cpus-per-gpu=${CPUS_PER_GPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/OUT_%x.%j
#SBATCH --error=logs/ERR_%x.%j
EOF

        if [[ -n "${PARTITION}" ]]; then
            echo "#SBATCH --partition=${PARTITION}"
        fi

        if [[ -n "${EXTRA_SBATCH_DIRECTIVES}" ]]; then
            while IFS= read -r directive; do
                [[ -n "${directive}" ]] && echo "#SBATCH ${directive}"
            done <<< "${EXTRA_SBATCH_DIRECTIVES}"
        fi

        cat <<EOF

set -euo pipefail

cd "${ROOT_DIR}"
VENV_ACTIVATE="${VENV_ACTIVATE}"
if [[ ! -f "\${VENV_ACTIVATE}" ]]; then
  VENV_ACTIVATE="${ROOT_DIR}/.venv/bin/activate"
fi
source "\${VENV_ACTIVATE}"

export LEAP_E2E_MODE=${mode}
export LEAP_LAUNCHER=${mode}
export E2E_TMP_ROOT="${TMP_ROOT}"
export E2E_JOB_ROOT="${TMP_ROOT}/lft-e2e-\${SLURM_JOB_ID:-manual}-${mode}"
if (( \${#E2E_JOB_ROOT} > 35 )); then
  echo "TMP_ROOT produces an overly long Ray temp path: \${E2E_JOB_ROOT}" >&2
  echo "Set TMP_ROOT to a shorter node-local path (for example /tmp)." >&2
  exit 2
fi
export TMPDIR="\${E2E_JOB_ROOT}"
rm -rf -- "\${E2E_JOB_ROOT}"
mkdir -p "\${TMPDIR}"
if python - <<'PY' >/dev/null 2>&1
import sys
import torch
sys.exit(0 if getattr(torch.version, "hip", None) else 1)
PY
then
  if [[ -n "\${ROCR_VISIBLE_DEVICES:-}" && -z "\${HIP_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="\${ROCR_VISIBLE_DEVICES}"
  fi
  unset ROCR_VISIBLE_DEVICES
  unset CUDA_VISIBLE_DEVICES
else
  unset ROCR_VISIBLE_DEVICES
  unset HIP_VISIBLE_DEVICES
fi

export RAY_TMPDIR="\${E2E_JOB_ROOT}"
mkdir -p "\${RAY_TMPDIR}"
export TORCH_EXTENSIONS_DIR="\${TMPDIR}/torch_extensions"
export TRITON_CACHE_DIR="\${TMPDIR}/triton_cache"
export OUTPUT_DIR="${output_dir}"
export PYTHONUNBUFFERED=1
export RAY_DATA_DISABLE_PROGRESS_BARS=1

cleanup_e2e_temp() {
  local status=$?
  trap - EXIT TERM INT HUP
  if [[ -n "\${SLURM_JOB_ID:-}" ]]; then
    local expected_root="\${E2E_TMP_ROOT}/lft-e2e-\${SLURM_JOB_ID}-${mode}"
    if [[ "\${TMPDIR:-}" == "\${expected_root}" && "\${RAY_TMPDIR:-}" == "\${expected_root}" ]]; then
      rm -rf -- "\${TMPDIR}"
    fi
  fi
  exit "\${status}"
}
trap cleanup_e2e_temp EXIT TERM INT HUP
EOF

        if [[ "${mode}" == "local" && -z "${pytest_args}" ]]; then
            cat <<'EOF'
# Keep native local GRPO/vLLM cases in separate Python processes. The vLLM
# CuMem allocator permits only one colocated instance per process.
for test_file in \
    tests/e2e/test_dense_e2e.py \
    tests/e2e/test_grpo_e2e.py \
    tests/e2e/test_moe_e2e.py \
    tests/e2e/test_vlm_e2e.py \
    tests/e2e/test_vlm_grpo_e2e.py; do
  python -m pytest "${test_file}"
done
EOF
        else
            cat <<EOF
python -m pytest ${pytest_args}
EOF
        fi

        cat <<'EOF'

echo "================================================"
echo "E2E TESTS DONE"
echo "================================================"
EOF
    } > "${script_path}"
    chmod +x "${script_path}"
    echo "Generated SLURM script: ${script_path}"
    GENERATED_SCRIPTS+=("${script_path}")
}

submit_script() {
    local script_path="$1"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi
    sbatch "${script_path}"
}

GENERATED_SCRIPTS=()
if [[ "${MODE}" == "all" || "${MODE}" == "ray" ]]; then
    write_script ray "${RAY_GPUS_PER_TASK}"
fi
if [[ "${MODE}" == "all" || "${MODE}" == "local" ]]; then
    write_script local "${LOCAL_GPUS_PER_TASK}"
fi

if [[ "${DRY_RUN}" == "0" ]]; then
    for script_path in "${GENERATED_SCRIPTS[@]}"; do
        submit_script "${script_path}"
    done
fi
