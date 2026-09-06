#!/usr/bin/env bash
set -euo pipefail

# Submit the GPU e2e pytest suite as one SLURM allocation. This keeps the
# tests' normal in-process assertions while letting Ray/torch see SLURM GPUs.

usage() {
    cat <<'EOF'
Usage: tests/e2e/slurm/submit_e2e_tests.sh [--dry-run]

Environment overrides:
  JOB_NAME                 SLURM job name (default: liquid_e2e_tests)
  PARTITION                Optional SLURM partition
  NODES                    Number of nodes (default: 1)
  GPUS_PER_TASK            GPUs in the allocation (default: 4)
  CPUS_PER_GPU             CPUs per GPU (default: 14)
  TIME_LIMIT               SLURM time limit (default: 06:00:00)
  OUTPUT_DIR               Test result directory (default: <repo>/.test-results/e2e)
  TMP_ROOT                 Node temp root (default: /tmp/$USER)
  PYTEST_ARGS              pytest args to run inside SLURM
  EXTRA_SBATCH_DIRECTIVES  Newline-separated extra #SBATCH directives
EOF
}

DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN=1
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SLURM_DIR="${ROOT_DIR}/tests/e2e/slurm/generated"
SCRIPT_PATH="${SLURM_DIR}/e2e_tests.sh"

JOB_NAME="${JOB_NAME:-liquid_e2e_tests}"
PARTITION="${PARTITION:-}"
NODES="${NODES:-1}"
GPUS_PER_TASK="${GPUS_PER_TASK:-4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-14}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/.test-results/e2e}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}}"
PYTEST_ARGS="${PYTEST_ARGS:-tests/e2e --dense --moe --vlm --retrieval}"
EXTRA_SBATCH_DIRECTIVES="${EXTRA_SBATCH_DIRECTIVES:-}"
VENV_ACTIVATE="${VIRTUAL_ENV:-${ROOT_DIR}/.venv}/bin/activate"
ROOT_VENV_ACTIVATE="${ROOT_DIR}/.venv/bin/activate"

mkdir -p "${SLURM_DIR}" "${ROOT_DIR}/logs" "${OUTPUT_DIR}" "${TMP_ROOT}"

{
    cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=${GPUS_PER_TASK}
#SBATCH --cpus-per-gpu=${CPUS_PER_GPU}
#SBATCH --time=${TIME_LIMIT}
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

cd ${ROOT_DIR}
VENV_ACTIVATE=${VENV_ACTIVATE}
if [[ ! -f "\${VENV_ACTIVATE}" ]]; then
  VENV_ACTIVATE=${ROOT_VENV_ACTIVATE}
fi
source "\${VENV_ACTIVATE}"

export TMPDIR=${TMP_ROOT}/liquid-e2e-\${SLURM_JOB_ID:-manual}
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

export RAY_TMPDIR=/tmp/r\${SLURM_JOB_ID:-manual}
mkdir -p "\${RAY_TMPDIR}"
export TORCH_EXTENSIONS_DIR=\${TMPDIR}/torch_extensions
export TRITON_CACHE_DIR=\${TMPDIR}/triton_cache
export OUTPUT_DIR=${OUTPUT_DIR}
export PYTHONUNBUFFERED=1
export RAY_DATA_DISABLE_PROGRESS_BARS=1

python -m pytest ${PYTEST_ARGS}

echo "================================================"
echo "E2E TESTS DONE"
echo "================================================"
EOF
} > "${SCRIPT_PATH}"

chmod +x "${SCRIPT_PATH}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Generated SLURM script: ${SCRIPT_PATH}"
else
    sbatch "${SCRIPT_PATH}"
fi
