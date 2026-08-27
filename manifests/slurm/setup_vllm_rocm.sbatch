#!/bin/bash
#SBATCH --job-name=qat-vllm-rocm-env
#SBATCH --partition=gpuworker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/OUT_%x.%j
#SBATCH --error=logs/ERR_%x.%j

set -euo pipefail

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi

: "${ROCM_VENV:?Set ROCM_VENV to a persistent shared path outside /tmp}"
REPO_ROOT="${LEAP_FINETUNE_ROOT:-$SLURM_SUBMIT_DIR}"
VLLM_ROCM_INDEX_URL="${VLLM_ROCM_INDEX_URL:-https://wheels.vllm.ai/rocm/799c3afa5d5b17b676d04e0b58a5628943bb4003/rocm722/}"

mkdir -p "$(dirname "$ROCM_VENV")" logs
uv venv --clear --python 3.12 "$ROCM_VENV"
uv pip install --python "$ROCM_VENV/bin/python" \
    --extra-index-url "$VLLM_ROCM_INDEX_URL" \
    --index-strategy unsafe-best-match \
    "torch==2.10.0+git8514f05" \
    "vllm==0.22.0+rocm722"
uv pip install --python "$ROCM_VENV/bin/python" "amd-quark==0.12.post1"
uv pip install --python "$ROCM_VENV/bin/python" \
    "transformers>=5.3.0,<5.4.0" \
    "huggingface_hub>=1.0" \
    "ray==2.51.1" \
    "numpy<2.5"
uv pip install --python "$ROCM_VENV/bin/python" \
    "trl==1.2.0" \
    "deepspeed>=0.17.1" \
    "wandb>=0.22.3" \
    "mpi4py>=4.1.0" \
    "turm>=0.13.0" \
    "trackio>=0.3" \
    "liger-kernel>=0.6.2"
uv pip install --python "$ROCM_VENV/bin/python" --no-deps -e "$REPO_ROOT"

"$ROCM_VENV/bin/python" - <<'PY'
import importlib.util

import torch
import vllm

required = ("accelerate", "datasets", "deepspeed", "peft", "trl", "wandb")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing training dependencies: {missing}")

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("accelerator", torch.cuda.get_device_name(0))
print("vllm", vllm.__version__)
if torch.version.hip is None:
    raise SystemExit("resolved a non-ROCm torch build")
PY
