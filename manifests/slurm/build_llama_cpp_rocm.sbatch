#!/bin/bash
#SBATCH --job-name=qat-llama-rocm
#SBATCH --partition=gpuworker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/OUT_%x.%j
#SBATCH --error=logs/ERR_%x.%j

set -euo pipefail

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi

: "${LLAMA_CPP_ROOT:?Set LLAMA_CPP_ROOT to a persistent shared path}"
LLAMA_CPP_REVISION="${LLAMA_CPP_REVISION:-c0bc8591e8815c63cb01dd3f051a8b0df02501c9}"

mkdir -p "$(dirname "$LLAMA_CPP_ROOT")" logs
if [ ! -d "$LLAMA_CPP_ROOT/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_ROOT"
fi

git -C "$LLAMA_CPP_ROOT" fetch --depth 1 origin "$LLAMA_CPP_REVISION"
git -C "$LLAMA_CPP_ROOT" checkout --detach FETCH_HEAD
cmake -S "$LLAMA_CPP_ROOT" -B "$LLAMA_CPP_ROOT/build" \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx942 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP_ROOT/build" --config Release -j "${SLURM_CPUS_PER_TASK:-16}" \
    --target llama-server llama-quantize llama-cli llama-bench

"$LLAMA_CPP_ROOT/build/bin/llama-server" --version
test -x "$LLAMA_CPP_ROOT/build/bin/llama-quantize"
