#!/usr/bin/env bash
set -euo pipefail

case "${DITTOPILOT_CLUSTER:-local}" in
  amd)
    exec uv sync --project envs/rocm
    ;;
  h100 | local)
    exec uv sync
    ;;
  *)
    echo "Unsupported Dittopilot cluster: ${DITTOPILOT_CLUSTER}" >&2
    exit 2
    ;;
esac
