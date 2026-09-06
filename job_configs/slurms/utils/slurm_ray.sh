#!/bin/bash

set -euo pipefail

RAY_SLURM_PIDS=()
RAY_SLURM_NODES=()
RAY_HEAD_NODE=""
RAY_HEAD_IP=""
RAY_PORT=""
RAY_ADDRESS=""
TOTAL_GPUS=0
RAY_TEMP_ROOT=""

_ray_slurm_detect_iface_for_ip() {
  local target_ip="$1"
  python - "$target_ip" <<'PY'
import re
import socket
import subprocess
import sys

target = sys.argv[1]

try:
    result = subprocess.run(
        ["ip", "-o", "route", "get", target],
        check=True,
        capture_output=True,
        text=True,
    )
except Exception:
    raise SystemExit(1)

match = re.search(r"\bdev\s+(\S+)", result.stdout)
if not match:
    raise SystemExit(1)

print(match.group(1), end="")
PY
}

ray_slurm_export_dist_env() {
  if [[ -z "${RAY_HEAD_IP}" ]]; then
    echo "ray_slurm_init must be called before ray_slurm_export_dist_env" >&2
    return 1
  fi

  local probe_ip="${RAY_HEAD_IP}"
  if (( ${#RAY_SLURM_NODES[@]} > 1 )); then
    local peer_node="${RAY_SLURM_NODES[1]}"
    local peer_ip
    peer_ip="$(srun --nodes=1 --ntasks=1 -w "${peer_node}" hostname -I | awk '{print $1}' || true)"
    if [[ -n "${peer_ip}" ]]; then
      probe_ip="${peer_ip}"
    fi
  fi

  local detected_iface="${LIQUID_SOCKET_IFNAME:-}"
  if [[ -z "${detected_iface}" ]]; then
    detected_iface="$(_ray_slurm_detect_iface_for_ip "${probe_ip}" || true)"
  fi

  if [[ -n "${detected_iface}" && "${detected_iface}" != "lo" ]]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${detected_iface}}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${detected_iface}}"
    echo "Using distributed network interface: ${detected_iface} (probe_ip=${probe_ip})"
  else
    echo "Warning: failed to auto-detect non-loopback distributed network interface for ${probe_ip}" >&2
  fi
}

ray_slurm_init() {
  local nnodes="$1"
  local gpus_per_node="$2"

  mapfile -t RAY_SLURM_NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
  if [[ "${#RAY_SLURM_NODES[@]}" -lt 1 ]]; then
    echo "Failed to resolve SLURM nodes from SLURM_NODELIST=${SLURM_NODELIST}" >&2
    return 1
  fi

  RAY_HEAD_NODE="${RAY_SLURM_NODES[0]}"
  RAY_PORT="${RAY_PORT:-6379}"
  RAY_HEAD_IP="$(srun --nodes=1 --ntasks=1 -w "${RAY_HEAD_NODE}" hostname -I | awk '{print $1}')"
  RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
  TOTAL_GPUS=$((nnodes * gpus_per_node))
  RAY_TEMP_ROOT="${RAY_TEMP_ROOT:-/tmp/ray-${USER}/${SLURM_JOB_ID}}"

  export RAY_HEAD_NODE RAY_HEAD_IP RAY_PORT RAY_ADDRESS TOTAL_GPUS RAY_TEMP_ROOT
}

ray_slurm_start_cluster_bg() {
  if [[ -z "${RAY_HEAD_NODE}" || -z "${RAY_ADDRESS}" ]]; then
    echo "ray_slurm_init must be called before ray_slurm_start_cluster_bg" >&2
    return 1
  fi

  local head_temp_dir="${RAY_TEMP_ROOT}/head"
  srun --nodes=1 --ntasks=1 -w "${RAY_HEAD_NODE}" mkdir -p "${head_temp_dir}"
  srun --nodes=1 --ntasks=1 -w "${RAY_HEAD_NODE}" \
    ray start --head --node-ip-address="${RAY_HEAD_IP}" --port="${RAY_PORT}" \
    --temp-dir="${head_temp_dir}" --disable-usage-stats --block &
  RAY_SLURM_PIDS+=("$!")

  sleep 5

  local node
  for node in "${RAY_SLURM_NODES[@]:1}"; do
    local worker_temp_dir="${RAY_TEMP_ROOT}/${node}"
    srun --nodes=1 --ntasks=1 -w "${node}" mkdir -p "${worker_temp_dir}"
    srun --nodes=1 --ntasks=1 -w "${node}" \
      ray start --address="${RAY_ADDRESS}" --temp-dir="${worker_temp_dir}" --disable-usage-stats --block &
    RAY_SLURM_PIDS+=("$!")
  done
}

ray_slurm_wait_ready() {
  local expected_nodes="$1"
  local expected_gpus="$2"
  local timeout_s="${3:-600}"
  local sleep_s="${4:-5}"
  local deadline=$((SECONDS + timeout_s))

  while (( SECONDS < deadline )); do
    if python - "${expected_nodes}" "${expected_gpus}" <<'PY'
import os
import sys
import ray

expected_nodes = int(sys.argv[1])
expected_gpus = float(sys.argv[2])
address = os.environ["RAY_ADDRESS"]

ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
cluster_gpus = float(ray.cluster_resources().get("GPU", 0.0))
ray.shutdown()

if len(alive_nodes) >= expected_nodes and cluster_gpus >= expected_gpus:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi

    sleep "${sleep_s}"
  done

  echo "Timed out waiting for Ray cluster readiness (address=${RAY_ADDRESS})" >&2
  return 1
}

ray_slurm_stop_cluster() {
  # The Ray processes are owned by background `srun ... ray start --block`
  # steps. Creating new `srun ray stop` steps here can deadlock teardown when
  # the allocation is still occupied by those blocking steps.
  local stop_timeout_s="${RAY_SLURM_STOP_TIMEOUT:-30}"
  local deadline=$((SECONDS + stop_timeout_s))
  local pid
  for pid in "${RAY_SLURM_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done

  for pid in "${RAY_SLURM_PIDS[@]:-}"; do
    while [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 1
    done
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
}
