import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any

import yaml

from liquid_finetune import LIQUID_FINETUNE_DIR


_PASSTHROUGH_ENV_VARS = (
    "NCCL_IB_DISABLE",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_SOCKET_IFNAME",
    "NCCL_SOCKET_FAMILY",
    "GLOO_SOCKET_IFNAME",
    "TORCH_DISTRIBUTED_DEBUG",
    "LIQUID_DISABLE_DATASETS_TORCH_SHM",
    "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE",
    "LIQUID_SOCKET_IFNAME",
)


def check_and_handle_slurm(
    config_path_arg: str | None = None,
    *,
    config_dict: dict | None = None,
) -> bool:
    if os.environ.get("LIQUID_FINETUNE_FROM_SLURM") == "1":
        return False

    config_path = None
    if config_dict is None:
        if not config_path_arg:
            return False

        from liquid_finetune.config.parser import resolve_config_path

        try:
            config_path = resolve_config_path(config_path_arg)
        except FileNotFoundError:
            return False

        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}

        if not isinstance(config_dict, dict):
            raise ValueError(f"Config must be a YAML mapping: {config_path}")

    slurm_config = config_dict.get("slurm") if isinstance(config_dict, dict) else None
    if not slurm_config:
        return False

    if config_path is None:
        job_name = (
            config_dict.get("project_name")
            or config_dict.get("job_name")
            or "liquid_finetune"
        )
        output_dir = pathlib.Path.cwd() / "slurms"
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / f"{job_name}.yaml"
        config_path.write_text(yaml.safe_dump(config_dict, sort_keys=False))
        script_path = generate_slurm_script(
            config_path, config_dict, output_dir, auto_submit=False
        )
        print(
            f"Config contains SLURM settings - generated submission artifacts: {script_path}"
        )
    else:
        output_dir = config_path.parent / "slurms"
        script_path = output_dir / f"{config_path.stem}.sh"
        if script_path.exists():
            print(
                f"Config contains SLURM settings - using existing script: {script_path}"
            )
        else:
            print("Config contains SLURM settings - generating SLURM script...")
            script_path = generate_slurm_script(
                config_path, config_dict, output_dir, auto_submit=False
            )

    print("Submitting SLURM job...")
    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Failed to submit job: {result.stderr}")
        sys.exit(1)

    print(f"SLURM job submitted: {result.stdout.strip()}")
    return True


def _default_judge_gpus(config_dict: dict[str, Any]) -> int:
    rewards = config_dict.get("rewards")
    if not isinstance(rewards, dict):
        return 0

    judge = rewards.get("judge")
    if not judge:
        return 0
    if isinstance(judge, dict) and judge.get("base_url"):
        return 0

    rollout = config_dict.get("grpo_rollout") or {}
    return int(rollout.get("judge_gpus", 1))


def _default_gpus_per_task(config_dict: dict[str, Any]) -> int:
    training_config = config_dict.get("training_config", {})
    if config_dict.get("training_type") not in ("grpo", "vlm_grpo"):
        return 1

    rollout = config_dict.get("grpo_rollout") or {}
    judge_gpus = _default_judge_gpus(config_dict)
    if training_config.get("vllm_mode") != "server":
        return max(1, judge_gpus + int(rollout.get("training_gpus", 1)))

    if "server_gpus" in rollout:
        server_gpus = int(rollout["server_gpus"])
    elif "dedicated_gpus" in rollout:
        server_gpus = int(rollout["dedicated_gpus"])
    elif "training_gpus" in rollout:
        server_gpus = 1
    else:
        server_gpus = 1

    training_gpus = int(rollout.get("training_gpus", 1))
    if server_gpus == 0:
        return max(1, judge_gpus + training_gpus)
    return max(1, server_gpus + judge_gpus + training_gpus)


def _render_export_block(is_multinode: bool) -> str:
    lines = [
        "export LIQUID_FINETUNE_FROM_SLURM=1",
        "export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}",
        "if python - <<'PY' >/dev/null 2>&1",
        "import sys",
        "import torch",
        "sys.exit(0 if getattr(torch.version, 'hip', None) else 1)",
        "PY",
        "then",
        '  if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then',
        '    export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"',
        "  fi",
        "  unset ROCR_VISIBLE_DEVICES",
        "  unset CUDA_VISIBLE_DEVICES",
        "else",
        "  unset ROCR_VISIBLE_DEVICES",
        "  unset HIP_VISIBLE_DEVICES",
        "fi",
    ]

    if is_multinode:
        # Multi-node defaults should prefer the working CP/FSDP transport path.
        defaults = {
            "NCCL_IB_DISABLE": "0",
            "LIQUID_DISABLE_DATASETS_TORCH_SHM": "1",
            "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": "1",
        }
        for key, value in defaults.items():
            lines.append(f'export {key}="${{{key}:-{value}}}"')

    for key in _PASSTHROUGH_ENV_VARS:
        if (
            key
            in {
                "NCCL_IB_DISABLE",
                "LIQUID_DISABLE_DATASETS_TORCH_SHM",
                "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE",
            }
            and is_multinode
        ):
            continue

        value = os.environ.get(key)
        if value:
            lines.append(f"export {key}={shlex.quote(value)}")

    return "\n".join(lines)


def render_venv_activation_block(project_root: pathlib.Path) -> str:
    """Render shell that activates the current backend venv when available."""
    root_activate = project_root / ".venv" / "bin" / "activate"
    active_venv = os.environ.get("VIRTUAL_ENV")
    active_activate = (
        pathlib.Path(active_venv) / "bin" / "activate" if active_venv else root_activate
    )
    return "\n".join(
        [
            f"VENV_ACTIVATE={shlex.quote(str(active_activate))}",
            'if [[ ! -f "${VENV_ACTIVATE}" ]]; then',
            f"  VENV_ACTIVATE={shlex.quote(str(root_activate))}",
            "fi",
            'source "${VENV_ACTIVATE}"',
        ]
    )


def generate_slurm_script(
    config_path: pathlib.Path,
    config_dict: dict[str, Any],
    output_dir: pathlib.Path | None = None,
    auto_submit: bool = False,
) -> pathlib.Path:
    slurm_config = config_dict.get("slurm", {})

    defaults = {
        "job_name": config_dict.get("project_name", "liquid_finetune"),
        "nodes": 1,
        "ntasks_per_node": 1,
        "gpus_per_node": None,
        "gpus_per_task": _default_gpus_per_task(config_dict),
        "cpus_per_gpu": 8,
        "output": "logs/OUT_%x.%j",
        "error": "logs/ERR_%x.%j",
    }

    slurm_settings = {**defaults, **slurm_config}

    if output_dir is None:
        output_dir = config_path.parent
    else:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    config_name = config_path.stem
    script_path = output_dir / f"{config_name}.sh"

    project_root = LIQUID_FINETUNE_DIR
    ray_helper_path = (
        project_root
        / "src"
        / "liquid_finetune"
        / "distribution"
        / "backends"
        / "slurm_ray.sh"
    )

    config_relative_path = (
        config_path.relative_to(project_root)
        if config_path.is_relative_to(project_root)
        else config_path
    )

    gpus_per_node = slurm_settings.get("gpus_per_node")
    gpus_directive = (
        f"#SBATCH --gpus-per-node={gpus_per_node}"
        if gpus_per_node is not None
        else f"#SBATCH --gpus-per-task={slurm_settings['gpus_per_task']}"
    )

    script_content = f"""#!/bin/bash

#SBATCH --job-name={slurm_settings["job_name"]}
#SBATCH --nodes={slurm_settings["nodes"]}
#SBATCH --ntasks-per-node={slurm_settings["ntasks_per_node"]}
{gpus_directive}
#SBATCH --output={slurm_settings["output"]}
#SBATCH --error={slurm_settings["error"]}
#SBATCH --cpus-per-gpu={slurm_settings["cpus_per_gpu"]}
"""

    additional_directives = slurm_config.get("directives", [])
    for directive in additional_directives:
        script_content += f"#SBATCH {directive}\n"

    setup_commands = slurm_config.get("setup_commands", [])
    setup_block = "\n".join(setup_commands) + "\n" if setup_commands else ""

    script_content += f"""
cd {project_root}

set -euo pipefail

{render_venv_activation_block(project_root)}

{setup_block}
{_render_export_block(is_multinode=int(slurm_settings["nodes"]) > 1)}
"""

    is_multinode = int(slurm_settings["nodes"]) > 1
    if slurm_settings.get("gpus_per_node") is not None:
        gpus_per_node = int(slurm_settings["gpus_per_node"])
    else:
        gpus_per_node = int(slurm_settings["gpus_per_task"]) * int(
            slurm_settings["ntasks_per_node"]
        )

    if is_multinode:
        script_content += f"""
export PYTHONUNBUFFERED=1
export LIQUID_RAY_NUM_WORKERS=$((SLURM_NNODES * {gpus_per_node}))

# shellcheck source={ray_helper_path}
source {shlex.quote(str(ray_helper_path))}

ray_slurm_init "${{SLURM_NNODES}}" "{gpus_per_node}"
ray_slurm_export_dist_env
trap ray_slurm_stop_cluster EXIT
ray_slurm_start_cluster_bg
ray_slurm_wait_ready "${{SLURM_NNODES}}" "${{TOTAL_GPUS}}" 600 5

echo "Ray cluster up: ${{TOTAL_GPUS}} GPUs across ${{SLURM_NNODES}} nodes (RAY_ADDRESS=${{RAY_ADDRESS}})"

export RAY_ADDRESS
liquid-finetune {config_relative_path}
"""
    else:
        script_content += f"""
liquid-finetune {config_relative_path}
"""

    script_content += """

echo "================================================"
echo "RUN DONE"
echo "================================================"
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    print(f"Generated SLURM script: {script_path}")

    if auto_submit:
        import subprocess

        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Submitted job: {result.stdout.strip()}")
        else:
            print(f"Failed to submit job: {result.stderr}")

    return script_path
