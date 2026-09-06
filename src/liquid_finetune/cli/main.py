import argparse
import pathlib
import sys

import yaml


def _load_config_dict(config_path: pathlib.Path) -> dict:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f) or {}
    if not isinstance(config_dict, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config_dict


def _parse_cli_args():
    command = None
    config_path_arg = None
    output_dir_arg = None

    if len(sys.argv) > 1:
        if sys.argv[1] == "slurm":
            command = "slurm"
            parser = argparse.ArgumentParser(description="Generate SLURM script")
            parser.add_argument("command", choices=["slurm"])
            parser.add_argument("config_path", help="Path to YAML job config file")
            parser.add_argument(
                "--output-dir",
                "-o",
                help="Directory to save SLURM script",
                default=None,
            )
            args = parser.parse_args()
            config_path_arg = args.config_path
            output_dir_arg = args.output_dir
        elif sys.argv[1] == "eval":
            command = "eval"
            parser = argparse.ArgumentParser(description="Run standalone evals")
            parser.add_argument("command", choices=["eval"])
            parser.add_argument("config_path", help="Path to YAML eval config file")
            parser.add_argument(
                "--output",
                "-o",
                help="Optional JSON metrics output path",
                default=None,
            )
            args = parser.parse_args()
            config_path_arg = args.config_path
            output_dir_arg = args.output
        elif sys.argv[1] == "run":
            command = "run"
            parser = argparse.ArgumentParser(description="Run training job")
            parser.add_argument("command", choices=["run"])
            parser.add_argument(
                "config_path", help="Path to YAML job config file", nargs="?"
            )
            args = parser.parse_args()
            config_path_arg = args.config_path
        else:
            config_path_arg = sys.argv[1]

    return command, config_path_arg, output_dir_arg


def _generate_slurm_script(config_path_arg: str | None, output_dir_arg: str | None):
    from liquid_finetune.distribution.backends.slurm import generate_slurm_script
    from liquid_finetune.config.parser import resolve_config_path

    if not config_path_arg:
        print("No config file provided.")
        print("Usage: liquid-finetune slurm <path_to_config.yaml>")
        sys.exit(1)

    config_path = resolve_config_path(config_path_arg)
    config_dict = _load_config_dict(config_path)

    if output_dir_arg:
        output_dir = pathlib.Path(output_dir_arg)
    else:
        output_dir = config_path.parent / "slurms"

    generate_slurm_script(config_path, config_dict, output_dir, auto_submit=False)


def _run_standalone_eval(config_path_arg: str | None, output_path_arg: str | None):
    if not config_path_arg:
        print("No eval config file provided.")
        print("Usage: liquid-finetune eval <path_to_eval_config.yaml>")
        sys.exit(1)

    from liquid_finetune.evaluation.runner import run_eval_config

    results = run_eval_config(config_path_arg, output_path=output_path_arg)
    print(yaml.safe_dump(results, sort_keys=True))


def _assert_local_cuda_available() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
    except (ImportError, RuntimeError):
        print("Local training requires GPU dependencies and CUDA.")
        print(
            "For remote execution, add a 'modal:' or 'slurm:' section to your config."
        )
        sys.exit(1)


def check_and_handle_slurm(
    config_path_arg: str | None = None,
    *,
    config_dict: dict | None = None,
) -> bool:
    from liquid_finetune.distribution.backends.slurm import (
        check_and_handle_slurm as _impl,
    )

    return _impl(config_path_arg, config_dict=config_dict)


def run_config(config_path, *, output_path: str | pathlib.Path | None = None):
    """Launch a training job or standalone eval from a config path/model.

    This is the programmatic equivalent of `liquid-finetune <config>`. Training
    configs keep the same backend dispatch behavior: configs with `slurm`,
    `kuberay`, or `modal` sections submit remotely; other training configs
    launch local Ray training. Eval-only configs run benchmarks without
    starting training.
    """
    from liquid_finetune.config import EvalRunConfig, JobConfig
    from liquid_finetune.config.parser import (
        materialize_job_config,
        normalized_job_config_dict,
        parse_eval_config,
        parse_job_config,
        print_job_config_summary,
    )

    parsed_job = None
    config_dict = None
    config_path_arg = None
    if isinstance(config_path, EvalRunConfig):
        from liquid_finetune.evaluation.runner import run_eval_config as _run_eval_config

        return _run_eval_config(config_path, output_path=output_path)
    if isinstance(config_path, JobConfig):
        parsed_job = config_path
        config_dict = normalized_job_config_dict(parsed_job)
    else:
        config_path_arg = str(config_path)

    # === Remote backend dispatch ===
    # Keep this path light: Slurm/Modal submission should not import torch, Ray,
    # PEFT, or datasets unless we are actually launching local training.
    if check_and_handle_slurm(config_path_arg, config_dict=config_dict):
        return

    from liquid_finetune.distribution.backends.kuberay import check_and_handle_kuberay

    if check_and_handle_kuberay(config_path_arg, config_dict=config_dict):
        return

    from liquid_finetune.distribution.backends.modal import check_and_handle_modal

    if check_and_handle_modal(config_path_arg, config_dict=config_dict):
        return

    if parsed_job is None:
        try:
            eval_config = parse_eval_config(config_path_arg)
        except Exception:
            eval_config = None
        else:
            from liquid_finetune.evaluation.runner import (
                run_eval_config as _run_eval_config,
            )

            return _run_eval_config(eval_config, output_path=output_path)

    _assert_local_cuda_available()

    # Heavy imports deferred to here to keep remote-submit codepaths fast.
    from liquid_finetune.data_loading.dataset_loader import DatasetLoader
    from liquid_finetune.distribution.local_trainer import local_trainer, should_use_local
    from liquid_finetune.distribution.ray_trainer import ray_trainer
    from liquid_finetune.training.utils.logging import setup_training_environment

    setup_training_environment()

    print("Launching liquid-finetune")

    try:
        if parsed_job is None:
            parsed_job = parse_job_config(config_path_arg)
        job_config = materialize_job_config(parsed_job)
        print_job_config_summary(job_config)

        if isinstance(job_config.dataset, DatasetLoader):
            job_config.dataset.quick_validate()

        job_config_dict = job_config.to_dict()
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path_arg}")
    except Exception as e:
        raise ValueError(f"Issue parsing configuration: {e}") from e

    trainer = local_trainer if should_use_local(job_config_dict) else ray_trainer
    return trainer(job_config_dict)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "env":
        from liquid_finetune.cli.env import main as env_main

        sys.exit(env_main(sys.argv[2:]))

    command, config_path_arg, output_dir_arg = _parse_cli_args()

    if command == "slurm":
        _generate_slurm_script(config_path_arg, output_dir_arg)
        return
    if command == "eval":
        _run_standalone_eval(config_path_arg, output_dir_arg)
        return

    if not config_path_arg:
        print("No config file provided. Please provide a path to a YAML config file.")
        print("Usage: liquid-finetune <path_to_config.yaml>")
        print("   or: liquid-finetune slurm <path_to_config.yaml>")
        print("   or: liquid-finetune eval <path_to_eval_config.yaml>")
        print("   or: liquid-finetune env fa2-status")
        sys.exit(1)

    run_config(config_path_arg)
