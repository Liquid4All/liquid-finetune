import pathlib
import sys

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_HF_SECRET_NAME = "huggingface-secret"


def check_and_handle_modal(
    config_path_arg: str | None = None,
    *,
    config_dict: dict | None = None,
) -> bool:
    if config_dict is None:
        if not config_path_arg:
            return False

        try:
            from liquid_finetune.config.parser import resolve_config_path

            config_path = resolve_config_path(config_path_arg)
        except Exception:
            return False

        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        except SystemExit:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"\nError submitting Modal job: {e}")
            sys.exit(1)

    modal_cfg = config_dict.get("modal") if isinstance(config_dict, dict) else None
    if not modal_cfg:
        return False

    try:
        print("Config contains Modal settings - submitting Modal job...\n")
        _preflight_checks(config_dict, modal_cfg)
        _print_config_summary(config_dict, modal_cfg)
        _submit(config_dict, modal_cfg)
        return True
    except SystemExit:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nError submitting Modal job: {e}")
        sys.exit(1)


# === Preflight checks ===


def _preflight_checks(config_dict: dict, modal_cfg: dict) -> None:
    # HF token is needed for model downloads (rate limits / gated models) and trackio
    token = _get_local_hf_token()
    if not token:
        print("Error: HF token not found. Required for model downloads on Modal.")
        print("  Run: huggingface-cli login")
        sys.exit(1)

    # Auto-add and auto-create the Modal secret from local token
    secrets = modal_cfg.get("secrets") or []
    if _HF_SECRET_NAME not in secrets:
        secrets.append(_HF_SECRET_NAME)
        modal_cfg["secrets"] = secrets

    _ensure_modal_secret(token)


def _get_local_hf_token() -> str | None:
    try:
        import huggingface_hub

        return huggingface_hub.get_token()
    except Exception:
        return None


def _ensure_modal_secret(token: str) -> None:
    import subprocess

    result = subprocess.run(
        ["modal", "secret", "list", "--json"], capture_output=True, text=True
    )
    if result.returncode == 0:
        import json

        existing = {s["Name"] for s in json.loads(result.stdout)}
        if _HF_SECRET_NAME in existing:
            return

    import modal

    print(f"Creating Modal secret '{_HF_SECRET_NAME}' from local HF token...")
    modal.Secret.create_deployed(_HF_SECRET_NAME, {"HF_TOKEN": token})
    print(f"  Created '{_HF_SECRET_NAME}' on Modal.\n")


# === Config summary ===


def _print_config_summary(config_dict: dict, modal_cfg: dict) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    train_cfg = config_dict.get("training_config", {})
    ds_cfg = config_dict.get("dataset", {})
    peft_cfg = config_dict.get("peft_config", {})

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="bold cyan", min_width=18)
    table.add_column("Value", style="green")

    table.add_row("Model", config_dict.get("model_name", "?"))
    table.add_row("Training Type", config_dict.get("training_type", "?").upper())
    table.add_row("Dataset", ds_cfg.get("path", "?"))
    if ds_cfg.get("limit"):
        table.add_row("Dataset Limit", f"{ds_cfg['limit']:,}")
    table.add_row("GPU", modal_cfg.get("gpu", "H100"))

    if train_cfg.get("learning_rate"):
        table.add_row("Learning Rate", f"{float(train_cfg['learning_rate']):.2e}")
    if train_cfg.get("per_device_train_batch_size"):
        table.add_row("Batch Size", str(train_cfg["per_device_train_batch_size"]))
    if train_cfg.get("num_train_epochs"):
        table.add_row("Epochs", str(train_cfg["num_train_epochs"]))
    if peft_cfg.get("use_peft"):
        table.add_row("PEFT", f"Enabled ({peft_cfg.get('extends', 'custom')})")

    panel = Panel(
        table,
        title="[bold blue]Modal Training Configuration[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )
    Console().print(panel)


# === Submit job ===


def _submit(config_dict: dict, modal_cfg: dict) -> None:
    from contextlib import nullcontext

    import modal

    detach = modal_cfg.get("detach", False)

    # Strip modal key so the container doesn't re-dispatch
    config_dict = {k: v for k, v in config_dict.items() if k != "modal"}

    # Point output_dir at the mounted volume
    output_dir = modal_cfg.get("output_dir", "/outputs")
    config_dict.setdefault("training_config", {})["output_dir"] = output_dir

    # Print trackio dashboard URL before training starts
    space_id = config_dict.get("training_config", {}).get("trackio_space_id")
    if space_id:
        print(f"\nTrackio dashboard: https://huggingface.co/spaces/{space_id}\n")

    config_str = yaml.dump(config_dict)

    app = modal.App(modal_cfg.get("app_name", "liquid-finetune"))
    image = _build_image(modal_cfg)
    volume = modal.Volume.from_name(
        modal_cfg.get("output_volume", "liquid-finetune"),
        create_if_missing=True,
    )
    secrets = [modal.Secret.from_name(s) for s in (modal_cfg.get("secrets") or [])]

    @app.function(
        image=image,
        gpu=modal_cfg.get("gpu", "H100"),
        timeout=modal_cfg.get("timeout", 86400),
        volumes={output_dir: volume},
        secrets=secrets,
        serialized=True,
    )
    def train(cfg: str) -> None:
        import os
        import sys
        import tempfile

        for prefix in ("liquid_finetune", "huggingface_hub", "datasets", "fsspec"):
            for mod_name in [m for m in sys.modules if m.startswith(prefix)]:
                del sys.modules[mod_name]

        os.environ["LIQUID_FINETUNE_DIR"] = "/app"
        os.environ["OUTPUT_DIR"] = output_dir
        os.environ.pop("HF_HUB_OFFLINE", None)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg)
            tmp_path = f.name

        sys.argv = ["liquid-finetune", tmp_path]
        from liquid_finetune import main as liquid_main

        liquid_main()

    if detach:
        print("Submitting detached Modal job (image is cached after first build)...")
    else:
        print("Waiting for container to start (image is cached after first build)...")

    output_context = nullcontext() if detach else modal.enable_output()
    with output_context:
        with app.run(detach=detach):
            if detach:
                call = train.spawn(config_str)
                print(
                    f"Modal job submitted (detached). Function call ID: {call.object_id}"
                )
                if app.app_id:
                    print(f"Modal app ID: {app.app_id}")
                    print(f"Dashboard: https://modal.com/id/{app.app_id}")
                    print(f"Monitor with: modal app logs {app.app_id}")
                    print(f"Stop with: modal app stop {app.app_id}")
            else:
                train.remote(config_str)


# === Image build ===


def _build_image(modal_cfg: dict):
    import modal

    base_image = modal_cfg.get("base_image", "nvidia/cuda:12.8.0-devel-ubuntu22.04")
    pyproject = _REPO_ROOT / "pyproject.toml"
    lockfile = _REPO_ROOT / "uv.lock"

    image = (
        modal.Image.from_registry(base_image, add_python="3.12")
        .pip_install("uv")
        .add_local_file(str(pyproject), remote_path="/app/pyproject.toml", copy=True)
        .add_local_file(str(lockfile), remote_path="/app/uv.lock", copy=True)
        .run_commands(
            # Export exact versions from lockfile, strip flash-attn (OOMs compiling
            # from source on Modal builders), then install everything else.
            "cd /app && uv export --frozen --no-dev --no-emit-project --no-hashes"
            " > requirements.txt"
            " && grep -v flash-attn requirements.txt > requirements-modal.txt"
            " && uv pip install --system -r requirements-modal.txt",
        )
    )

    src_dir = _REPO_ROOT / "src" / "liquid_finetune"
    image = image.add_local_dir(
        str(src_dir), remote_path="/app/src/liquid_finetune", copy=True
    )
    image = image.env(
        {
            "PYTHONPATH": "/app/src",
            "LIQUID_FINETUNE_DIR": "/app",
            "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": "1",
        }
    )
    return image
