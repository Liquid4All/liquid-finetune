import argparse
import importlib.metadata
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass

from liquid_finetune.distribution.ray_runtime import normalize_visible_devices

FLASH_ATTN_VERSION = "2.8.3"


@dataclass(frozen=True)
class RuntimeTarget:
    backend: str
    python_tag: str
    sys_platform: str
    machine: str
    torch_version: str | None
    cuda_version: str | None
    hip_version: str | None
    accelerator_available: bool | None = None


@dataclass(frozen=True)
class FlashAttnWheel:
    backend: str
    python_tag: str
    sys_platform: str
    machine: str
    torch_prefix: str
    cuda_prefix: str | None
    hip_prefix: str | None
    url: str


PINNED_FLASH_ATTN_WHEELS = (
    FlashAttnWheel(
        backend="cuda",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_prefix="2.11.",
        cuda_prefix="13.",
        hip_prefix=None,
        url=(
            "https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/"
            "flash_attn-2.8.3%2Bcu13torch2.11cxx11abiTRUE-cp312-cp312-"
            "linux_x86_64.whl"
            "#sha256=eea423825f3e12818b98b2078e2cb5ce6fe6b73d22612316d2a55fad4701938f"
        ),
    ),
    FlashAttnWheel(
        backend="rocm",
        python_tag="cp312",
        sys_platform="linux",
        machine="x86_64",
        torch_prefix="2.10.0+git8514f05",
        cuda_prefix=None,
        hip_prefix="7.2",
        url=(
            "https://wheels.vllm.ai/rocm/799c3afa5d5b17b676d04e0b58a5628943bb4003/"
            "flash_attn-2.8.3-cp312-cp312-manylinux_2_34_x86_64.whl"
            "#sha256=72bf51493106a01ac85d96493bdef3637f099c607fe2a1326f86d7b8436c89cf"
        ),
    ),
)


def detect_runtime_target() -> RuntimeTarget:
    normalize_visible_devices()

    torch_version = None
    cuda_version = None
    hip_version = None
    accelerator_available = None
    backend = "unknown"

    try:
        import torch
    except ImportError:
        pass
    else:
        torch_version = torch.__version__
        cuda_version = getattr(torch.version, "cuda", None)
        hip_version = getattr(torch.version, "hip", None)
        accelerator_available = torch.cuda.is_available()
        if hip_version:
            backend = "rocm"
        elif cuda_version:
            backend = "cuda"
        elif accelerator_available:
            backend = "cuda"

    return RuntimeTarget(
        backend=backend,
        python_tag=f"cp{sys.version_info.major}{sys.version_info.minor}",
        sys_platform=sys.platform,
        machine=platform.machine().lower(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        hip_version=hip_version,
        accelerator_available=accelerator_available,
    )


def _package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _flash_attn_2_status() -> tuple[bool, str]:
    normalize_visible_devices()
    from liquid_finetune.checkpointing.model_loading import (
        _flash_attn_2_status as get_status,
    )

    return get_status()


def _matching_pinned_wheel(target: RuntimeTarget) -> FlashAttnWheel | None:
    for wheel in PINNED_FLASH_ATTN_WHEELS:
        if wheel.backend != target.backend:
            continue
        if wheel.python_tag != target.python_tag:
            continue
        if wheel.sys_platform != target.sys_platform:
            continue
        if wheel.machine != target.machine:
            continue
        if not target.torch_version or not target.torch_version.startswith(
            wheel.torch_prefix
        ):
            continue
        if wheel.cuda_prefix and not (
            target.cuda_version and target.cuda_version.startswith(wheel.cuda_prefix)
        ):
            continue
        if wheel.hip_prefix and not (
            target.hip_version and target.hip_version.startswith(wheel.hip_prefix)
        ):
            continue
        return wheel
    return None


def _print_target(target: RuntimeTarget) -> None:
    print(f"backend: {target.backend}")
    print(f"python: {sys.version.split()[0]} ({target.python_tag})")
    print(f"platform: {target.sys_platform}/{target.machine}")
    print(f"torch: {target.torch_version or 'not installed'}")
    print(f"cuda: {target.cuda_version or 'none'}")
    print(f"hip: {target.hip_version or 'none'}")
    available = (
        str(target.accelerator_available).lower()
        if target.accelerator_available is not None
        else "unknown"
    )
    print(f"accelerator_available: {available}")


def fa2_status(*, require: bool = False) -> int:
    target = detect_runtime_target()
    usable, reason = _flash_attn_2_status()
    selected = "flash_attention_2" if usable else "sdpa"

    _print_target(target)
    print(f"flash-attn: {_package_version('flash-attn') or 'not installed'}")
    print(f"fa2_status: {'usable' if usable else 'unavailable'}")
    print(f"attn_implementation: {selected}")
    print(f"reason: {reason}")

    if require and not usable:
        return 1
    return 0


def _run_uv_pip_install(requirement: str, *extra_args: str) -> bool:
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall-package",
        "flash-attn",
        *extra_args,
        requirement,
    ]
    print("+ " + shlex.join(command))
    result = subprocess.run(command, check=False)
    return result.returncode == 0


def install_fa2(*, allow_source_build: bool = False, require: bool = False) -> int:
    target = detect_runtime_target()
    _print_target(target)

    wheel = _matching_pinned_wheel(target)
    if wheel:
        print("Trying pinned FlashAttention 2 wheel")
        if _run_uv_pip_install(wheel.url) and fa2_status(require=require) == 0:
            return 0
    else:
        print("No pinned FlashAttention 2 wheel matches this runtime target")

    print("Trying binary-only FlashAttention 2 resolution")
    binary_ok = _run_uv_pip_install(
        f"flash-attn=={FLASH_ATTN_VERSION}",
        "--only-binary",
        "flash-attn",
    )
    if binary_ok and fa2_status(require=require) == 0:
        return 0

    if allow_source_build:
        print("Trying FlashAttention 2 source build")
        if (
            _run_uv_pip_install(
                f"flash-attn=={FLASH_ATTN_VERSION}",
                "--no-binary",
                "flash-attn",
                "--no-build-isolation-package",
                "flash-attn",
            )
            and fa2_status(require=require) == 0
        ):
            return 0

    usable, reason = _flash_attn_2_status()
    if usable:
        return 0

    print(f"FlashAttention 2 unavailable; runtime will fall back to SDPA: {reason}")
    if require:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Environment diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("fa2-status", help="Check FlashAttention 2 status")
    status.add_argument(
        "--require",
        action="store_true",
        help="Exit nonzero when FlashAttention 2 is unavailable",
    )

    install = subparsers.add_parser(
        "install-fa2",
        help="Install or repair FlashAttention 2 for the current runtime",
    )
    install.add_argument(
        "--allow-source-build",
        action="store_true",
        help="Allow source build if no pinned or binary wheel works",
    )
    install.add_argument(
        "--require",
        action="store_true",
        help="Exit nonzero when FlashAttention 2 is still unavailable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fa2-status":
        return fa2_status(require=args.require)
    if args.command == "install-fa2":
        return install_fa2(
            allow_source_build=args.allow_source_build,
            require=args.require,
        )
    parser.error(f"Unknown env command: {args.command}")
    return 2
