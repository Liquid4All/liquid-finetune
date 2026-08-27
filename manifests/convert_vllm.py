from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


IGNORED_LAYERS = [
    "lm_head",
    "re:.*(?:router|mlp.gate)$",
    "re:.*(?:vision_tower|vision_model|visual).*",
]


def _convert_with_llmcompressor(checkpoint: str, output_dir: str, profile: str) -> None:
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as exc:
        raise SystemExit(
            "Install llmcompressor in a separate conversion environment whose "
            "Transformers version satisfies the package constraints. The "
            "leap-finetune Transformers 5.3 runtime is intentionally separate."
        ) from exc

    scheme = {
        "vllm_fp8": "FP8_DYNAMIC",
        "vllm_mxfp4": "MXFP4",
        "vllm_mxfp8": "MXFP8",
        "vllm_nvfp4": "NVFP4",
    }[profile]
    modifier = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=IGNORED_LAYERS,
    )
    oneshot(model=checkpoint, recipe=modifier, output_dir=output_dir)


def _resolve_quark_script(value: str | None) -> pathlib.Path:
    raw = value or os.environ.get("QUARK_LLM_PTQ_SCRIPT")
    if not raw:
        raise SystemExit(
            "AMD Quark conversion requires --quark-script or "
            "QUARK_LLM_PTQ_SCRIPT pointing to "
            "examples/torch/language_modeling/llm_ptq/quantize_quark.py"
        )
    path = pathlib.Path(raw).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"AMD Quark conversion script not found: {path}")
    return path


def _convert_with_quark(
    checkpoint: str,
    output_dir: str,
    profile: str,
    quark_script: str | None,
    num_calib_data: int,
) -> None:
    script = _resolve_quark_script(quark_script)
    schemes = {
        "vllm_fp8": "ptpc_fp8",
        "vllm_mxfp4": "mxfp4",
        "vllm_mxfp8": "mxfp8",
    }
    if profile not in schemes:
        raise SystemExit(
            f"AMD Quark conversion is not configured for {profile}; "
            "use --tool llmcompressor on NVIDIA Blackwell"
        )
    scheme = schemes[profile]
    command = [
        sys.executable,
        str(script),
        "--model_dir",
        checkpoint,
        "--output_dir",
        output_dir,
        "--quant_scheme",
        scheme,
        "--num_calib_data",
        str(num_calib_data),
        "--model_export",
        "hf_format",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an HF checkpoint to a vLLM deployment format"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--profile",
        choices=(
            "vllm_fp8",
            "vllm_mxfp4",
            "vllm_mxfp8",
            "vllm_nvfp4",
        ),
        default="vllm_mxfp4",
    )
    parser.add_argument(
        "--tool",
        choices=("llmcompressor", "quark"),
        default=None,
        help="Defaults to Quark for MXFP4 and llm-compressor for FP8, MXFP8, "
        "and NVFP4.",
    )
    parser.add_argument("--quark-script")
    parser.add_argument("--num-calib-data", type=int, default=32)
    args = parser.parse_args()

    tool = args.tool or ("quark" if args.profile == "vllm_mxfp4" else "llmcompressor")
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if tool == "quark":
        _convert_with_quark(
            args.checkpoint,
            args.output_dir,
            args.profile,
            args.quark_script,
            args.num_calib_data,
        )
    else:
        _convert_with_llmcompressor(args.checkpoint, args.output_dir, args.profile)


if __name__ == "__main__":
    main()
