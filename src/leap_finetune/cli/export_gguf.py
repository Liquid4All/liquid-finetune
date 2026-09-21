import argparse
import logging
import pathlib

from leap_finetune.quantization.gguf_export import (
    ALL_QUANTS,
    export_gguf,
    validate_model_path,
)


def _parse_quant_types(values: list[str] | None) -> list[str]:
    quant_types = [value.upper() for value in values or ["F16"]]
    unsupported = sorted(set(quant_types) - ALL_QUANTS)
    if unsupported:
        supported = ", ".join(sorted(ALL_QUANTS))
        raise argparse.ArgumentTypeError(
            f"Unsupported GGUF quant type(s): {', '.join(unsupported)}. "
            f"Supported values: {supported}"
        )
    return quant_types


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a HuggingFace model or PEFT adapter to GGUF."
    )
    parser.add_argument(
        "model_path",
        type=pathlib.Path,
        help="Path to a HuggingFace model directory or PEFT adapter directory.",
    )
    parser.add_argument(
        "--quant",
        action="append",
        dest="quant_types",
        metavar="TYPE",
        help="GGUF quant type to export. Repeat for multiple types. Defaults to F16.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        default=None,
        help="Directory for GGUF output files. Defaults to <model_path>/gguf.",
    )
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Base model path for PEFT adapter GGUF export.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="llama.cpp checkout containing build/bin/llama-quantize for K-quants.",
    )
    parser.add_argument(
        "--token-embedding-type",
        choices=("F16", "F32", "Q8_0"),
        default=None,
        help=(
            "Override token_embd.weight in llama-quantize. QAT profiles that "
            "exclude tied embeddings should use F16."
        ),
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()

    try:
        quant_types = _parse_quant_types(args.quant_types)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    model_path = args.model_path.expanduser().resolve()
    try:
        validate_model_path(model_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else model_path / "gguf"
    )

    paths = export_gguf(
        model_path=model_path,
        quant_types=quant_types,
        output_dir=output_dir,
        base_model_path=args.base_model_path,
        llama_cpp_dir=args.llama_cpp_dir,
        token_embedding_type=args.token_embedding_type,
    )

    print("GGUF export complete:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
