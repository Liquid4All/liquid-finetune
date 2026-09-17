import logging
import os
import pathlib
import subprocess
import sys

logger = logging.getLogger(__name__)

# === Quantization type sets ===

# Types that convert_hf_to_gguf.py can produce directly via --outtype
DIRECT_QUANTS = {"F16", "BF16", "F32", "Q8_0"}

# Types that require two-step: convert to F16 first, then llama-quantize
# Includes both legacy quants (Q4_0, Q5_0) and K-quants (Q4_K_M, Q5_K_S, etc.)
QUANTIZE_QUANTS = {
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_0",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_0",
    "Q5_K_S",
    "Q5_K_M",
    "Q5_K_L",
    "Q6_K",
}

ALL_QUANTS = DIRECT_QUANTS | QUANTIZE_QUANTS

# Adapters only support these types via convert_lora_to_gguf.py
ADAPTER_QUANTS = {"F16", "BF16", "F32", "Q8_0"}

# Map CLI quant names to convert script --outtype values
OUTTYPE_MAP = {"F16": "f16", "BF16": "bf16", "F32": "f32", "Q8_0": "q8_0"}


def resolve_llama_cpp_dir(cli_arg: str | None) -> pathlib.Path:
    raw = cli_arg or os.environ.get("LLAMA_CPP_DIR")
    if not raw:
        raise FileNotFoundError(
            "A llama.cpp checkout is required for GGUF export.\n\n"
            "Set LLAMA_CPP_DIR or use --llama-cpp-dir:\n\n"
            "  git clone https://github.com/ggml-org/llama.cpp\n"
            "  export LLAMA_CPP_DIR=/path/to/llama.cpp\n"
        )

    llama_dir = pathlib.Path(raw).expanduser().resolve()
    if not llama_dir.is_dir():
        raise FileNotFoundError(f"llama.cpp directory does not exist: {llama_dir}")
    return llama_dir


def resolve_convert_script(llama_dir: pathlib.Path, name: str) -> pathlib.Path:
    script = llama_dir / name
    if not script.exists():
        raise FileNotFoundError(
            f"{name} not found in {llama_dir}. Update your llama.cpp checkout."
        )
    return script


def resolve_quantize_binary(llama_dir: pathlib.Path) -> pathlib.Path:
    candidates = [
        llama_dir / "build" / "bin" / "llama-quantize",
        llama_dir / "llama-quantize",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"llama-quantize binary not found in {llama_dir}\n\n"
        "llama.cpp appears cloned but not built. Run:\n"
        f"  cd {llama_dir} && cmake -B build && cmake --build build --config Release\n"
    )


def is_adapter_path(model_path: pathlib.Path) -> bool:
    return (model_path / "adapter_config.json").exists()


def validate_model_path(model_path: pathlib.Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Path not found: {model_path}")
    if not model_path.is_dir():
        raise ValueError(f"Expected a directory, got a file: {model_path}")

    has_model = (model_path / "config.json").exists()
    has_adapter = (model_path / "adapter_config.json").exists()
    if not has_model and not has_adapter:
        raise ValueError(
            f"No config.json or adapter_config.json found in {model_path}\n"
            "Expected a HuggingFace model or PEFT adapter directory."
        )


def _run_subprocess(cmd: list[str], description: str) -> None:
    logger.info("%s: %s", description, " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed (exit code {result.returncode})")


def convert_hf_to_gguf(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    convert_script: pathlib.Path,
    outtype: str = "f16",
) -> pathlib.Path:
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile",
        str(output_path),
        "--outtype",
        outtype,
    ]
    _run_subprocess(cmd, f"Converting to GGUF ({outtype})")
    logger.info("Created %s (%.2f GB)", output_path, output_path.stat().st_size / 1e9)
    return output_path


def convert_lora_to_gguf(
    adapter_path: pathlib.Path,
    output_path: pathlib.Path,
    convert_script: pathlib.Path,
    outtype: str = "f16",
    base_model_path: str | None = None,
) -> pathlib.Path:
    cmd = [
        sys.executable,
        str(convert_script),
        str(adapter_path),
        "--outfile",
        str(output_path),
        "--outtype",
        outtype,
    ]
    if base_model_path:
        cmd.extend(["--base", str(base_model_path)])

    _run_subprocess(cmd, f"Converting LoRA adapter to GGUF ({outtype})")
    logger.info("Created %s (%.2f GB)", output_path, output_path.stat().st_size / 1e9)
    return output_path


def quantize_gguf(
    input_gguf: pathlib.Path,
    output_gguf: pathlib.Path,
    quant_type: str,
    quantize_bin: pathlib.Path,
) -> pathlib.Path:
    cmd = [str(quantize_bin), str(input_gguf), str(output_gguf), quant_type]
    _run_subprocess(cmd, f"Quantizing to {quant_type}")
    logger.info("Created %s (%.2f GB)", output_gguf, output_gguf.stat().st_size / 1e9)
    return output_gguf


def export_gguf(
    model_path: pathlib.Path,
    quant_types: list[str],
    output_dir: pathlib.Path,
    base_model_path: str | None = None,
    llama_cpp_dir: str | None = None,
) -> list[pathlib.Path]:
    model_name = model_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = is_adapter_path(model_path)
    results = []

    llama_dir = resolve_llama_cpp_dir(llama_cpp_dir)

    if adapter:
        unsupported = set(quant_types) - ADAPTER_QUANTS
        if unsupported:
            raise ValueError(
                f"Adapter exports only support {sorted(ADAPTER_QUANTS)}, "
                f"got unsupported type(s): {sorted(unsupported)}\n"
                "For K-quants, merge the adapter into the base model first, "
                "then export the merged model."
            )

        convert_lora = resolve_convert_script(llama_dir, "convert_lora_to_gguf.py")
        for quant in quant_types:
            outtype = OUTTYPE_MAP[quant]
            out_path = output_dir / f"{model_name}-lora-{quant}.gguf"
            convert_lora_to_gguf(
                model_path, out_path, convert_lora, outtype, base_model_path
            )
            results.append(out_path)

        return results

    convert_hf = resolve_convert_script(llama_dir, "convert_hf_to_gguf.py")

    # === Full model export ===
    direct = [q for q in quant_types if q in DIRECT_QUANTS]
    needs_quantize = [q for q in quant_types if q in QUANTIZE_QUANTS]

    # Direct quants (F16, BF16, F32, Q8_0) — single step via the convert script
    for quant in direct:
        outtype = OUTTYPE_MAP[quant]
        out_path = output_dir / f"{model_name}-{quant}.gguf"
        convert_hf_to_gguf(model_path, out_path, convert_hf, outtype)
        results.append(out_path)

    # Quantize quants — need F16 intermediate, then llama-quantize binary
    if needs_quantize:
        quantize_bin = resolve_quantize_binary(llama_dir)

        f16_requested = "F16" in direct
        f16_path = output_dir / f"{model_name}-F16.gguf"

        if not f16_path.exists():
            convert_hf_to_gguf(model_path, f16_path, convert_hf, "f16")

        for quant in needs_quantize:
            out_path = output_dir / f"{model_name}-{quant}.gguf"
            quantize_gguf(f16_path, out_path, quant, quantize_bin)
            results.append(out_path)

        # Clean up intermediate F16 if it wasn't explicitly requested
        if not f16_requested and f16_path.exists():
            f16_path.unlink()
            logger.info("Cleaned up intermediate F16 file")

    return results
