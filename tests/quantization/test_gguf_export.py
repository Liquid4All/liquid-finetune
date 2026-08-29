from __future__ import annotations

from pathlib import Path

from leap_finetune.quantization import gguf_export


def test_quantize_gguf_can_preserve_token_embeddings(monkeypatch, tmp_path):
    input_path = tmp_path / "input.gguf"
    output_path = tmp_path / "output.gguf"
    input_path.touch()
    commands = []

    def fake_run(command, description):
        commands.append((command, description))
        output_path.write_bytes(b"gguf")

    monkeypatch.setattr(gguf_export, "_run_subprocess", fake_run)
    gguf_export.quantize_gguf(
        input_path,
        output_path,
        "Q4_0",
        Path("/opt/llama-quantize"),
        token_embedding_type="F16",
    )

    assert commands[0][0] == [
        "/opt/llama-quantize",
        "--token-embedding-type",
        "f16",
        str(input_path),
        str(output_path),
        "Q4_0",
    ]


def test_q8_embedding_override_uses_llama_quantize(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    output_dir = tmp_path / "output"
    conversions = []
    quantizations = []

    def fake_convert(model, output, outtype):
        conversions.append((model, output, outtype))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"f16")
        return output

    def fake_quantize(source, output, quant, binary, token_embedding_type=None):
        quantizations.append((source, output, quant, binary, token_embedding_type))
        output.write_bytes(b"q8")
        return output

    monkeypatch.setattr(gguf_export, "convert_hf_to_gguf", fake_convert)
    monkeypatch.setattr(
        gguf_export, "resolve_quantize_binary", lambda _: Path("/opt/llama-quantize")
    )
    monkeypatch.setattr(gguf_export, "quantize_gguf", fake_quantize)

    results = gguf_export.export_gguf(
        model_path,
        ["Q8_0"],
        output_dir,
        llama_cpp_dir="/opt/llama.cpp",
        token_embedding_type="F16",
    )

    assert conversions[0][2] == "f16"
    assert quantizations[0][2:] == (
        "Q8_0",
        Path("/opt/llama-quantize"),
        "F16",
    )
    assert results == [output_dir / "model-Q8_0.gguf"]
