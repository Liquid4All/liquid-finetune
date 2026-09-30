import pathlib

import pytest

from leap_finetune.quantization import gguf_export


def _make_executable(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def _capture_conversions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[list[str]]:
    commands: list[list[str]] = []

    def run(cmd: list[str], _description: str) -> None:
        commands.append(cmd)
        output = pathlib.Path(cmd[cmd.index("--outfile") + 1])
        output.write_bytes(b"gguf")

    monkeypatch.setattr(gguf_export, "_run_subprocess", run)
    return commands


def test_export_uses_llama_cpp_venv(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama_dir = tmp_path / "llama.cpp"
    converter_python = _make_executable(llama_dir / ".venv" / "bin" / "python")
    (llama_dir / "convert_hf_to_gguf.py").write_text("")

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"vision_config": {}}')
    output_dir = tmp_path / "output"
    commands = _capture_conversions(monkeypatch)

    results = gguf_export.export_gguf(
        model_path=model,
        quant_types=["F16"],
        output_dir=output_dir,
        llama_cpp_dir=str(llama_dir),
    )

    assert [command[0] for command in commands] == [str(converter_python)] * 2
    assert "--mmproj" not in commands[0]
    assert "--mmproj" in commands[1]
    assert results == [
        output_dir / "model-F16.gguf",
        output_dir / "mmproj-model-F16.gguf",
    ]


def test_adapter_export_uses_python_override_and_base(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir()
    (llama_dir / "convert_lora_to_gguf.py").write_text("")
    converter_python = _make_executable(tmp_path / "converter-python")

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    commands = _capture_conversions(monkeypatch)

    gguf_export.export_gguf(
        model_path=adapter,
        quant_types=["F16"],
        output_dir=tmp_path / "output",
        base_model_path="/models/base",
        llama_cpp_dir=str(llama_dir),
        llama_cpp_python=str(converter_python),
    )

    assert commands[0][0] == str(converter_python)
    assert commands[0][-2:] == ["--base", "/models/base"]


def test_resolver_preserves_venv_symlink(tmp_path: pathlib.Path) -> None:
    real_python = _make_executable(tmp_path / "python-real")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)

    resolved = gguf_export.resolve_converter_python(tmp_path, None)

    assert resolved == venv_python.absolute()
    assert resolved != real_python


def test_missing_converter_python_has_setup_guidance(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLAMA_CPP_PYTHON", raising=False)

    with pytest.raises(FileNotFoundError) as exc_info:
        gguf_export.resolve_converter_python(tmp_path, None)

    message = str(exc_info.value)
    assert "uv venv --seed .venv" in message
    assert "requirements.txt" in message
