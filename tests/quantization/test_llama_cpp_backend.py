from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.evaluation


def test_llama_cpp_eval_backend_config_parses():
    from leap_finetune.config import EvalRunConfig

    cfg = EvalRunConfig.model_validate(
        {
            "checkpoint": "/models/test.gguf",
            "evals": {"benchmarks": []},
            "backend": {
                "type": "llama_cpp",
                "server_binary": "/opt/llama-server",
                "mmproj": "/models/mmproj.gguf",
                "server_args": ["--ctx-size", "4096"],
            },
        }
    )

    assert cfg.backend.type == "llama_cpp"
    assert cfg.backend.n_gpu_layers == 999
    assert cfg.backend.mmproj == "/models/mmproj.gguf"


def test_build_llama_server_command_is_shell_free():
    from leap_finetune.evaluation.backend import build_llama_server_command

    command = build_llama_server_command(
        server_binary="/opt/llama-server",
        model_path="/models/model with spaces.gguf",
        host="127.0.0.1",
        port=8123,
        n_gpu_layers=999,
        model_id="qat-model",
        mmproj="/models/mmproj.gguf",
        server_args=["--ctx-size", "4096"],
    )

    assert command[0] == "/opt/llama-server"
    assert "/models/model with spaces.gguf" in command
    assert command[-2:] == ["--ctx-size", "4096"]
    assert "--mmproj" in command


def test_external_llama_cpp_backend_does_not_spawn(monkeypatch):
    import requests

    session = MagicMock()
    monkeypatch.setattr(requests, "Session", lambda: session)
    popen = MagicMock()
    monkeypatch.setattr("subprocess.Popen", popen)

    from leap_finetune.evaluation.backend import LlamaCppServerBackend

    backend = LlamaCppServerBackend(
        "/models/not-local.gguf",
        base_url="http://compute-node:8080",
        model_id="qat-model",
    )
    try:
        assert backend.base_url == "http://compute-node:8080"
        assert backend.model_id == "qat-model"
        popen.assert_not_called()
    finally:
        backend.close()
    session.close.assert_called_once()


def test_runner_constructs_llama_cpp_backend_without_model_resolution(monkeypatch):
    from leap_finetune.evaluation import runner

    calls = {}

    class FakeBackend:
        def __init__(self, model_path, **kwargs):
            calls["model_path"] = model_path
            calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "leap_finetune.evaluation.backend.LlamaCppServerBackend", FakeBackend
    )
    runner.create_llama_cpp_backend(
        "/models/model.gguf",
        {"base_url": "http://node:8080", "n_gpu_layers": 42},
    )

    assert calls["model_path"] == "/models/model.gguf"
    assert calls["kwargs"]["base_url"] == "http://node:8080"
    assert calls["kwargs"]["n_gpu_layers"] == 42


def test_vllm_eval_preserves_quantization_config(monkeypatch):
    from transformers import AutoTokenizer

    from leap_finetune.evaluation import runner

    calls = {}

    def fake_from_pretrained(model_ref, **kwargs):
        calls["processor_model_ref"] = model_ref
        return MagicMock()

    class FakeVLLMBackend:
        def __init__(self, model_path, **kwargs):
            calls["model_path"] = model_path
            calls["kwargs"] = kwargs

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        "leap_finetune.evaluation.backend.VLLMInProcessBackend",
        FakeVLLMBackend,
    )

    runner.load_eval_processor("LFM2-1.2B", modality="text")
    runner.create_vllm_backend(
        "LFM2-1.2B", {"tensor_parallel_size": 1, "quantization": "fp8"}
    )

    assert calls["processor_model_ref"] == "LiquidAI/LFM2-1.2B"
    assert calls["model_path"] == "LiquidAI/LFM2-1.2B"
    assert calls["kwargs"]["quantization"] == "fp8"


def test_llm_benchmark_factory_preserves_configured_generation_metric():
    from leap_finetune.evaluation.llm_config import create_llm_benchmarks_from_config

    benchmarks = create_llm_benchmarks_from_config(
        {
            "benchmarks": [
                {
                    "name": "rouge",
                    "path": "UNUSED",
                    "metric": "rouge_l",
                }
            ]
        },
        tokenizer=None,
    )

    assert benchmarks[0].metric == "rouge_l"
