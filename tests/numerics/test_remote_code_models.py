"""Regression coverage for Hub models that ship custom code, including LFM3."""

import signal
from types import SimpleNamespace

import pytest
import torch

from liquid_finetune.checkpointing.hf_export import original_model_class_for_save
from liquid_finetune.distribution import vllm_server
from liquid_finetune.training.utils.generation_config import (
    apply_model_generation_defaults,
)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)


def _fsdp2_style_subclass(cls):
    # ``fully_shard`` swaps ``type(module)`` for a generated subclass defined in
    # torch.distributed.fsdp; save_pretrained would record that class in auto_map.
    return type(
        f"FSDP{cls.__name__}",
        (cls,),
        {"__module__": "torch.distributed.fsdp._fully_shard._fully_shard"},
    )


def test_original_model_class_is_restored_for_save():
    model = _TinyModel()
    fsdp_cls = _fsdp2_style_subclass(_TinyModel)
    model.__class__ = fsdp_cls
    with original_model_class_for_save(model):
        assert type(model) is _TinyModel
    assert type(model) is fsdp_cls


def test_original_model_class_is_restored_after_an_error():
    model = _TinyModel()
    fsdp_cls = _fsdp2_style_subclass(_TinyModel)
    model.__class__ = fsdp_cls
    with pytest.raises(RuntimeError):
        with original_model_class_for_save(model):
            raise RuntimeError("save failed")
    assert type(model) is fsdp_cls


def test_unsharded_model_class_is_untouched():
    model = _TinyModel()
    with original_model_class_for_save(model):
        assert type(model) is _TinyModel
    assert type(model) is _TinyModel


def test_vllm_server_sets_visibility_and_session(monkeypatch):
    captured = {}
    process = SimpleNamespace(pid=123)

    def popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["start_new_session"] = kwargs["start_new_session"]
        return process

    monkeypatch.setattr(vllm_server.shutil, "which", lambda _: "/bin/trl")
    monkeypatch.setattr(vllm_server.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm_server, "wait_for_vllm_health", lambda *a, **k: None)
    monkeypatch.setattr(vllm_server.atexit, "register", lambda *a, **k: None)

    vllm_server.launch_vllm_server(
        "LiquidAI/LFM3-1.7B",
        [0],
        {},
    )

    command = captured["command"]
    assert "--trust_remote_code" not in command
    assert "--vllm_model_impl" not in command
    assert "--enforce_eager" not in command
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["start_new_session"] is True
    if "HIP_VISIBLE_DEVICES" in captured["env"]:
        assert captured["env"]["HIP_VISIBLE_DEVICES"] == "0"


def test_vllm_server_terminates_the_whole_process_group(monkeypatch):
    signals = []

    class Process:
        pid = 123

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(
        vllm_server.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    process = Process()
    vllm_server._terminate_server(process)
    vllm_server._terminate_server(process)

    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]


def test_vllm_server_can_explicitly_enable_remote_code(monkeypatch):
    captured = {}

    def popen(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(vllm_server.shutil, "which", lambda _: "/bin/trl")
    monkeypatch.setattr(vllm_server.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm_server, "wait_for_vllm_health", lambda *a, **k: None)
    monkeypatch.setattr(vllm_server.atexit, "register", lambda *a, **k: None)

    vllm_server.launch_vllm_server(
        "org/remote-model",
        [0],
        {"trust_remote_code": True},
    )

    command = captured["command"]
    assert command[command.index("--trust_remote_code") + 1] == "true"


def test_checkpoint_sampling_defaults_preserve_explicit_overrides():
    args = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
    )
    model = SimpleNamespace(
        generation_config=SimpleNamespace(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            min_p=0.05,
            repetition_penalty=1.1,
        )
    )

    apply_model_generation_defaults(args, model, {"top_p"})

    assert args.temperature == 0.1
    assert args.top_p == 1.0
    assert args.top_k == 50
    assert args.min_p == 0.05
    assert args.repetition_penalty == 1.1
