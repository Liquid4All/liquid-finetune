"""Models that ship their own code (``auto_map``), e.g. LFM3."""

from types import SimpleNamespace

import pytest
import torch

from liquid_finetune.checkpointing import model_loading
from liquid_finetune.checkpointing.hf_export import original_model_class_for_save


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


def test_text_model_loading_allows_remote_code(monkeypatch):
    calls = {}

    def record(name, result):
        def from_pretrained(*args, **kwargs):
            calls[name] = kwargs
            return result

        return from_pretrained

    config = SimpleNamespace(model_type="lfm3")
    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="lfm3",
            architectures=["Lfm3ForCausalLM"],
            num_hidden_layers=2,
            hidden_size=8,
            vocab_size=16,
            use_cache=True,
        )
    )
    tokenizer = SimpleNamespace(chat_template=None)
    monkeypatch.setattr(
        model_loading.AutoConfig, "from_pretrained", record("config", config)
    )
    monkeypatch.setattr(
        model_loading.AutoModelForCausalLM,
        "from_pretrained",
        record("model", fake_model),
    )
    monkeypatch.setattr(
        model_loading.AutoTokenizer, "from_pretrained", record("tokenizer", tokenizer)
    )

    model_loading.load_tokenizer("org/remote-code-model")
    assert calls["tokenizer"]["trust_remote_code"] is True

    model_loading.load_model(
        "org/remote-code-model",
        install_memory_efficient_loss=False,
        enable_grouped_mm=False,
    )
    assert calls["config"]["trust_remote_code"] is True
    assert calls["model"]["trust_remote_code"] is True
    assert calls["tokenizer"]["trust_remote_code"] is True
