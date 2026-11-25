"""Model-config overrides reach the VLM config before the model is built."""

from types import SimpleNamespace

from liquid_finetune.checkpointing import model_loading


def test_vlm_loading_applies_model_config_overrides(monkeypatch):
    calls = {}
    config = SimpleNamespace(model_type="lfm3_vl")
    fake_model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    fake_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="left", pad_token="<pad>")
    )
    monkeypatch.setattr(
        model_loading.AutoConfig, "from_pretrained", lambda *a, **k: config
    )

    def load_model(*args, **kwargs):
        calls["model"] = kwargs
        return fake_model

    monkeypatch.setattr(
        model_loading.AutoModelForImageTextToText, "from_pretrained", load_model
    )
    monkeypatch.setattr(
        model_loading.AutoProcessor, "from_pretrained", lambda *a, **k: fake_processor
    )
    model_loading.load_vlm_model("org/vlm", model_config={"mtp_soft": True})
    assert calls["model"]["config"] is config and config.mtp_soft is True
