import types

from leap_finetune.checkpointing import model_loading


def test_vlm_attention_uses_fa2_only_for_text_backbone(monkeypatch):
    monkeypatch.setattr(
        model_loading,
        "_get_attn_implementation",
        lambda: "flash_attention_2",
    )

    assert model_loading._get_vlm_attn_implementation() == {
        "": "sdpa",
        "text_config": "flash_attention_2",
        "vision_config": "sdpa",
    }



def test_vlm_attention_falls_back_to_sdpa(monkeypatch):
    monkeypatch.setattr(model_loading, "_get_attn_implementation", lambda: "sdpa")

    assert model_loading._get_vlm_attn_implementation() == {
        "": "sdpa",
        "text_config": "sdpa",
        "vision_config": "sdpa",
    }


def test_load_vlm_model_passes_mixed_attention_mapping(monkeypatch):
    calls = {}

    class FakeModel:
        config = types.SimpleNamespace(use_cache=True)

    class FakeProcessor:
        tokenizer = types.SimpleNamespace(padding_side=None, pad_token=None, eos_token="<eos>")

    def fake_model_loader(*args, **kwargs):
        calls["model"] = kwargs
        return FakeModel()

    def fake_processor_loader(*args, **kwargs):
        calls["processor"] = kwargs
        return FakeProcessor()

    monkeypatch.setattr(
        model_loading,
        "AutoModelForImageTextToText",
        types.SimpleNamespace(from_pretrained=fake_model_loader),
    )
    monkeypatch.setattr(
        model_loading,
        "AutoProcessor",
        types.SimpleNamespace(from_pretrained=fake_processor_loader),
    )
    monkeypatch.setattr(model_loading, "_resolve_model_id", lambda _: "test-model")
    monkeypatch.setattr(
        model_loading,
        "_get_vlm_attn_implementation",
        lambda: {"": "sdpa", "text_config": "flash_attention_2", "vision_config": "sdpa"},
    )

    model, processor = model_loading.load_vlm_model(
        "LFM2-VL-1.6B", max_image_tokens=256, do_image_splitting=False
    )

    assert calls["model"]["attn_implementation"] == {
        "": "sdpa",
        "text_config": "flash_attention_2",
        "vision_config": "sdpa",
    }
    assert calls["processor"]["max_image_tokens"] == 256
    assert calls["processor"]["do_image_splitting"] is False
    assert model.config.use_cache is False
    assert processor.tokenizer.padding_side == "right"
    assert processor.tokenizer.pad_token == "<eos>"
