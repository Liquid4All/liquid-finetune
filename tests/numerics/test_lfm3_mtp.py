"""MTP start points for LFM3 (``liquid-prepare-mtp``) on a tiny random LFM3."""

import json
import shutil

import pytest
import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from liquid_finetune.models.lfm3_mtp import prepare_lfm3_mtp_checkpoint
from liquid_finetune.training.utils.causal_lm_loss import (
    install_memory_efficient_causal_lm_loss,
)

LFM3_REPO = "LiquidAI/LFM3-1.7B"
LFM3_FILES = (
    "config.json",
    "configuration_lfm3.py",
    "modeling_lfm3.py",
    "lfm3_conv.py",
)
LAYER_TYPES = ["conv", "full_attention", "conv", "sliding_attention"]


@pytest.fixture(scope="module")
def tiny_lfm3(tmp_path_factory):
    """A 4-layer LFM3 built from the published remote code (no weights downloaded)."""
    hub = pytest.importorskip("huggingface_hub")
    root = tmp_path_factory.mktemp("tiny_lfm3")
    try:
        for name in LFM3_FILES:
            shutil.copy(hub.hf_hub_download(LFM3_REPO, name), root / name)
    except Exception as exc:  # offline or Hub unavailable
        pytest.skip(f"LFM3 remote code unavailable: {exc}")
    config = json.loads((root / "config.json").read_text())
    config.update(
        num_hidden_layers=len(LAYER_TYPES),
        layer_types=LAYER_TYPES,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    (root / "config.json").write_text(json.dumps(config))
    # Build through the dynamic classes, not AutoModel: Auto registration would
    # pin Lfm3Config to Lfm3ForCausalLM for the rest of the process.
    config_cls = get_class_from_dynamic_module(
        "configuration_lfm3.Lfm3Config", str(root)
    )
    model_cls = get_class_from_dynamic_module(
        "modeling_lfm3.Lfm3ForCausalLM", str(root)
    )
    torch.manual_seed(0)
    model = model_cls(config_cls.from_pretrained(str(root)))
    state = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(root / "model.safetensors"), metadata={"format": "pt"})
    return root


def _load_mtp(model_dir):
    cls = get_class_from_dynamic_module(
        "modeling_lfm3_mtp.Lfm3ForCausalLMWithMTP", str(model_dir)
    )
    # What AutoModelForCausalLM does for remote code; save_pretrained relies on it
    # to copy the modeling files next to the weights.
    cls.register_for_auto_class("AutoModelForCausalLM")
    return cls.from_pretrained(str(model_dir), dtype=torch.float32)


def test_heads_mode_writes_config_and_mtp_tensors(tiny_lfm3, tmp_path):
    summary = prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "mtp"))
    config = json.loads((tmp_path / "mtp" / "config.json").read_text())
    assert config["auto_map"]["AutoModelForCausalLM"] == (
        "modeling_lfm3_mtp.Lfm3ForCausalLMWithMTP"
    )
    assert config["mtp_train_only"] is True
    assert config["mtp_init_from_layer"] == 1  # last full-attention layer
    assert summary["added_tensors"] > 0
    assert (tmp_path / "mtp" / "modeling_lfm3_mtp.py").is_file()


def test_frozen_backbone_trains_only_the_mtp_module(tiny_lfm3, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "mtp"))
    model = _load_mtp(tmp_path / "mtp")
    trainable = {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    }
    assert trainable == {"mtp"}

    forward = model.forward
    install_memory_efficient_causal_lm_loss(model)
    assert model.forward == forward  # the MTP model keeps its own loss

    input_ids = torch.randint(0, 1000, (2, 16))
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        (p.grad is not None) == n.startswith("mtp.")
        for n, p in model.named_parameters()
    )


def test_add_module_appends_a_second_module(tiny_lfm3, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "heads"))
    summary = prepare_lfm3_mtp_checkpoint(
        str(tmp_path / "heads"), str(tmp_path / "two"), mode="add-module"
    )
    config = json.loads((tmp_path / "two" / "config.json").read_text())
    assert summary["mtp_modules"] == 2
    assert config["mtp_num_modules"] == 2 and config["mtp_train_modules"] == [2]
    model = _load_mtp(tmp_path / "two")
    trainable = {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    }
    assert trainable == {"mtp2"}


def test_saved_mtp_checkpoint_reloads_with_its_heads(tiny_lfm3, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "mtp"))
    model = _load_mtp(tmp_path / "mtp")
    model.save_pretrained(str(tmp_path / "saved"))
    config = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert config["auto_map"]["AutoModelForCausalLM"].startswith("modeling_lfm3_mtp.")
    assert {"modeling_lfm3_mtp.py", "modeling_lfm3.py", "lfm3_conv.py"} <= {
        p.name for p in (tmp_path / "saved").iterdir()
    }
    reloaded = _load_mtp(tmp_path / "saved")
    assert sum(k.startswith("mtp.") for k in reloaded.state_dict()) == sum(
        k.startswith("mtp.") for k in model.state_dict()
    )
    with torch.no_grad():  # inference (no labels) is the unchanged LFM3 forward
        logits = reloaded(input_ids=torch.randint(0, 1000, (1, 8))).logits
    assert logits.shape[:2] == (1, 8)


def test_rejects_non_lfm3_checkpoints(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.json").write_text(json.dumps({"model_type": "lfm2"}))
    with pytest.raises(ValueError, match="LFM3"):
        prepare_lfm3_mtp_checkpoint(str(tmp_path / "src"), str(tmp_path / "out"))
