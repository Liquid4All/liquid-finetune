"""MTP start points for LFM3 (``liquid-prepare-mtp``) on a tiny random LFM3."""

import json
import shutil

import pytest
import torch
import torch.nn.functional as F
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


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_mtp_loss_respects_left_padding(tiny_lfm3, tmp_path, depth):
    model_dir = tmp_path / "heads"
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(model_dir))
    for module in range(2, depth + 1):
        next_dir = tmp_path / {2: "two", 3: "three"}[module]
        prepare_lfm3_mtp_checkpoint(str(model_dir), str(next_dir), mode="add-module")
        model_dir = next_dir
    model = _load_mtp(model_dir).eval()

    torch.manual_seed(1)
    input_ids = torch.randint(1, 1000, (1, 8))
    unpadded = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
    ).loss

    padded_ids = torch.cat([torch.zeros((1, 4), dtype=torch.long), input_ids], dim=1)
    attention_mask = torch.cat(
        [torch.zeros((1, 4), dtype=torch.long), torch.ones_like(input_ids)], dim=1
    )
    labels = padded_ids.masked_fill(attention_mask == 0, -100)
    unpadded_positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
    padded_positions = torch.arange(padded_ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        unpadded_hidden = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=unpadded_positions,
        ).last_hidden_state
        padded_hidden = model.model(
            input_ids=padded_ids,
            attention_mask=attention_mask,
            position_ids=padded_positions,
        ).last_hidden_state
        torch.testing.assert_close(
            padded_hidden[:, -input_ids.shape[1] :],
            unpadded_hidden,
            rtol=1e-5,
            atol=1e-5,
        )
        if depth == 1:
            unpadded_draft = model.mtp_hidden(
                unpadded_hidden,
                input_ids,
                unpadded_positions,
                torch.ones_like(input_ids),
            )
            padded_draft = model.mtp_hidden(
                padded_hidden,
                padded_ids,
                padded_positions,
                attention_mask,
            )
            torch.testing.assert_close(
                padded_draft[:, 4:10],
                unpadded_draft[:, :6],
                rtol=1e-5,
                atol=1e-5,
            )
    padded = model(
        input_ids=padded_ids,
        attention_mask=attention_mask,
        labels=labels,
    ).loss

    torch.testing.assert_close(padded, unpadded, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_packed_documents_match_separate_losses(tiny_lfm3, tmp_path, depth):
    model_dir = tmp_path / "heads"
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(model_dir))
    for module in range(2, depth + 1):
        next_dir = tmp_path / {2: "two", 3: "three"}[module]
        prepare_lfm3_mtp_checkpoint(str(model_dir), str(next_dir), mode="add-module")
        model_dir = next_dir
    model = _load_mtp(model_dir).eval()
    torch.manual_seed(2)
    first = torch.randint(1, 1000, (1, 7))
    second = torch.randint(1, 1000, (1, 6))

    first_loss = model(input_ids=first, labels=first).loss
    second_loss = model(input_ids=second, labels=second).loss
    packed_ids = torch.cat([first, second], dim=1)
    packed_positions = torch.cat(
        [torch.arange(first.shape[1]), torch.arange(second.shape[1])]
    ).unsqueeze(0)
    packed_loss = model(
        input_ids=packed_ids,
        attention_mask=torch.ones_like(packed_ids),
        position_ids=packed_positions,
        labels=packed_ids,
    ).loss
    shift = depth + 1
    first_items, second_items = first.shape[1] - shift, second.shape[1] - shift
    expected = (first_loss * first_items + second_loss * second_items) / (
        first_items + second_items
    )
    torch.testing.assert_close(packed_loss, expected, rtol=1e-5, atol=1e-5)


def test_empty_assistant_mask_has_backward_safe_zero_loss(tiny_lfm3, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "mtp"))
    model = _load_mtp(tmp_path / "mtp").train()
    input_ids = torch.randint(0, 1000, (2, 8))
    labels = torch.full_like(input_ids, -100)
    loss = model(input_ids=input_ids, labels=labels).loss
    assert loss.item() == 0.0
    loss.backward()
    assert model.mtp.fc.weight.grad is not None
    assert not model.mtp.fc.weight.grad.any()


@pytest.mark.parametrize("objective", ["hard", "soft"])
def test_add_module_appends_a_second_module(tiny_lfm3, tmp_path, objective):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "heads"))
    summary = prepare_lfm3_mtp_checkpoint(
        str(tmp_path / "heads"),
        str(tmp_path / "two"),
        mode="add-module",
        objective=objective,
    )
    config = json.loads((tmp_path / "two" / "config.json").read_text())
    assert summary["mtp_modules"] == 2
    assert config["mtp_num_modules"] == 2 and config["mtp_train_modules"] == [2]
    model = _load_mtp(tmp_path / "two")
    trainable = {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    }
    assert trainable == {"mtp2"}
    input_ids = torch.randint(0, 1000, (2, 16))
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        (p.grad is not None) == n.startswith("mtp2.")
        for n, p in model.named_parameters()
    )


def test_add_module_appends_and_trains_a_third_module(tiny_lfm3, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "heads"))
    prepare_lfm3_mtp_checkpoint(
        str(tmp_path / "heads"), str(tmp_path / "two"), mode="add-module"
    )
    summary = prepare_lfm3_mtp_checkpoint(
        str(tmp_path / "two"), str(tmp_path / "three"), mode="add-module"
    )
    assert summary["mtp_modules"] == 3
    model = _load_mtp(tmp_path / "three")
    trainable = {
        name.split(".")[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {"mtp3"}
    input_ids = torch.randint(0, 1000, (2, 16))
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        (parameter.grad is not None) == name.startswith("mtp3.")
        for name, parameter in model.named_parameters()
    )


def test_joint_mode_trains_backbone_and_mtp(tiny_lfm3, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "joint"), mode="joint")
    model = _load_mtp(tmp_path / "joint").train()
    input_ids = torch.randint(0, 1000, (2, 16))
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None for n, p in model.named_parameters() if n.startswith("mtp.")
    )
    assert any(
        p.grad is not None
        for n, p in model.named_parameters()
        if n.startswith("model.layers.")
    )


@pytest.mark.parametrize(("mode", "expected_scale"), [("heads", 1.0), ("joint", 0.3)])
def test_soft_objective_routes_kl_gradient(tiny_lfm3, tmp_path, mode, expected_scale):
    prepare_lfm3_mtp_checkpoint(
        str(tiny_lfm3), str(tmp_path / "soft"), mode=mode, objective="soft"
    )
    model = _load_mtp(tmp_path / "soft").train()
    input_ids = torch.randint(0, 1000, (2, 8))
    labels = input_ids.clone()

    model(input_ids=input_ids, labels=labels).loss.backward()
    routed_grad = model.mtp.fc.weight.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    positions = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
    with torch.no_grad():
        hidden = model.model(
            input_ids=input_ids, position_ids=positions
        ).last_hidden_state
        target_probs = F.softmax(
            F.linear(hidden[:, 1:-1], model.lm_head.weight).float(), dim=-1
        )
    draft_hidden = model.mtp_hidden(hidden, input_ids, positions)
    draft_log_probs = F.log_softmax(
        F.linear(draft_hidden[:, :-1], model.lm_head.weight.detach()).float(), dim=-1
    )
    kl = expected_scale * (
        F.kl_div(draft_log_probs, target_probs, reduction="sum") / labels[:, 2:].numel()
    )
    kl.backward()

    torch.testing.assert_close(
        routed_grad, model.mtp.fc.weight.grad, rtol=2e-4, atol=2e-5
    )


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


def test_rejects_invalid_prepare_options(tiny_lfm3, tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        prepare_lfm3_mtp_checkpoint(
            str(tiny_lfm3), str(tmp_path / "negative"), loss_weight=-0.1
        )
    with pytest.raises(ValueError, match="inside the source"):
        prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tiny_lfm3 / "nested-output"))
    with pytest.raises(ValueError, match="init_from_layer"):
        prepare_lfm3_mtp_checkpoint(
            str(tiny_lfm3), str(tmp_path / "bad-layer"), init_from_layer=-1
        )

    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "heads"))
    prepare_lfm3_mtp_checkpoint(
        str(tmp_path / "heads"), str(tmp_path / "two"), mode="add-module"
    )
    with pytest.raises(ValueError, match="single MTP module"):
        prepare_lfm3_mtp_checkpoint(
            str(tmp_path / "two"), str(tmp_path / "invalid-joint"), mode="joint"
        )
