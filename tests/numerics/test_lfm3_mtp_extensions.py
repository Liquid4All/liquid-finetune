"""Extensions of the base MTP tests: LFM3-VL, 1-5 draft modules, batching edge cases, preparation."""

import json
import shutil

import pytest
import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from liquid_finetune.models.lfm3_mtp import prepare_lfm3_mtp_checkpoint

LFM3_REPO = "LiquidAI/LFM3-1.7B"
LFM3_REVISION = "1da06bec6a819da10b54c67e5c90972b46d81871"
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
            shutil.copy(
                hub.hf_hub_download(LFM3_REPO, name, revision=LFM3_REVISION),
                root / name,
            )
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


LFM3_VL_REPO = "LiquidAI/LFM3-VL-2B"
LFM3_VL_REVISION = "23387beda43c60310348e83e25f78f33c3140d3a"
LFM3_VL_FILES = (
    "config.json",
    "configuration_lfm3.py",
    "configuration_lfm3_vl.py",
    "modeling_lfm3.py",
    "modeling_lfm3_vl.py",
    "lfm3_conv.py",
)
VL_PATCH_SIZE, VL_TEMPORAL_PATCH, VL_MERGE = 16, 2, 2


@pytest.fixture(scope="module")
def tiny_lfm3_vl(tmp_path_factory):
    """A tiny LFM3-VL (4-layer text, 2-layer vision) from the published remote code."""
    hub = pytest.importorskip("huggingface_hub")
    root = tmp_path_factory.mktemp("tiny_lfm3_vl")
    try:
        for name in LFM3_VL_FILES:
            shutil.copy(
                hub.hf_hub_download(LFM3_VL_REPO, name, revision=LFM3_VL_REVISION),
                root / name,
            )
    except Exception as exc:  # offline or Hub unavailable
        pytest.skip(f"LFM3-VL remote code unavailable: {exc}")
    config = json.loads((root / "config.json").read_text())
    config["text_config"].update(
        num_hidden_layers=len(LAYER_TYPES),
        layer_types=LAYER_TYPES,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config["vision_config"].update(
        num_hidden_layers=2,
        layer_types=["conv", "full_attention"],
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config["projector_hidden_size"] = 64
    (root / "config.json").write_text(json.dumps(config))
    config_cls = get_class_from_dynamic_module(
        "configuration_lfm3_vl.Lfm3VlConfig", str(root)
    )
    model_cls = get_class_from_dynamic_module(
        "modeling_lfm3_vl.Lfm3VlForConditionalGeneration", str(root)
    )
    torch.manual_seed(0)
    model = model_cls(config_cls.from_pretrained(str(root)))
    state = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(root / "model.safetensors"), metadata={"format": "pt"})
    return root


def _vl_batch(model_dir):
    """Two rows: text, one 4x4-patch image (-> 4 image tokens), more text; labels on text only."""
    config = json.loads((model_dir / "config.json").read_text())
    image_token = config["image_token_id"]
    grid = torch.tensor([[1, 4, 4]])
    n_image_tokens = int(grid.prod()) // VL_MERGE**2
    patch_dim = 3 * VL_TEMPORAL_PATCH * VL_PATCH_SIZE**2
    rows = []
    for _ in range(2):
        rows.append(
            torch.cat(
                [
                    torch.randint(0, 1000, (3,)),
                    torch.full((n_image_tokens,), image_token),
                    torch.randint(0, 1000, (9,)),
                ]
            )
        )
    input_ids = torch.stack(rows)
    labels = input_ids.clone()
    labels[input_ids == image_token] = -100
    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": torch.randn(2 * int(grid.prod()), patch_dim),
        "image_grid_thw": grid.repeat(2, 1),
    }


def _load_vl_mtp(model_dir):
    cls = get_class_from_dynamic_module(
        "modeling_lfm3_vl_mtp.Lfm3VlForConditionalGenerationWithMTP", str(model_dir)
    )
    cls.register_for_auto_class("AutoModelForImageTextToText")
    return cls.from_pretrained(str(model_dir), dtype=torch.float32)


def test_vl_heads_mode_writes_vl_auto_map(tiny_lfm3_vl, tmp_path):
    summary = prepare_lfm3_mtp_checkpoint(str(tiny_lfm3_vl), str(tmp_path / "mtp"))
    config = json.loads((tmp_path / "mtp" / "config.json").read_text())
    assert summary["backbone"] == "lfm3_vl"
    assert config["auto_map"]["AutoModelForImageTextToText"] == (
        "modeling_lfm3_vl_mtp.Lfm3VlForConditionalGenerationWithMTP"
    )
    assert "AutoModelForCausalLM" not in config["auto_map"]
    assert config["mtp_init_from_layer"] == 1  # last full-attention text layer
    assert {"modeling_lfm3_mtp.py", "modeling_lfm3_vl_mtp.py"} <= {
        p.name for p in (tmp_path / "mtp").iterdir()
    }


def test_vl_frozen_backbone_trains_only_the_mtp_module(tiny_lfm3_vl, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3_vl), str(tmp_path / "mtp"))
    model = _load_vl_mtp(tmp_path / "mtp")
    trainable = {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    }
    assert trainable == {"mtp"}  # vision tower and projector stay frozen too

    torch.manual_seed(0)
    batch = _vl_batch(tmp_path / "mtp")
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        (p.grad is not None) == n.startswith("mtp.")
        for n, p in model.named_parameters()
    )


def test_vl_draft_input_uses_the_image_features(tiny_lfm3_vl, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3_vl), str(tmp_path / "mtp"))
    model = _load_vl_mtp(tmp_path / "mtp")
    batch = _vl_batch(tmp_path / "mtp")
    input_ids = batch["input_ids"]
    positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
    with torch.no_grad():
        h, embeds = model._mtp_backbone(
            input_ids,
            None,
            positions,
            {
                "pixel_values": batch["pixel_values"],
                "image_grid_thw": batch["image_grid_thw"],
            },
        )
        token_embeds = model.get_input_embeddings()(input_ids)
    image = input_ids == model.config.image_token_id
    assert h.shape == embeds.shape == token_embeds.shape
    torch.testing.assert_close(embeds[~image], token_embeds[~image])
    assert not torch.allclose(embeds[image], token_embeds[image])


def test_text_and_vl_mtp_modules_share_one_weight_layout(
    tiny_lfm3, tiny_lfm3_vl, tmp_path
):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "text"))
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3_vl), str(tmp_path / "vl"))
    text = {
        k: v.shape
        for k, v in _load_mtp(tmp_path / "text").state_dict().items()
        if k.startswith("mtp")
    }
    vl = {
        k: v.shape
        for k, v in _load_vl_mtp(tmp_path / "vl").state_dict().items()
        if k.startswith("mtp")
    }
    assert text and text == vl


def test_vl_add_module_and_reload(tiny_lfm3_vl, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3_vl), str(tmp_path / "heads"))
    prepare_lfm3_mtp_checkpoint(
        str(tmp_path / "heads"), str(tmp_path / "two"), mode="add-module"
    )
    model = _load_vl_mtp(tmp_path / "two")
    assert {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    } == {"mtp2"}
    batch = _vl_batch(tmp_path / "two")
    model(**batch).loss.backward()
    model.save_pretrained(str(tmp_path / "saved"))
    config = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert config["auto_map"]["AutoModelForImageTextToText"].startswith(
        "modeling_lfm3_vl_mtp."
    )
    reloaded = _load_vl_mtp(tmp_path / "saved")
    assert sum(k.startswith("mtp2.") for k in reloaded.state_dict()) > 0
    with torch.no_grad():  # inference (no labels) is the unchanged LFM3-VL forward
        batch.pop("labels")
        logits = reloaded(**batch).logits
    assert logits.shape[:2] == batch["input_ids"].shape


def test_num_draft_tokens_builds_that_many_sequential_modules(tiny_lfm3, tmp_path):
    summary = prepare_lfm3_mtp_checkpoint(
        str(tiny_lfm3), str(tmp_path / "five"), num_draft_tokens=5
    )
    assert summary["mtp_modules"] == 5
    model = _load_mtp(tmp_path / "five")
    assert model._present_modules() == [1, 2, 3, 4, 5]
    assert {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    } == {"mtp", "mtp2", "mtp3", "mtp4", "mtp5"}
    input_ids = torch.randint(0, 1000, (2, 16))
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    assert all(
        p.grad is not None for n, p in model.named_parameters() if p.requires_grad
    )
    positions = torch.arange(16)[None].expand(2, -1)
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids).last_hidden_state
        states = model.mtp_draft_states(
            hidden, model.model.embed_tokens(input_ids), positions, 3
        )
    assert [s.shape[1] for s in states] == [15, 14, 13]


def test_more_draft_tokens_tops_up_and_trains_only_the_new_modules(tiny_lfm3, tmp_path):
    prepare_lfm3_mtp_checkpoint(
        str(tiny_lfm3), str(tmp_path / "two"), num_draft_tokens=2
    )
    prepare_lfm3_mtp_checkpoint(
        str(tmp_path / "two"), str(tmp_path / "four"), num_draft_tokens=4
    )
    config = json.loads((tmp_path / "four" / "config.json").read_text())
    assert config["mtp_num_modules"] == 4 and config["mtp_train_modules"] == [3, 4]
    model = _load_mtp(tmp_path / "four")
    assert {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    } == {"mtp3", "mtp4"}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_draft_tokens": 6}, "1-5"),
        ({"num_draft_tokens": 2, "mode": "joint"}, "joint"),
    ],
)
def test_rejects_unsupported_draft_counts(tiny_lfm3, tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "out"), **kwargs)


def test_vl_mtp_checkpoint_generates_with_images(tiny_lfm3_vl, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3_vl), str(tmp_path / "mtp"))
    model = _load_vl_mtp(tmp_path / "mtp").eval()
    batch = _vl_batch(tmp_path / "mtp")
    batch.pop("labels")
    out = model.generate(**batch, max_new_tokens=3, do_sample=False)
    assert out.shape == (2, batch["input_ids"].shape[1] + 3)


def _frozen_model(tiny_lfm3, tmp_path, num_draft_tokens=1, **kwargs):
    prepare_lfm3_mtp_checkpoint(
        str(tiny_lfm3),
        str(tmp_path / f"mtp{num_draft_tokens}"),
        num_draft_tokens=num_draft_tokens,
        **kwargs,
    )
    return _load_mtp(tmp_path / f"mtp{num_draft_tokens}")


@pytest.mark.parametrize("num_draft_tokens", [1, 3])
def test_a_batch_without_supervised_tokens_still_backpropagates(
    tiny_lfm3, tmp_path, num_draft_tokens
):
    model = _frozen_model(tiny_lfm3, tmp_path, num_draft_tokens)
    input_ids = torch.randint(0, 1000, (2, 12))
    labels = torch.full_like(input_ids, -100)
    loss = model(
        input_ids=input_ids, labels=labels, num_items_in_batch=torch.tensor(0)
    ).loss
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()  # a detached zero would raise here and stall the other ranks
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("length", [2, 5, 6])
def test_batches_shorter_than_the_deepest_module_train(tiny_lfm3, tmp_path, length):
    model = _frozen_model(tiny_lfm3, tmp_path, num_draft_tokens=5)
    input_ids = torch.randint(0, 1000, (2, length))
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("num_draft_tokens", [1, 2])
@pytest.mark.parametrize("objective", ["hard", "soft"])
def test_packed_documents_score_like_separate_documents(
    tiny_lfm3, tmp_path, num_draft_tokens, objective
):
    """No draft may mix the end of one document with the start of the next."""
    model = _frozen_model(
        tiny_lfm3,
        tmp_path / f"{num_draft_tokens}{objective}",
        num_draft_tokens,
        objective=objective,
    )
    model.eval()
    torch.manual_seed(1)
    doc_a, doc_b = torch.randint(0, 1000, (1, 9)), torch.randint(0, 1000, (1, 7))
    packed_ids = torch.cat([doc_a, doc_b], dim=1)
    packed_positions = torch.cat([torch.arange(9), torch.arange(7)])[None]
    total = torch.tensor(100)
    with torch.no_grad():
        packed = model(
            input_ids=packed_ids,
            position_ids=packed_positions,
            labels=packed_ids,
            num_items_in_batch=total,
        ).loss
        separate = sum(
            model(input_ids=doc, labels=doc, num_items_in_batch=total).loss
            for doc in (doc_a, doc_b)
        )
    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=1e-5)


def test_checkpoint_without_its_mtp_weights_is_rejected(tiny_lfm3, tmp_path):
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(tmp_path / "mtp"))
    config_path = tmp_path / "mtp" / "config.json"
    config = json.loads(config_path.read_text())
    config["mtp_num_modules"] = 2  # asks for mtp2.*, which the weights do not have
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="no weights for its configured MTP modules"):
        _load_mtp(tmp_path / "mtp")


def test_a_failed_preparation_leaves_no_start_point_behind(tiny_lfm3, tmp_path):
    out = tmp_path / "mtp"
    with pytest.raises(ValueError, match="init_from_layer"):
        prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(out), init_from_layer=0)
    assert not out.exists()
    prepare_lfm3_mtp_checkpoint(str(tiny_lfm3), str(out))  # a retry starts clean
    assert (out / "mtp_manifest.json").is_file()


@pytest.mark.parametrize("count", [0, -1, 6])
def test_invalid_draft_counts_are_rejected(tiny_lfm3, tmp_path, count):
    from liquid_finetune.models.lfm3_mtp.prepare import main

    with pytest.raises(ValueError, match="1-5"):
        prepare_lfm3_mtp_checkpoint(
            str(tiny_lfm3), str(tmp_path / "out"), num_draft_tokens=count
        )
    with pytest.raises(SystemExit):  # argparse: must be between 1 and 5
        main([str(tiny_lfm3), str(tmp_path / "cli"), "--num-draft-tokens", str(count)])
