from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from leap_finetune.quantization.qat import prepare_model_for_qat, set_qat_enabled
from leap_finetune.quantization.qat.ops import (
    mxfp4,
    mxfp8,
    nvfp4,
    q4_0_ste,
    q8_0_ste,
)


LIQUID_LFM_ROOT = Path(os.environ.get("LIQUID_LFM_ROOT", Path.home() / "liquid_lfm"))
LIQUID_LFM_QAT = LIQUID_LFM_ROOT / "liquid_lfm" / "quantization" / "q4_0_fake_quant.py"


def _load_liquid_lfm_qat():
    if not LIQUID_LFM_QAT.is_file():
        pytest.skip(f"Liquid LFM checkout is not available at {LIQUID_LFM_QAT}")
    spec = importlib.util.spec_from_file_location(
        "liquid_lfm_qat_reference", LIQUID_LFM_QAT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gguf_ste_gradients_match_liquid_lfm(dtype):
    reference = _load_liquid_lfm_qat()
    value = torch.randn(3, 64, dtype=dtype)
    value[:, 0] = 0

    coefficient = torch.randn_like(value)
    leap_value = value.detach().clone().requires_grad_()
    ref_value = value.detach().clone().requires_grad_()
    (q4_0_ste(leap_value) * coefficient).sum().backward()
    (reference.fake_quantize_q4_0_ste(ref_value) * coefficient).sum().backward()
    torch.testing.assert_close(leap_value.grad, ref_value.grad)

    leap_value = value.detach().clone().requires_grad_()
    ref_value = value.detach().clone().requires_grad_()
    (q8_0_ste(leap_value) * coefficient).sum().backward()
    (reference.fake_quantize_q8_0_ste(ref_value) * coefficient).sum().backward()
    torch.testing.assert_close(leap_value.grad, ref_value.grad)


def test_mxfp4_matches_vllm_native_torch_reference():
    pytest.importorskip("vllm")
    from vllm.third_party.triton_kernels.numerics_details.mxfp import (
        downcast_to_mxfp_torch,
        upcast_from_mxfp_torch,
    )

    torch.manual_seed(42)
    value = torch.randn(2, 3, 64) * 10
    packed, scale = downcast_to_mxfp_torch(value, torch.uint8, axis=-1)
    reference = upcast_from_mxfp_torch(packed, scale, torch.float32, axis=-1)
    torch.testing.assert_close(mxfp4(value), reference, rtol=0, atol=0)


def test_mxfp8_matches_transformer_engine_scale_contract():
    torch.manual_seed(42)
    value = torch.randn(2, 3, 64) * 100
    blocks = value.reshape(-1, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scale = torch.pow(2.0, exponent)
    reference = (
        (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    ).reshape_as(value)
    torch.testing.assert_close(mxfp8(value), reference, rtol=0, atol=0)


def test_nvfp4_matches_vllm_reference_quant_dequant():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        ref_nvfp4_quant_dequant,
    )

    torch.manual_seed(42)
    value = torch.randn(6, 64, dtype=torch.float32) * 10
    quant_multiplier = torch.tensor([2688.0 / value.abs().max()], dtype=torch.float32)
    reference = ref_nvfp4_quant_dequant(value, quant_multiplier, block_size=16)
    torch.testing.assert_close(nvfp4(value), reference, rtol=0, atol=0)


def _dense_config():
    from transformers import Lfm2Config

    return Lfm2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        block_auto_adjust_ff_dim=False,
        layer_types=["full_attention"],
    )


def test_real_lfm2_dense_gguf_forward_backward_and_state_contract():
    from transformers import Lfm2ForCausalLM

    model = Lfm2ForCausalLM(_dense_config())
    keys_before = list(model.state_dict())
    report = prepare_model_for_qat(model, {"qat": {"type": "gguf_q4_0"}})
    assert report is not None and report.transformed_count > 0
    assert report.incompatible == []
    assert list(model.state_dict()) == keys_before

    input_ids = torch.randint(0, 64, (2, 8))
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_real_lfm2_moe_disable_restores_float_reference_path():
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM

    config = Lfm2MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_dense_layers=1,
        num_experts_per_tok=1,
        num_experts=2,
        layer_types=["full_attention", "full_attention"],
    )
    model = Lfm2MoeForCausalLM(config).eval()
    input_ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        float_logits = model(input_ids=input_ids, use_cache=False).logits

    keys_before = list(model.state_dict())
    report = prepare_model_for_qat(model, {"qat": {"type": "gguf_q4_0"}})
    assert report is not None
    assert any(name.endswith("experts.gate_up_proj") for name in report.expert_tensors)
    assert any(name.endswith("feed_forward.gate") for name in report.excluded)
    assert report.incompatible == []
    assert list(model.state_dict()) == keys_before

    qat_loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    assert torch.isfinite(qat_loss)
    qat_loss.backward()
    experts = model.model.layers[1].feed_forward.experts
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None

    set_qat_enabled(model, False)
    with torch.no_grad():
        restored_logits = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(restored_logits, float_logits)


def test_real_lfm2_vl_multimodal_forward_excludes_vision_and_trains_projector():
    from transformers import Lfm2VlConfig, Siglip2VisionConfig
    from transformers.models.lfm2_vl.modeling_lfm2_vl import (
        Lfm2VlForConditionalGeneration,
    )

    vision_config = Siglip2VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_patches=4,
        patch_size=2,
    )
    config = Lfm2VlConfig(
        vision_config=vision_config.to_dict(),
        text_config=_dense_config().to_dict(),
        projector_hidden_size=32,
        downsample_factor=1,
        image_token_id=63,
    )
    model = Lfm2VlForConditionalGeneration(config)
    keys_before = list(model.state_dict())
    report = prepare_model_for_qat(model, {"qat": {"type": "mlx_q4"}}, is_vlm=True)
    assert report is not None
    assert any(name.startswith("model.vision_tower") for name in report.excluded)
    assert "model.multi_modal_projector.linear_1" in report.linears
    assert "model.multi_modal_projector.linear_2" in report.linears
    assert list(model.state_dict()) == keys_before

    input_ids = torch.tensor([[63, 63, 63, 63, 4, 5, 6, 7]])
    output = model(
        input_ids=input_ids,
        labels=input_ids,
        pixel_values=torch.randn(1, 4, 12),
        spatial_shapes=torch.tensor([[2, 2]]),
        pixel_attention_mask=torch.ones(1, 4, dtype=torch.bool),
        use_cache=False,
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.model.multi_modal_projector.linear_1.weight.grad is not None
