from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from leap_finetune.config.job_config import JobConfig
from leap_finetune.quantization.gguf_export import GGUF_DIR
from leap_finetune.quantization.qat import (
    PROFILES,
    finalize_qat_after_peft,
    find_qat_config,
    get_profile,
    load_qat_metadata,
    prepare_dpo_reference_model,
    prepare_model_for_qat,
    set_qat_enabled,
    write_qat_metadata,
)
from leap_finetune.quantization.qat.ops import (
    affine_groupwise_ste,
    fp8_e4m3_per_tensor_ste,
    fp8_e4m3_per_token_ste,
    mxfp4_ste,
    mxfp8,
    mxfp8_ste,
    nvfp4,
    nvfp4_activation_ste,
    q4_0,
    q4_0_ste,
    q8_0,
    q8_0_ste,
    uniform_block_noise_ste,
    vllm_fp8_weight_ste,
)


def _native_q4_reference(value: torch.Tensor) -> torch.Tensor:
    blocks = value.reshape(-1, 32).float()
    index = blocks.abs().argmax(dim=1, keepdim=True)
    scale = blocks.gather(1, index) / -8.0
    inverse = torch.where(scale != 0, scale.reciprocal(), torch.zeros_like(scale))
    quantized = (blocks * inverse + 8.5).to(torch.int32).clamp(0, 15)
    stored_scale = scale.half().float()
    return (
        ((quantized.float() - 8.0) * stored_scale).to(value.dtype).reshape(value.shape)
    )


def _native_q8_reference(value: torch.Tensor) -> torch.Tensor:
    blocks = value.reshape(-1, 32).float()
    scale = blocks.abs().amax(dim=1, keepdim=True) / 127.0
    inverse = torch.where(scale != 0, scale.reciprocal(), torch.zeros_like(scale))
    scaled = blocks * inverse
    quantized = scaled.abs().add(0.5).floor().copysign(scaled).clamp(-128, 127)
    stored_scale = scale.half().float()
    return (quantized * stored_scale).to(value.dtype).reshape(value.shape)


def test_gguf_kernels_match_native_equations():
    value = torch.linspace(-5.0, 4.0, 64, dtype=torch.bfloat16).reshape(2, 32)
    torch.testing.assert_close(q4_0(value), _native_q4_reference(value))
    torch.testing.assert_close(q8_0(value), _native_q8_reference(value))


@pytest.mark.parametrize(("qtype_name", "quantizer"), [("Q4_0", q4_0), ("Q8_0", q8_0)])
def test_gguf_kernels_match_bundled_native_reference(
    monkeypatch, qtype_name, quantizer
):
    monkeypatch.syspath_prepend(str(GGUF_DIR / "gguf-py"))
    gguf = importlib.import_module("gguf")
    generator = np.random.default_rng(42)
    value = generator.normal(size=(1000, 32)).astype(np.float32)
    qtype = getattr(gguf.GGMLQuantizationType, qtype_name)
    reference = gguf.dequantize(gguf.quantize(value, qtype), qtype)
    actual = quantizer(torch.from_numpy(value)).numpy()
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize(
    ("weight_qtype_name", "weight_quantizer"),
    [("Q4_0", q4_0), ("Q8_0", q8_0)],
)
def test_gguf_fake_quantized_linear_matches_bundled_native_dequantized_matmul(
    monkeypatch, weight_qtype_name, weight_quantizer
):
    """Check the composed W4A8/W8A8 layer contract, not just each tensor."""
    monkeypatch.syspath_prepend(str(GGUF_DIR / "gguf-py"))
    gguf = importlib.import_module("gguf")
    generator = np.random.default_rng(42)
    activation = generator.normal(size=(7, 64)).astype(np.float32)
    weight = generator.normal(size=(48, 64)).astype(np.float32)

    q8_type = gguf.GGMLQuantizationType.Q8_0
    weight_type = getattr(gguf.GGMLQuantizationType, weight_qtype_name)
    native_activation = gguf.dequantize(gguf.quantize(activation, q8_type), q8_type)
    native_weight = gguf.dequantize(gguf.quantize(weight, weight_type), weight_type)
    expected = native_activation @ native_weight.T

    actual = F.linear(
        q8_0(torch.from_numpy(activation)),
        weight_quantizer(torch.from_numpy(weight)),
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)


def test_bundled_gguf_maps_current_lfm2_dense_tensor_names(monkeypatch):
    monkeypatch.syspath_prepend(str(GGUF_DIR / "gguf-py"))
    gguf = importlib.import_module("gguf")
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LFM2, 1)
    suffixes = (".weight", ".bias")

    expected = {
        "model.layers.0.feed_forward.w1.weight": "blk.0.ffn_gate.weight",
        "model.layers.0.feed_forward.w2.weight": "blk.0.ffn_down.weight",
        "model.layers.0.feed_forward.w3.weight": "blk.0.ffn_up.weight",
        "model.layers.0.ffn_norm.weight": "blk.0.ffn_norm.weight",
        "model.layers.0.self_attn.q_layernorm.weight": "blk.0.attn_q_norm.weight",
        "model.layers.0.self_attn.k_layernorm.weight": "blk.0.attn_k_norm.weight",
    }
    assert {
        name: mapping.get_name(name, try_suffixes=suffixes) for name in expected
    } == expected


@pytest.mark.parametrize(
    "quantizer",
    [
        q4_0_ste,
        q8_0_ste,
        lambda x: affine_groupwise_ste(x, bits=4),
        lambda x: affine_groupwise_ste(x, bits=8),
        fp8_e4m3_per_tensor_ste,
        fp8_e4m3_per_token_ste,
        mxfp4_ste,
        mxfp8_ste,
        nvfp4_activation_ste,
        lambda x: uniform_block_noise_ste(x, bits=4),
        lambda x: uniform_block_noise_ste(x, bits=8),
    ],
)
def test_quantizers_preserve_shape_dtype_and_use_ste(quantizer):
    value = torch.randn(2, 3, 37, dtype=torch.float32, requires_grad=True)
    output = quantizer(value)
    assert output.shape == value.shape
    assert output.dtype == value.dtype
    assert torch.isfinite(output).all()
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_noise_is_seeded_and_non_constant():
    value = torch.linspace(-1, 1, 64)
    torch.manual_seed(9)
    first = uniform_block_noise_ste(value, bits=4)
    torch.manual_seed(9)
    second = uniform_block_noise_ste(value, bits=4)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, value)


class _DenseModel(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        self.lm_head = nn.Linear(width, width, bias=False)

    def forward(self, value):
        return self.lm_head(self.proj(value))


def test_gguf_linear_injection_quantizes_input_and_weight_at_the_matmul():
    model = _DenseModel()
    value = torch.randn(2, 3, 32)
    weight = model.proj.weight.detach().clone()
    prepare_model_for_qat(model, {"qat": {"type": "gguf_q4_0"}})
    expected = F.linear(q8_0(value), q4_0(weight))
    torch.testing.assert_close(model.proj(value), expected)


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_every_profile_prepares_dense_model_without_state_dict_changes(profile_name):
    model = _DenseModel()
    keys_before = list(model.state_dict())
    report = prepare_model_for_qat(model, {"qat": {"type": profile_name}})
    assert report is not None
    assert report.linears == ["proj"]
    assert report.excluded == ["lm_head"]
    assert list(model.state_dict()) == keys_before
    output = model(torch.randn(2, 32)).sum()
    output.backward()
    assert model.proj.weight.grad is not None


def test_dpo_reference_preparation_stays_inside_qat_api():
    loaded = []

    def load_model():
        model = _DenseModel()
        loaded.append(model)
        return model

    config = {"qat": {"type": "gguf_q4_0", "quantize_reference": True}}
    reference = prepare_dpo_reference_model(
        config, policy_uses_peft=False, load_model=load_model
    )
    assert reference is loaded[0]
    assert find_qat_config(reference) == config["qat"]

    reused = prepare_dpo_reference_model(
        config, policy_uses_peft=True, load_model=load_model
    )
    assert reused is None
    assert len(loaded) == 1

    floating = prepare_dpo_reference_model(
        {"qat": {"type": "gguf_q4_0", "quantize_reference": False}},
        policy_uses_peft=True,
        load_model=load_model,
    )
    assert floating is loaded[-1]
    assert find_qat_config(floating) is None


def test_qat_can_be_disabled_without_changing_parameters():
    model = _DenseModel()
    prepare_model_for_qat(model, {"qat": {"type": "gguf_q4_0"}})
    value = torch.randn(2, 32)
    set_qat_enabled(model, False)
    torch.testing.assert_close(
        model(value), F.linear(F.linear(value, model.proj.weight), model.lm_head.weight)
    )


class _ToyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Sequential(nn.Linear(32, 32))
        self.multi_modal_projector = nn.Sequential(nn.Linear(32, 32))
        self.language_model = nn.Sequential(nn.Linear(32, 32))
        self.lm_head = nn.Linear(32, 32)


@pytest.mark.parametrize(
    ("profile", "projector_quantized"),
    [
        ("gguf_q4_0", False),
        ("mlx_q4", True),
        ("vllm_fp8", True),
        ("vllm_mxfp8", True),
        ("vllm_nvfp4", True),
        ("noise_q4", True),
    ],
)
def test_vlm_targeting_never_quantizes_vision_tower(profile, projector_quantized):
    model = _ToyVLM()
    report = prepare_model_for_qat(model, {"qat": {"type": profile}}, is_vlm=True)
    assert report is not None
    assert "vision_tower.0" in report.excluded
    assert ("multi_modal_projector.0" in report.linears) is projector_quantized
    assert "language_model.0" in report.linears
    assert "lm_head" in report.excluded


class _ToyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.gate_up_proj = nn.Parameter(torch.randn(2, 64, 32))
        self.down_proj = nn.Parameter(torch.randn(2, 32, 32))
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        raise AssertionError("QAT should replace this execution path")


class _ToyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(32, 2, bias=False)
        self.experts = _ToyExperts()
        self.lm_head = nn.Linear(32, 32, bias=False)


@pytest.mark.parametrize("profile", ["gguf_q4_0", "vllm_mxfp8", "vllm_nvfp4"])
def test_moe_targets_packed_experts_but_not_router(profile):
    model = _ToyMoE()
    keys_before = list(model.state_dict())
    report = prepare_model_for_qat(model, {"qat": {"type": profile}})
    assert report is not None
    assert report.expert_tensors == ["experts.gate_up_proj", "experts.down_proj"]
    assert "gate" in report.excluded
    assert list(model.state_dict()) == keys_before

    hidden = torch.randn(4, 32, requires_grad=True)
    indices = torch.tensor([[0], [1], [0], [1]])
    weights = torch.ones(4, 1)
    model.experts(hidden, indices, weights).sum().backward()
    assert model.experts.gate_up_proj.grad is not None
    assert model.experts.down_proj.grad is not None


def test_expert_parallel_compute_uses_fake_quantized_weights(monkeypatch):
    from leap_finetune.training.moe_utils import ep_runtime

    experts = _ToyExperts()
    prepare_model_for_qat(experts, {"qat": {"type": "gguf_q4_0"}})
    observed_weights = []

    def fake_grouped_mm(value, weights, offsets):
        observed_weights.append(weights)
        chunks = []
        start = 0
        for expert_idx, end in enumerate(offsets.tolist()):
            chunks.append(value[start:end] @ weights[expert_idx])
            start = end
        return torch.cat(chunks)

    monkeypatch.setattr(ep_runtime, "grouped_mm", fake_grouped_mm)
    tokens = torch.randn(4, 32, requires_grad=True)
    output = ep_runtime.compute_local_experts(
        experts,
        tokens,
        torch.tensor([2, 2]),
    )
    output.sum().backward()

    assert len(observed_weights) == 2
    torch.testing.assert_close(
        observed_weights[0], q4_0(experts.gate_up_proj).transpose(-2, -1)
    )
    torch.testing.assert_close(
        observed_weights[1], q4_0(experts.down_proj).transpose(-2, -1)
    )
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None


class _AdapterLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(32, 32, bias=False)
        self.adapter = nn.Linear(32, 32, bias=False)
        self.adapter_input = None

    def forward(self, value):
        self.adapter_input = value.detach().clone()
        return self.base_layer(value) + self.adapter(value)


class _AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _AdapterLayer()

    def forward(self, value):
        return self.proj(value)


def test_peft_style_wrapper_quantizes_input_once_for_base_and_adapter():
    model = _AdapterModel()
    prepare_model_for_qat(model, {"qat": {"type": "gguf_q4_0"}})
    finalize_qat_after_peft(model)
    value = torch.randn(2, 32)
    expected = q8_0(value)
    model(value)
    torch.testing.assert_close(model.proj.adapter_input, expected)
    assert model.proj.base_layer._leap_qat_quantize_activation is False


def test_peft_style_wrapper_can_disable_weight_and_shared_activation_qat():
    model = _AdapterModel()
    prepare_model_for_qat(model, {"qat": {"type": "gguf_q4_0"}})
    finalize_qat_after_peft(model)
    value = torch.randn(2, 32)
    set_qat_enabled(model, False)
    model(value)
    torch.testing.assert_close(model.proj.adapter_input, value)
    expected = F.linear(value, model.proj.base_layer.weight) + F.linear(
        value, model.proj.adapter.weight
    )
    torch.testing.assert_close(model(value), expected)


def test_metadata_round_trip_and_resume_mismatch(tmp_path):
    model = _DenseModel()
    config = {"type": "gguf_q8_0", "quantize_reference": True}
    prepare_model_for_qat(model, {"qat": config})
    write_qat_metadata(tmp_path, model)
    assert load_qat_metadata(tmp_path) == config
    payload = json.loads((tmp_path / "qat_config.json").read_text())
    assert payload["format_version"] == 1

    with pytest.raises(ValueError, match="profile mismatch"):
        prepare_model_for_qat(
            _DenseModel(),
            {"qat": {"type": "gguf_q4_0", "quantize_reference": True}},
            resume_from_checkpoint=str(tmp_path),
        )


def _job_payload(training_type="sft", **training_config):
    return {
        "project_name": "qat-test",
        "model_name": "LiquidAI/LFM2-1.2B",
        "training_type": training_type,
        "dataset": {
            "path": "dummy",
            "type": "grpo" if "grpo" in training_type else "sft",
        },
        "training_config": training_config,
    }


def test_qat_config_is_nested_and_strict():
    job = JobConfig.model_validate(_job_payload(qat={"type": "mlx_q4"}))
    assert job.training_config.qat.type == "mlx_q4"
    assert job.training_config.qat.quantize_reference is True
    with pytest.raises(ValueError, match="literal_error"):
        JobConfig.model_validate(_job_payload(qat={"type": "q4"}))
    with pytest.raises(ValueError, match="extra_forbidden"):
        JobConfig.model_validate(_job_payload(qat={"type": "mlx_q4", "bits": 4}))

    targeted = JobConfig.model_validate(
        _job_payload(qat={"type": "vllm_fp8", "target": "rocm_mi300"})
    )
    assert targeted.training_config.qat.target == "rocm_mi300"
    with pytest.raises(ValueError, match="target is only valid for vllm_fp8"):
        JobConfig.model_validate(
            _job_payload(qat={"type": "gguf_q4_0", "target": "cuda"})
        )


def test_qat_grpo_rejects_vllm_rollouts():
    with pytest.raises(ValueError, match="use_vllm: false"):
        JobConfig.model_validate(
            _job_payload("grpo", qat={"type": "gguf_q4_0"}, use_vllm=True)
        )


def test_grpo_reference_is_prepared_at_creation_before_wrapping(monkeypatch):
    import trl.trainer.grpo_trainer as grpo_module

    from leap_finetune.quantization.qat.grpo import QATGRPOReferenceMixin

    events = []

    def reference_factory(*args, **kwargs):
        events.append("created")
        return _DenseModel()

    monkeypatch.setattr(grpo_module, "create_model_from_path", reference_factory)

    class FakeGRPOBase:
        def __init__(self):
            self._is_vlm = True
            self.ref_model = grpo_module.create_model_from_path("unused")
            assert find_qat_config(self.ref_model) == {"type": "gguf_q8_0"}
            events.append("wrapped")

    class FakeQATGRPOTrainer(QATGRPOReferenceMixin, FakeGRPOBase):
        pass

    trainer = FakeQATGRPOTrainer(qat_config={"type": "gguf_q8_0"})
    assert events == ["created", "wrapped"]
    assert trainer.ref_model.proj._leap_qat_profile == "gguf_q8_0"


def test_quality_matrix_expands_baseline_and_qat_profiles(tmp_path):
    from leap_finetune.quantization.qat.matrix import (
        expand_quality_manifest,
        write_expanded_configs,
    )

    manifest = {
        "seed": 42,
        "profiles": ["gguf_q4_0", "noise_q8"],
        "runs": [
            {
                "id": "dense_sft",
                "config": _job_payload("sft", extends="DEFAULT_SFT"),
            }
        ],
    }
    expanded = expand_quality_manifest(manifest)
    assert set(expanded) == {
        "dense_sft__baseline",
        "dense_sft__gguf_q4_0",
        "dense_sft__noise_q8",
    }
    assert "qat" not in expanded["dense_sft__baseline"]["training_config"]
    assert expanded["dense_sft__gguf_q4_0"]["training_config"]["qat"] == {
        "type": "gguf_q4_0"
    }

    manifest_path = tmp_path / "matrix.yaml"
    import yaml

    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    paths = write_expanded_configs(manifest_path, tmp_path / "jobs")
    assert len(paths) == 3
    assert all(path.is_file() for path in paths)


def test_quality_report_keeps_pending_cells_and_computes_deltas(tmp_path):
    from leap_finetune.quantization.qat.report import write_quality_report

    rows = [
        {
            "model_type": "dense",
            "training_type": "sft",
            "profile": "gguf_q4_0",
            "deployment_format": "gguf_q4_0",
            "metric": "accuracy",
            "bf16_score": 0.8,
            "ptq_score": 0.6,
            "qat_score": 0.7,
        },
        {
            "model_type": "vision",
            "training_type": "dpo",
            "profile": "mlx_q4",
            "deployment_format": "mlx_q4",
        },
    ]
    paths = write_quality_report(rows, tmp_path)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))["results"]
    assert payload[0]["qat_vs_ptq_delta"] == pytest.approx(0.1)
    assert payload[0]["ptq_degradation"] == pytest.approx(0.2)
    assert payload[0]["qat_degradation"] == pytest.approx(0.1)
    assert payload[1]["status"] == "pending"
    lower_is_better = {
        "metric": "perplexity",
        "higher_is_better": False,
        "bf16_score": 10.0,
        "ptq_score": 12.0,
        "qat_score": 11.0,
    }
    lower = write_quality_report([lower_is_better], tmp_path / "lower")
    lower_payload = json.loads(lower["json"].read_text(encoding="utf-8"))["results"][0]
    assert lower_payload["qat_vs_ptq_delta"] == pytest.approx(1.0)
    assert lower_payload["ptq_degradation"] == pytest.approx(2.0)
    assert lower_payload["qat_degradation"] == pytest.approx(1.0)
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "| metric |" in markdown
    assert "accuracy" in markdown
    assert "| 0.8 | 0.6 | 0.7 | 0.1 |" in markdown
    assert "pending" in markdown
    assert paths["csv"].is_file()


def test_mxfp4_matches_native_round_up_contract():
    value = torch.zeros(32)
    value[0] = 7.0
    value[1] = 3.0
    quantized = mxfp4_ste(value)
    # vLLM rounds 7 / 6 up to an E8M0 scale of 2, then resolves the
    # normalized 3.5 E2M1 tie upward to 4.
    assert quantized[0].item() == 8.0
    assert quantized[1].item() == 3.0


def test_mxfp8_uses_e4m3_values_and_e8m0_group32_scales():
    value = torch.zeros(64)
    value[0] = 500.0
    value[32] = 1.0
    quantized = mxfp8(value)
    first_scale = 2.0  # ceil(log2(500 / 448))
    second_scale = 2.0**-8  # ceil(log2(1 / 448))
    expected = value.clone()
    expected[:32] = (value[:32] / first_scale).to(
        torch.float8_e4m3fn
    ).float() * first_scale
    expected[32:] = (value[32:] / second_scale).to(
        torch.float8_e4m3fn
    ).float() * second_scale
    torch.testing.assert_close(quantized, expected, rtol=0, atol=0)


def test_nvfp4_matches_vllm_reference_scale_and_tie_contract():
    value = torch.zeros(2, 16)
    value[0, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    value[1, 0] = 12.0
    result = nvfp4(value)

    global_multiplier = 2688.0 / value.abs().max()
    blocks = value.reshape(-1, 16)
    block_amax = blocks.abs().amax(dim=1, keepdim=True)
    block_scale = (global_multiplier * block_amax / 6.0).clamp(0, 448)
    block_scale = block_scale.to(torch.float8_e4m3fn).float()
    scaled = (blocks * global_multiplier / block_scale.clamp_min(1e-30)).clamp(-6, 6)
    reference = torch.zeros_like(scaled)
    magnitude = scaled.abs()
    reference = torch.where((magnitude > 0.25) & (magnitude < 0.75), 0.5, reference)
    reference = torch.where((magnitude >= 0.75) & (magnitude <= 1.25), 1.0, reference)
    reference = torch.where((magnitude > 1.25) & (magnitude < 1.75), 1.5, reference)
    reference = torch.where((magnitude >= 1.75) & (magnitude <= 2.5), 2.0, reference)
    reference = torch.where((magnitude > 2.5) & (magnitude < 3.5), 3.0, reference)
    reference = torch.where((magnitude >= 3.5) & (magnitude <= 5.0), 4.0, reference)
    reference = torch.where(magnitude > 5.0, 6.0, reference)
    reference = reference.copysign(scaled) * block_scale / global_multiplier
    torch.testing.assert_close(result, reference.reshape_as(value), rtol=0, atol=0)


def test_grouped_profiles_report_incompatible_weights_instead_of_padding():
    model = _DenseModel(width=37)
    model.compatible_proj = nn.Linear(32, 32, bias=False)
    report = prepare_model_for_qat(model, {"qat": {"type": "gguf_q4_0"}})
    assert report is not None
    assert report.linears == ["compatible_proj"]
    assert report.incompatible == ["proj.weight"]
    assert "lm_head" in report.excluded
    assert not hasattr(model.proj, "_leap_qat_profile")


def test_fp8_profile_accepts_non_grouped_weight_widths():
    model = _DenseModel(width=37)
    report = prepare_model_for_qat(model, {"qat": {"type": "vllm_fp8"}})
    assert report is not None
    assert report.linears == ["proj"]
    assert report.incompatible == []


def test_vllm_fp8_target_selects_math_and_is_persisted(tmp_path):
    value = torch.ones(2, 2, 32)
    value[1] *= 1000
    cuda_value = get_profile("vllm_fp8", "cuda").activation_quantizer(value)
    rocm_value = get_profile("vllm_fp8", "rocm_mi300").activation_quantizer(value)
    torch.testing.assert_close(cuda_value, fp8_e4m3_per_token_ste(value))
    assert not torch.equal(cuda_value, rocm_value)

    model = _DenseModel()
    requested = {
        "type": "vllm_fp8",
        "target": "rocm_mi300",
        "quantize_reference": True,
    }
    prepare_model_for_qat(model, {"qat": requested})
    assert find_qat_config(model) == requested
    assert model.proj._leap_qat_target == "rocm_mi300"
    write_qat_metadata(tmp_path, model)
    assert load_qat_metadata(tmp_path) == requested

    with pytest.raises(ValueError, match="profile mismatch"):
        prepare_model_for_qat(
            _DenseModel(),
            {
                "qat": {
                    "type": "vllm_fp8",
                    "target": "cuda",
                    "quantize_reference": True,
                }
            },
            resume_from_checkpoint=str(tmp_path),
        )


def test_quantize_reference_override_is_dpo_only():
    dpo = JobConfig.model_validate(
        _job_payload(
            "dpo",
            qat={"type": "gguf_q4_0", "quantize_reference": False},
        )
    )
    assert dpo.training_config.qat.quantize_reference is False
    with pytest.raises(ValueError, match="only valid for DPO"):
        JobConfig.model_validate(
            _job_payload(
                "sft",
                qat={"type": "gguf_q4_0", "quantize_reference": False},
            )
        )


def test_fp8_packed_moe_weights_use_one_scale_per_expert_matrix():
    first = torch.linspace(-1.0, 1.0, 32 * 32).reshape(32, 32)
    second = torch.linspace(-1000.0, 1000.0, 32 * 32).reshape(32, 32)
    packed = torch.stack((first, second))
    packed_quantized = vllm_fp8_weight_ste(packed)
    torch.testing.assert_close(packed_quantized[0], fp8_e4m3_per_tensor_ste(first))
    torch.testing.assert_close(packed_quantized[1], fp8_e4m3_per_tensor_ste(second))


def test_fp8_per_tensor_activation_uses_one_scale_for_a_3d_tensor():
    value = torch.ones(2, 2, 32)
    value[1] *= 1000
    quantized = fp8_e4m3_per_tensor_ste(value)
    scale = value.abs().max() / torch.finfo(torch.float8_e4m3fn).max
    expected = (value / scale).clamp(
        -torch.finfo(torch.float8_e4m3fn).max, torch.finfo(torch.float8_e4m3fn).max
    ).to(torch.float8_e4m3fn).float() * scale
    torch.testing.assert_close(quantized, expected)


def test_manual_sharded_root_metadata_writes_qat_sidecar(tmp_path):
    from leap_finetune.checkpointing.manual_sharded import _save_root_metadata

    config = {"type": "noise_q8", "quantize_reference": True}
    _save_root_metadata(
        str(tmp_path),
        save_only_model=True,
        checkpoint_format="hf",
        export_metadata={"training_config": {"train_config": {"qat": config}}},
    )
    assert load_qat_metadata(tmp_path) == config
