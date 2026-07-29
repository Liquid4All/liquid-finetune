from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from leap_finetune.quantization.qat import ops
from leap_finetune.quantization.qat.experts import prepare_qat_experts
from leap_finetune.quantization.qat.metadata import find_qat_config, validate_qat_resume
from leap_finetune.quantization.qat.profiles import QATProfile, get_profile

logger = logging.getLogger(__name__)

_VISION_MARKERS = ("vision_tower", "vision_model", "vision_encoder")
_PROJECTOR_MARKERS = ("multi_modal_projector", "multimodal_projector", "projector")
_EXCLUDED_LEAVES = {"lm_head", "gate", "router", "embed_tokens", "embedding"}


@dataclass
class QATPreparationReport:
    profile: str
    linears: list[str] = field(default_factory=list)
    expert_tensors: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    incompatible: list[str] = field(default_factory=list)

    @property
    def transformed_count(self) -> int:
        return len(self.linears) + len(self.expert_tensors)


def _should_quantize_linear(name: str, profile: QATProfile, is_vlm: bool) -> bool:
    lowered = name.lower()
    leaf = lowered.rsplit(".", 1)[-1]
    if leaf in _EXCLUDED_LEAVES or "embed" in leaf:
        return False
    if is_vlm and any(marker in lowered for marker in _VISION_MARKERS):
        return False
    if (
        is_vlm
        and any(marker in lowered for marker in _PROJECTOR_MARKERS)
        and not profile.quantize_projector
    ):
        return False
    return True


def _qat_linear_forward(module: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    if not getattr(module, "_leap_qat_enabled", True):
        return F.linear(value, module.weight, module.bias)
    profile = get_profile(
        module._leap_qat_profile, getattr(module, "_leap_qat_target", None)
    )
    if profile.activation_quantizer is not None and getattr(
        module, "_leap_qat_quantize_activation", True
    ):
        value = profile.activation_quantizer(value)
    return F.linear(value, profile.weight_quantizer(module.weight), module.bias)


def _prepare_linear(
    module: nn.Linear, profile: QATProfile, target: str | None = None
) -> None:
    if hasattr(module, "_leap_qat_original_forward"):
        if module._leap_qat_profile != profile.name:
            raise ValueError(
                f"Linear already prepared with {module._leap_qat_profile}, cannot change to {profile.name}"
            )
        return
    module._leap_qat_original_forward = module.forward
    module._leap_qat_profile = profile.name
    module._leap_qat_target = target
    module._leap_qat_enabled = True
    module._leap_qat_quantize_activation = True
    module.forward = types.MethodType(_qat_linear_forward, module)


def _qat_experts_forward(experts, hidden_states, top_k_index, top_k_weights):
    """Reference packed-expert execution with fake-quantized weights."""
    if not getattr(experts, "_leap_qat_enabled", True):
        return experts._leap_qat_original_forward(
            hidden_states, top_k_index, top_k_weights
        )
    qat = prepare_qat_experts(experts, hidden_states)
    hidden_states = qat.tokens
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=experts.num_experts).permute(
            2, 1, 0
        )
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor[0])
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current = hidden_states[token_idx]
        gate, up = F.linear(current, qat.gate_up_weight[expert_idx]).chunk(2, dim=-1)
        current = experts.act_fn(gate) * up
        current = qat.prepare_down_input(current)
        current = F.linear(current, qat.down_weight[expert_idx])
        current = current * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, current.to(final.dtype))
    return final


def _is_compatible(weight: torch.Tensor, profile: QATProfile) -> bool:
    group_size = profile.weight_group_size
    return group_size is None or weight.shape[-1] % group_size == 0


def _prepare_experts(
    name: str, module: nn.Module, profile: QATProfile, target: str | None = None
) -> tuple[list[str], list[str]]:
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    if not isinstance(gate_up, nn.Parameter) or not isinstance(down, nn.Parameter):
        return [], []
    if gate_up.ndim != 3 or down.ndim != 3:
        return [], []
    gate_up_name = f"{name}.gate_up_proj"
    down_name = f"{name}.down_proj"
    quantize_gate_up = _is_compatible(gate_up, profile)
    quantize_down = _is_compatible(down, profile)
    compatible = [
        tensor_name
        for tensor_name, should_quantize in (
            (gate_up_name, quantize_gate_up),
            (down_name, quantize_down),
        )
        if should_quantize
    ]
    incompatible = [
        tensor_name
        for tensor_name, should_quantize in (
            (gate_up_name, quantize_gate_up),
            (down_name, quantize_down),
        )
        if not should_quantize
    ]
    if not compatible:
        return [], incompatible
    if not hasattr(module, "_leap_qat_original_forward"):
        module._leap_qat_original_forward = module.forward
        module._leap_qat_profile = profile.name
        module._leap_qat_target = target
        module._leap_qat_enabled = True
        module.forward = types.MethodType(_qat_experts_forward, module)
    module._leap_qat_quantize_gate_up = quantize_gate_up
    module._leap_qat_quantize_down = quantize_down
    return compatible, incompatible


def _qat_peft_input_hook(module: nn.Module, args):
    if not args:
        return args
    if not getattr(module, "_leap_qat_enabled", True):
        return args
    quantizer = get_profile(
        module._leap_qat_profile, getattr(module, "_leap_qat_target", None)
    ).activation_quantizer
    if quantizer is None:
        return args
    return (quantizer(args[0]), *args[1:])


def finalize_qat_after_peft(model: nn.Module) -> None:
    """Make PEFT base and adapter branches consume one quantized activation."""
    config = find_qat_config(model)
    if config is None:
        return
    profile = get_profile(config["type"], config.get("target"))
    if profile.activation_quantizer is None:
        return
    hooked = 0
    for module in model.modules():
        base_layer = getattr(module, "base_layer", None)
        if not isinstance(base_layer, nn.Linear):
            continue
        if getattr(base_layer, "_leap_qat_profile", None) != profile.name:
            continue
        if not hasattr(module, "_leap_qat_input_hook_handle"):
            module._leap_qat_profile = profile.name
            module._leap_qat_target = config.get("target")
            module._leap_qat_enabled = True
            module._leap_qat_input_hook_handle = module.register_forward_pre_hook(
                _qat_peft_input_hook
            )
            base_layer._leap_qat_quantize_activation = False
            hooked += 1
    if hooked:
        logger.info("QAT installed shared activation hooks on %d PEFT layers", hooked)


def prepare_model_for_qat(
    model: nn.Module,
    train_config: dict[str, Any],
    *,
    is_vlm: bool = False,
    resume_from_checkpoint: str | None = None,
) -> QATPreparationReport | None:
    raw_config = train_config.get("qat")
    if raw_config is None:
        return None
    config = (
        raw_config.model_dump(exclude_none=True)
        if hasattr(raw_config, "model_dump")
        else dict(raw_config)
    )
    if config["type"] == "vllm_fp8":
        parameter = next(model.parameters(), None)
        config["target"] = ops.resolve_vllm_fp8_target(config.get("target"), parameter)
    elif config.get("target") is not None:
        raise ValueError("QAT target is only valid for vllm_fp8")
    profile = get_profile(config["type"], config.get("target"))
    if resume_from_checkpoint:
        validate_qat_resume(Path(resume_from_checkpoint), config)
    existing = find_qat_config(model)
    if existing is not None:
        if existing != config:
            raise ValueError(
                f"Model already uses QAT config {existing}, requested {config}"
            )
        return QATPreparationReport(profile=profile.name)

    report = QATPreparationReport(profile=profile.name)
    for name, module in list(model.named_modules()):
        expert_names, incompatible_experts = _prepare_experts(
            name, module, profile, config.get("target")
        )
        if incompatible_experts:
            report.incompatible.extend(incompatible_experts)
        if expert_names or incompatible_experts:
            report.expert_tensors.extend(expert_names)
            continue
        if not isinstance(module, nn.Linear):
            continue
        if _should_quantize_linear(name, profile, is_vlm):
            if _is_compatible(module.weight, profile):
                _prepare_linear(module, profile, config.get("target"))
                report.linears.append(name)
            else:
                report.incompatible.append(f"{name}.weight")
        else:
            report.excluded.append(name)
    if report.transformed_count == 0:
        raise ValueError(
            f"QAT profile {profile.name!r} matched no supported model tensors"
        )
    model._leap_qat_config = config
    model._leap_qat_report = report
    logger.info(
        "QAT prepared profile=%s linears=%d expert_tensors=%d excluded=%d incompatible=%d",
        profile.name,
        len(report.linears),
        len(report.expert_tensors),
        len(report.excluded),
        len(report.incompatible),
    )
    logger.debug("QAT linear targets: %s", report.linears)
    logger.debug("QAT expert targets: %s", report.expert_tensors)
    logger.debug("QAT excluded targets: %s", report.excluded)
    logger.debug("QAT incompatible targets: %s", report.incompatible)
    return report


def set_qat_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "_leap_qat_enabled"):
            module._leap_qat_enabled = enabled
