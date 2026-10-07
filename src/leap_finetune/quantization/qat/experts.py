from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from leap_finetune.quantization.qat.profiles import QATProfile, get_profile


@dataclass(frozen=True)
class QATExpertsContext:
    """Fake-quantized inputs and weights for one packed expert MLP."""

    tokens: torch.Tensor
    gate_up_weight: torch.Tensor
    down_weight: torch.Tensor
    profile: QATProfile | None = None
    quantize_down: bool = False

    def prepare_down_input(self, value: torch.Tensor) -> torch.Tensor:
        if (
            self.profile is not None
            and self.profile.activation_quantizer is not None
            and self.quantize_down
        ):
            return self.profile.activation_quantizer(value)
        return value


def prepare_qat_experts(experts: nn.Module, tokens: torch.Tensor) -> QATExpertsContext:
    """Prepare an expert call without exposing QAT internals to the runtime."""
    profile_name = getattr(experts, "_leap_qat_profile", None)
    if not profile_name or not getattr(experts, "_leap_qat_enabled", True):
        return QATExpertsContext(
            tokens=tokens,
            gate_up_weight=experts.gate_up_proj,
            down_weight=experts.down_proj,
        )

    profile = get_profile(profile_name, getattr(experts, "_leap_qat_target", None))
    quantize_gate_up = getattr(experts, "_leap_qat_quantize_gate_up", True)
    quantize_down = getattr(experts, "_leap_qat_quantize_down", True)
    if profile.activation_quantizer is not None and quantize_gate_up:
        tokens = profile.activation_quantizer(tokens)

    return QATExpertsContext(
        tokens=tokens,
        gate_up_weight=(
            profile.weight_quantizer(experts.gate_up_proj)
            if quantize_gate_up
            else experts.gate_up_proj
        ),
        down_weight=(
            profile.weight_quantizer(experts.down_proj)
            if quantize_down
            else experts.down_proj
        ),
        profile=profile,
        quantize_down=quantize_down,
    )
