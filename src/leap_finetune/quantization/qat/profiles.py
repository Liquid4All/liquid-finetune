from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from leap_finetune.quantization.qat import ops

TensorQuantizer = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class QATProfile:
    name: str
    weight_quantizer: TensorQuantizer
    activation_quantizer: TensorQuantizer | None
    quantize_projector: bool
    deployment_formats: tuple[str, ...]
    weight_group_size: int | None = None


PROFILES: dict[str, QATProfile] = {
    "gguf_q4_0": QATProfile(
        "gguf_q4_0", ops.q4_0_ste, ops.q8_0_ste, False, ("gguf_q4_0",), 32
    ),
    "gguf_q8_0": QATProfile(
        "gguf_q8_0", ops.q8_0_ste, ops.q8_0_ste, False, ("gguf_q8_0",), 32
    ),
    "mlx_q4": QATProfile(
        "mlx_q4",
        lambda x: ops.affine_groupwise_ste(x, bits=4),
        None,
        True,
        ("mlx_q4",),
        32,
    ),
    "mlx_q8": QATProfile(
        "mlx_q8",
        lambda x: ops.affine_groupwise_ste(x, bits=8),
        None,
        True,
        ("mlx_q8",),
        32,
    ),
    "vllm_fp8": QATProfile(
        "vllm_fp8",
        ops.vllm_fp8_weight_ste,
        ops.vllm_fp8_activation_ste,
        True,
        ("vllm_fp8",),
    ),
    "vllm_mxfp4": QATProfile(
        "vllm_mxfp4", ops.mxfp4_ste, ops.mxfp4_ste, True, ("vllm_mxfp4",), 32
    ),
    "noise_q4": QATProfile(
        "noise_q4",
        lambda x: ops.uniform_block_noise_ste(x, bits=4),
        lambda x: ops.uniform_block_noise_ste(x, bits=4),
        True,
        ("gguf_q4_0", "mlx_q4", "vllm_mxfp4"),
        32,
    ),
    "noise_q8": QATProfile(
        "noise_q8",
        lambda x: ops.uniform_block_noise_ste(x, bits=8),
        lambda x: ops.uniform_block_noise_ste(x, bits=8),
        True,
        ("gguf_q8_0", "mlx_q8", "vllm_fp8"),
        32,
    ),
}


def get_profile(name: str, target: str | None = None) -> QATProfile:
    try:
        profile = PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown QAT type {name!r}; expected one of {sorted(PROFILES)}"
        ) from exc
    if name != "vllm_fp8" or target is None:
        return profile
    resolved = ops.resolve_vllm_fp8_target(target)
    return QATProfile(
        name=profile.name,
        weight_quantizer=lambda value: ops.vllm_fp8_weight_ste(value, target=resolved),
        activation_quantizer=lambda value: ops.vllm_fp8_activation_ste(
            value, target=resolved
        ),
        quantize_projector=profile.quantize_projector,
        deployment_formats=profile.deployment_formats,
        weight_group_size=profile.weight_group_size,
    )
