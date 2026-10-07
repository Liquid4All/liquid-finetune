from __future__ import annotations

import math

import torch

BLOCK_SIZE = 32


def _block_view(tensor: torch.Tensor, block_size: int = BLOCK_SIZE):
    """Return a float block view, padding only the final dimension if needed."""
    if tensor.ndim == 0:
        raise ValueError("QAT requires a tensor with at least one dimension")
    original_shape = tensor.shape
    width = original_shape[-1]
    padded_width = math.ceil(width / block_size) * block_size
    value = tensor.float()
    if width == 0:
        raise ValueError("QAT cannot quantize an empty final dimension")
    if padded_width != width:
        value = torch.nn.functional.pad(value, (0, padded_width - width))
    return value.reshape(-1, block_size), original_shape, padded_width


def _restore_blocks(blocks, original, original_shape, padded_width):
    restored = blocks.reshape(*original_shape[:-1], padded_width)
    return restored[..., : original_shape[-1]].to(original.dtype)


def _ste(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    return original + (quantized - original).detach()


def q4_0(tensor: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Simulate native llama.cpp Q4_0 quantization and dequantization."""
    blocks, shape, padded_width = _block_view(tensor, block_size)
    max_indices = blocks.abs().argmax(dim=1, keepdim=True)
    signed_max = blocks.gather(1, max_indices)
    # ggml chooses the integer codes with the FP32 scale, then stores that
    # scale as FP16. Dequantization therefore uses the rounded stored scale.
    scale = signed_max / -8.0
    inverse = torch.where(scale != 0, scale.reciprocal(), torch.zeros_like(scale))
    quants = (blocks * inverse + 8.5).to(torch.int32).clamp_(0, 15)
    stored_scale = scale.half().float()
    return _restore_blocks(
        (quants.float() - 8.0) * stored_scale, tensor, shape, padded_width
    )


def q4_0_ste(tensor: torch.Tensor) -> torch.Tensor:
    return _ste(tensor, q4_0(tensor))


def q8_0(tensor: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Simulate native llama.cpp Q8_0 quantization and dequantization."""
    blocks, shape, padded_width = _block_view(tensor, block_size)
    scale = blocks.abs().amax(dim=1, keepdim=True) / 127.0
    inverse = torch.where(scale != 0, scale.reciprocal(), torch.zeros_like(scale))
    scaled = blocks * inverse
    # ggml's roundf rounds exact half-way values away from zero; torch.round
    # uses ties-to-even.
    quants = scaled.abs().add(0.5).floor().copysign(scaled).clamp_(-128, 127)
    stored_scale = scale.half().float()
    return _restore_blocks(quants * stored_scale, tensor, shape, padded_width)


def q8_0_ste(tensor: torch.Tensor) -> torch.Tensor:
    return _ste(tensor, q8_0(tensor))


def affine_groupwise(
    tensor: torch.Tensor, *, bits: int, group_size: int = BLOCK_SIZE
) -> torch.Tensor:
    """MLX-compatible affine group quantization along the final dimension."""
    if bits not in (4, 8):
        raise ValueError(f"Affine QAT supports 4 or 8 bits, got {bits}")
    blocks, shape, padded_width = _block_view(tensor, group_size)
    qmax = (1 << bits) - 1
    minimum = blocks.amin(dim=1, keepdim=True)
    maximum = blocks.amax(dim=1, keepdim=True)
    scale = (maximum - minimum) / qmax
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    quants = ((blocks - minimum) / safe_scale).round().clamp_(0, qmax)
    dequantized = torch.where(scale != 0, quants * scale + minimum, minimum)
    return _restore_blocks(dequantized, tensor, shape, padded_width)


def affine_groupwise_ste(tensor: torch.Tensor, *, bits: int) -> torch.Tensor:
    return _ste(tensor, affine_groupwise(tensor, bits=bits))


def _float8_dtype(*, fnuz: bool = False):
    dtype_name = "float8_e4m3fnuz" if fnuz else "float8_e4m3fn"
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise RuntimeError(f"vLLM FP8 QAT requires torch.{dtype_name} support")
    return dtype


def fp8_e4m3_per_tensor(
    tensor: torch.Tensor, *, fnuz: bool = False, per_expert_matrix: bool = False
) -> torch.Tensor:
    value = tensor.float()
    fp8_dtype = _float8_dtype(fnuz=fnuz)
    fp8_max = torch.finfo(fp8_dtype).max
    reduce_dims = (-2, -1) if per_expert_matrix and value.ndim >= 3 else None
    scale = value.abs().amax(dim=reduce_dims, keepdim=reduce_dims is not None) / fp8_max
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    quantized = (value / safe_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    dequantized = quantized.float() * safe_scale
    return torch.where(scale != 0, dequantized, value).to(tensor.dtype)


def fp8_e4m3_per_tensor_ste(tensor: torch.Tensor) -> torch.Tensor:
    return _ste(tensor, fp8_e4m3_per_tensor(tensor))


def fp8_e4m3_per_token(tensor: torch.Tensor, *, fnuz: bool = False) -> torch.Tensor:
    value = tensor.float()
    fp8_dtype = _float8_dtype(fnuz=fnuz)
    fp8_max = torch.finfo(fp8_dtype).max
    scale = value.abs().amax(dim=-1, keepdim=True) / fp8_max
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    quantized = (value / safe_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    dequantized = quantized.float() * safe_scale
    return torch.where(scale != 0, dequantized, value).to(tensor.dtype)


def fp8_e4m3_per_token_ste(tensor: torch.Tensor) -> torch.Tensor:
    return _ste(tensor, fp8_e4m3_per_token(tensor))


def resolve_vllm_fp8_target(
    target: str | None = None, tensor: torch.Tensor | None = None
) -> str:
    """Resolve and validate the vLLM FP8 deployment contract."""
    requested = target or "auto"
    if requested in {"cuda", "rocm_mi300"}:
        return requested
    if requested != "auto":
        raise ValueError(
            f"Unknown vLLM FP8 target {requested!r}; expected auto, cuda, or rocm_mi300"
        )
    if torch.version.hip is None:
        return "cuda"
    if tensor is not None and tensor.device.type == "cuda":
        try:
            properties = torch.cuda.get_device_properties(tensor.device)
        except (AssertionError, RuntimeError):
            properties = None
        if properties is not None:
            arch = getattr(properties, "gcnArchName", "")
            if "gfx94" not in arch:
                raise ValueError(
                    f"Auto vLLM FP8 target does not support ROCm architecture {arch!r}; "
                    "set an explicitly supported target"
                )
    # The currently verified ROCm online-FP8 target is MI300/gfx94x. Explicit
    # metadata makes this assumption visible even if preparation happens on CPU.
    return "rocm_mi300"


def vllm_fp8_weight_ste(
    tensor: torch.Tensor, *, target: str | None = None
) -> torch.Tensor:
    """Match vLLM online FP8 weight quantization on the active platform."""
    resolved = resolve_vllm_fp8_target(target, tensor)
    return _ste(
        tensor,
        fp8_e4m3_per_tensor(
            tensor,
            fnuz=resolved == "rocm_mi300",
            per_expert_matrix=True,
        ),
    )


def vllm_fp8_activation_ste(
    tensor: torch.Tensor, *, target: str | None = None
) -> torch.Tensor:
    """Match vLLM online FP8 activation granularity on CUDA and ROCm."""
    resolved = resolve_vllm_fp8_target(target, tensor)
    if resolved == "rocm_mi300":
        quantized = fp8_e4m3_per_tensor(tensor, fnuz=True)
    else:
        quantized = fp8_e4m3_per_token(tensor)
    return _ste(tensor, quantized)


_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mxfp4(tensor: torch.Tensor, group_size: int = BLOCK_SIZE) -> torch.Tensor:
    """MXFP4 E2M1 values with an E8M0 power-of-two scale per group."""
    blocks, shape, padded_width = _block_view(tensor, group_size)
    amax = blocks.abs().amax(dim=1, keepdim=True)
    tiny = torch.tensor(torch.finfo(torch.float32).tiny, device=blocks.device)
    # vLLM's native MXFP4 path rounds amax / 6 upward to the next E8M0
    # power-of-two scale.
    exponent = torch.ceil(torch.log2(torch.clamp(amax / 6.0, min=tiny)))
    scale = torch.pow(2.0, exponent.clamp_(-127, 127))
    scale = torch.where(amax != 0, scale, torch.ones_like(scale))
    normalized = blocks / scale
    codebook = torch.tensor(_E2M1_POSITIVE, device=blocks.device, dtype=blocks.dtype)
    distances = (normalized.abs().unsqueeze(-1) - codebook).abs()
    # Native E2M1 conversion resolves exact ties upward.
    reverse_index = distances.shape[-1] - 1 - distances.flip(-1).argmin(dim=-1)
    magnitude = codebook[reverse_index]
    dequantized = magnitude.copysign(normalized) * scale
    dequantized = torch.where(amax != 0, dequantized, blocks)
    return _restore_blocks(dequantized, tensor, shape, padded_width)


def mxfp4_ste(tensor: torch.Tensor) -> torch.Tensor:
    return _ste(tensor, mxfp4(tensor))


def mxfp8(tensor: torch.Tensor, group_size: int = BLOCK_SIZE) -> torch.Tensor:
    """MXFP8 E4M3 values with an E8M0 power-of-two scale per group."""
    blocks, shape, padded_width = _block_view(tensor, group_size)
    amax = blocks.abs().amax(dim=1, keepdim=True)
    tiny = torch.tensor(torch.finfo(torch.float32).tiny, device=blocks.device)
    # E8M0 scales round upward so the block amax fits in E4M3FN (max 448).
    exponent = torch.ceil(torch.log2(torch.clamp(amax / 448.0, min=tiny)))
    scale = torch.pow(2.0, exponent.clamp_(-127, 127))
    scale = torch.where(amax != 0, scale, torch.ones_like(scale))
    normalized = (blocks / scale).clamp(-448.0, 448.0)
    quantized = normalized.to(_float8_dtype()).float() * scale
    quantized = torch.where(amax != 0, quantized, blocks)
    return _restore_blocks(quantized, tensor, shape, padded_width)


def mxfp8_ste(tensor: torch.Tensor) -> torch.Tensor:
    return _ste(tensor, mxfp8(tensor))


def _e2m1_round_to_even(value: torch.Tensor) -> torch.Tensor:
    """Round magnitudes to E2M1 using vLLM NVFP4's tie-to-even contract."""
    magnitude = value.abs()
    rounded = torch.zeros_like(magnitude)
    rounded = torch.where((magnitude > 0.25) & (magnitude < 0.75), 0.5, rounded)
    rounded = torch.where((magnitude >= 0.75) & (magnitude <= 1.25), 1.0, rounded)
    rounded = torch.where((magnitude > 1.25) & (magnitude < 1.75), 1.5, rounded)
    rounded = torch.where((magnitude >= 1.75) & (magnitude <= 2.5), 2.0, rounded)
    rounded = torch.where((magnitude > 2.5) & (magnitude < 3.5), 3.0, rounded)
    rounded = torch.where((magnitude >= 3.5) & (magnitude <= 5.0), 4.0, rounded)
    rounded = torch.where(magnitude > 5.0, 6.0, rounded)
    return rounded.copysign(value)


def nvfp4(
    tensor: torch.Tensor,
    group_size: int = 16,
    *,
    per_expert_matrix: bool = False,
) -> torch.Tensor:
    """NVFP4 E2M1 with E4M3 group-16 and FP32 tensor-level scaling.

    This is deterministic inference-format fake quantization. Transformer
    Engine training may additionally apply stochastic rounding and an RHT.
    """
    blocks, shape, padded_width = _block_view(tensor, group_size)
    value = tensor.float()
    reduce_dims = (-2, -1) if per_expert_matrix and value.ndim >= 3 else None
    tensor_amax = value.abs().amax(dim=reduce_dims, keepdim=reduce_dims is not None)
    global_multiplier = torch.where(
        tensor_amax != 0, 2688.0 / tensor_amax, torch.ones_like(tensor_amax)
    )
    if reduce_dims is None:
        block_multiplier = global_multiplier
    else:
        rows_per_expert = math.prod(shape[1:-1]) * (padded_width // group_size)
        block_multiplier = global_multiplier.reshape(-1, 1).repeat_interleave(
            rows_per_expert, dim=0
        )

    block_amax = blocks.abs().amax(dim=1, keepdim=True)
    encoded_scale = (block_multiplier * block_amax / 6.0).clamp(0.0, 448.0)
    encoded_scale = encoded_scale.to(_float8_dtype()).float()
    inverse_global = 1.0 / (
        block_multiplier + (block_multiplier == 0).to(block_multiplier.dtype) * 1e8
    )
    divisor = encoded_scale * inverse_global
    output_scale = 1.0 / (divisor + (divisor == 0).to(divisor.dtype) * 1e8)
    normalized = (blocks * output_scale).clamp(-6.0, 6.0)
    dequant_scale = encoded_scale / block_multiplier
    dequantized = _e2m1_round_to_even(normalized) * dequant_scale
    dequantized = torch.where(block_amax != 0, dequantized, blocks)
    return _restore_blocks(dequantized, tensor, shape, padded_width)


def nvfp4_weight_ste(tensor: torch.Tensor) -> torch.Tensor:
    return _ste(tensor, nvfp4(tensor, per_expert_matrix=True))


def nvfp4_activation_ste(tensor: torch.Tensor) -> torch.Tensor:
    return _ste(tensor, nvfp4(tensor))


def uniform_block_noise(
    tensor: torch.Tensor, *, bits: int, block_size: int = BLOCK_SIZE
) -> torch.Tensor:
    """Add zero-mean uniform noise with the local symmetric quantization step."""
    if bits not in (4, 8):
        raise ValueError(f"Noise QAT supports 4 or 8 bits, got {bits}")
    blocks, shape, padded_width = _block_view(tensor, block_size)
    qmax = (1 << (bits - 1)) - 1
    step = blocks.abs().amax(dim=1, keepdim=True) / qmax
    noisy = blocks + (torch.rand_like(blocks) - 0.5) * step
    return _restore_blocks(noisy, tensor, shape, padded_width)


def uniform_block_noise_ste(tensor: torch.Tensor, *, bits: int) -> torch.Tensor:
    return _ste(tensor, uniform_block_noise(tensor, bits=bits))
