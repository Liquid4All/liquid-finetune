import os

import torch
import torch.nn as nn


def grouped_mm(
    input_: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Dispatch to PyTorch grouped_mm with the dtype/layout it expects."""
    input_ = input_.to(weight.dtype)
    offs = offs.to(device=input_.device, dtype=torch.int32).contiguous()
    if hasattr(torch.nn.functional, "grouped_mm"):
        return torch.nn.functional.grouped_mm(input_, weight, offs=offs)
    return torch._grouped_mm(input_, weight, offs=offs)


def tokens_per_expert_from_routing(
    selected_experts: torch.Tensor, n_experts: int
) -> torch.Tensor:
    flat = selected_experts.reshape(-1)
    return torch.histc(flat.float(), bins=n_experts, min=0, max=n_experts - 1).long()


def permute_tokens(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    n_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_tokens, top_k = selected_experts.shape
    device = hidden_states.device

    token_idx = (
        torch.arange(n_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    )
    expert_ids = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)

    perm = torch.argsort(expert_ids, stable=True)
    token_idx = token_idx[perm]
    flat_weights = flat_weights[perm]

    permuted_tokens = hidden_states[token_idx]
    permuted_weights = flat_weights.unsqueeze(-1)
    tokens_per_expert = tokens_per_expert_from_routing(selected_experts, n_experts)

    return permuted_tokens, permuted_weights, token_idx, tokens_per_expert


def unpermute_tokens(
    hidden_states: torch.Tensor,
    reverse_map: torch.Tensor,
    probs: torch.Tensor | None,
    n_tokens: int,
    hidden_size: int,
) -> torch.Tensor:
    output = torch.zeros(
        n_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )

    chunk_rows = int(os.getenv("LIQUID_MOE_UNPERMUTE_CHUNK_ROWS", "4096"))
    probs = probs.to(hidden_states.dtype) if probs is not None else None

    for start in range(0, hidden_states.shape[0], chunk_rows):
        end = min(start + chunk_rows, hidden_states.shape[0])
        chunk = hidden_states[start:end]
        if probs is not None:
            chunk = chunk * probs[start:end]
        output.scatter_add_(
            0,
            reverse_map[start:end].unsqueeze(-1).expand_as(chunk),
            chunk,
        )

    return output


def route_tokens_to_experts(
    block: nn.Module, router_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score_function = getattr(block, "score_function", "sigmoid")
    if score_function != "sigmoid":
        raise ValueError(f"Unsupported router score function: {score_function}")

    router_probs = torch.sigmoid(router_logits)

    if hasattr(block, "route_tokens_to_experts"):
        selected_experts, routing_weights = block.route_tokens_to_experts(router_logits)
        return selected_experts, routing_weights, router_probs

    top_k = getattr(block, "top_k")
    expert_bias = getattr(block, "expert_bias", None)
    if expert_bias is not None:
        scores_for_routing = router_probs + expert_bias
        _, selected_experts = torch.topk(scores_for_routing, k=top_k, dim=-1)
        routing_weights = torch.gather(router_probs, dim=1, index=selected_experts)
        routing_weights = routing_weights.type_as(router_logits)
    else:
        routing_weights, selected_experts = torch.topk(router_probs, k=top_k, dim=-1)

    if getattr(block, "norm_topk_prob", False):
        routing_weights = routing_weights / (
            routing_weights.sum(dim=-1, keepdim=True) + 1e-20
        )

    scaling_factor = getattr(block, "scaling_factor", None)
    if scaling_factor:
        routing_weights = routing_weights * scaling_factor

    return selected_experts, routing_weights.to(router_logits.dtype), router_probs
