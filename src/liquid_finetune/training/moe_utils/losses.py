import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from liquid_finetune.training.moe_utils.ops import route_tokens_to_experts

logger = logging.getLogger(__name__)


@dataclass
class MoETrainingConfig:
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 0.001
    capacity_factor: float | None = None
    token_drop_policy: str = "probs"

    @classmethod
    def from_dict(cls, d: dict) -> "MoETrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class InjectAuxLoss(torch.autograd.Function):
    """Injects auxiliary loss into backward pass without affecting forward output."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (aux_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(aux_loss)


def switch_load_balancing_loss(
    router_probs: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    top_k: int,
    coeff: float,
) -> torch.Tensor:
    n_tokens = router_probs.shape[0]
    if n_tokens == 0:
        return router_probs.sum() * 0.0

    tokens_per_expert = torch.zeros(
        num_experts, device=router_probs.device, dtype=torch.float32
    )
    ones = torch.ones_like(selected_experts, dtype=torch.float32)
    tokens_per_expert.scatter_add_(0, selected_experts.reshape(-1), ones.reshape(-1))

    aggregated_probs = router_probs.float().sum(dim=0)
    return (
        torch.sum(aggregated_probs * tokens_per_expert)
        * num_experts
        * coeff
        / (n_tokens * n_tokens * top_k)
    )


def z_loss(router_logits: torch.Tensor, coeff: float) -> torch.Tensor:
    return coeff * router_logits.logsumexp(dim=-1).pow(2).mean()


def apply_router_token_dropping(
    router_probs: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    capacity_factor: float,
    drop_policy: str = "probs",
) -> tuple[torch.Tensor, torch.Tensor]:
    n_tokens, top_k = selected_experts.shape
    expert_capacity = int(capacity_factor * n_tokens / num_experts)

    new_routing_weights = routing_weights.clone()

    for k in range(top_k):
        expert_ids = selected_experts[:, k]

        for expert_idx in range(num_experts):
            mask = expert_ids == expert_idx
            token_indices = mask.nonzero(as_tuple=True)[0]

            if token_indices.numel() <= expert_capacity:
                continue

            if drop_policy == "probs":
                probs_for_expert = router_probs[token_indices, expert_idx]
                _, sorted_idx = probs_for_expert.sort(descending=True)
                drop_indices = token_indices[sorted_idx[expert_capacity:]]
            else:
                drop_indices = token_indices[expert_capacity:]

            new_routing_weights[drop_indices, k] = 0.0

    return selected_experts, new_routing_weights


def apply_moe_losses(model: nn.Module, config: MoETrainingConfig) -> None:
    """Patch all MoE blocks to inject aux losses into the forward pass."""
    patched = 0
    for module in model.modules():
        if type(module).__name__ == "Lfm2MoeSparseMoeBlock":
            _patch_block(module, config)
            patched += 1
    logger.info("Applied MoE aux losses to %s blocks", patched)


def store_moe_metrics(
    block: nn.Module,
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
) -> None:
    top_k = selected_experts.shape[-1]
    with torch.no_grad():
        tokens_per_expert = torch.zeros(
            num_experts, device=router_logits.device, dtype=torch.float32
        )
        ones = torch.ones_like(selected_experts[:, 0], dtype=torch.float32)
        for k in range(top_k):
            tokens_per_expert.scatter_add_(0, selected_experts[:, k], ones)

        block._moe_tokens_per_expert = tokens_per_expert
        block._moe_router_logits_mean = router_logits.mean().detach()
        block._moe_router_logits_std = router_logits.std().detach()
        block._moe_num_experts = num_experts


def _run_module_list_experts(
    block: nn.Module,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat_hidden_states = hidden_states.view(-1, hidden_dim)
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=torch.float32,
        device=flat_hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)

    expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(
        as_tuple=False
    )
    for expert_idx in expert_hitted.flatten().tolist():
        expert_layer = block.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = flat_hidden_states[top_x]
        current_hidden_states = expert_layer(current_state) * routing_weights[
            top_x, idx, None
        ].to(current_state.dtype)
        final_hidden_states.index_add_(
            0, top_x, current_hidden_states.to(torch.float32)
        )

    return final_hidden_states.to(flat_hidden_states.dtype).reshape(
        batch_size, sequence_length, hidden_dim
    )


def _patch_block(block: nn.Module, config: MoETrainingConfig) -> None:
    def enhanced_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_states.shape[-1])

        router_logits = block.gate(x)
        num_experts = router_logits.shape[-1]

        if config.z_loss_coef > 0:
            zl = z_loss(router_logits, config.z_loss_coef)
            router_logits = InjectAuxLoss.apply(router_logits, zl)
            block._moe_z_loss = zl.detach()

        selected_experts, routing_weights, router_probs = route_tokens_to_experts(
            block, router_logits
        )
        top_k = selected_experts.shape[-1]

        if config.capacity_factor is not None:
            selected_experts, routing_weights = apply_router_token_dropping(
                router_probs=router_probs,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
                num_experts=num_experts,
                capacity_factor=config.capacity_factor,
                drop_policy=config.token_drop_policy,
            )

        if config.aux_loss_coef > 0:
            aux = switch_load_balancing_loss(
                router_probs,
                selected_experts,
                num_experts,
                top_k,
                config.aux_loss_coef,
            )
            routing_weights = InjectAuxLoss.apply(routing_weights, aux)
            block._moe_aux_loss = aux.detach()

        if (
            hasattr(block.experts, "forward")
            and type(block.experts).__name__ != "ModuleList"
        ):
            output = block.experts(
                x,
                selected_experts,
                routing_weights,
            )
            output = output.view(batch_size, sequence_length, hidden_dim)
        else:
            output = _run_module_list_experts(
                block, hidden_states, selected_experts, routing_weights, num_experts
            )

        store_moe_metrics(block, selected_experts, router_logits, num_experts)
        return output

    block.forward = enhanced_forward
