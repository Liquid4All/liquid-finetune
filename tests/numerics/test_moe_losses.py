import torch
import torch.nn as nn

from liquid_finetune.training.moe_utils.losses import (
    MoETrainingConfig,
    apply_moe_losses,
    switch_load_balancing_loss,
)


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        self.top_k = 1
        self.score_function = "sigmoid"
        self.norm_topk_prob = False
        self.experts = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)])


class DummyContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = Lfm2MoeSparseMoeBlock()


class TrackingExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.called = False
        self.last_hidden_shape = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        self.called = True
        self.last_hidden_shape = tuple(hidden_states.shape)
        return hidden_states + routing_weights.sum(dim=-1, keepdim=True).to(
            hidden_states.dtype
        )


def test_switch_load_balancing_loss_matches_formula():
    router_probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    selected_experts = torch.tensor([[0], [1]])

    loss = switch_load_balancing_loss(
        router_probs=router_probs,
        selected_experts=selected_experts,
        num_experts=2,
        top_k=1,
        coeff=0.5,
    )

    aggregated_probs = router_probs.mean(dim=0)
    tokens_per_expert = torch.tensor([1.0, 1.0])
    expected = (aggregated_probs * tokens_per_expert).sum() * 2 * 0.5 / 2
    assert torch.allclose(loss, expected)


def test_apply_moe_losses_handles_blocks_without_num_experts_attr():
    torch.manual_seed(0)
    model = DummyContainer()

    apply_moe_losses(model, MoETrainingConfig(aux_loss_coef=0.01, z_loss_coef=0.001))

    hidden_states = torch.randn(2, 3, 4)
    output = model.block(hidden_states)

    assert output.shape == hidden_states.shape
    assert hasattr(model.block, "_moe_aux_loss")
    assert hasattr(model.block, "_moe_z_loss")
    assert model.block._moe_num_experts == 2


def test_apply_moe_losses_delegates_non_ep_expert_compute_to_model_experts():
    torch.manual_seed(0)
    model = DummyContainer()
    model.block.experts = TrackingExperts()

    apply_moe_losses(model, MoETrainingConfig(aux_loss_coef=0.01, z_loss_coef=0.001))

    hidden_states = torch.randn(2, 3, 4)
    output = model.block(hidden_states)

    assert output.shape == hidden_states.shape
    assert model.block.experts.called is True
    assert model.block.experts.last_hidden_shape == (6, 4)
