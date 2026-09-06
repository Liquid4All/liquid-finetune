from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from liquid_finetune.training.moe_sft import LFMMoeSFTTrainer
from liquid_finetune.training.utils.trainer_mixins import RayDataLoaderMixin
from liquid_finetune.training.moe_utils import ep_runtime as moe_ep_module


class LengthDataset(Dataset):
    column_names = ["length"]

    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item != "length":
                raise KeyError(item)
            return self.lengths
        return {"length": self.lengths[item]}


def test_sender_expert_layout_round_trip():
    counts_by_sender = torch.tensor([[2, 1], [1, 2]], dtype=torch.long)
    tokens = torch.arange(6, dtype=torch.float32).unsqueeze(-1)

    expert_major = moe_ep_module.sender_major_to_expert_major(
        tokens,
        counts_by_sender,
    )
    restored = moe_ep_module.expert_major_to_sender_major(
        expert_major,
        counts_by_sender,
    )

    assert expert_major.squeeze(-1).tolist() == [0, 1, 3, 2, 4, 5]
    assert torch.equal(restored, tokens)


def test_compute_local_experts_uses_expert_major_grouped_mm_with_weights():
    class DummyExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(2, 8, 4))
            self.down_proj = nn.Parameter(torch.zeros(2, 4, 4))
            self.act_fn = torch.nn.functional.silu

    experts = DummyExperts()
    with torch.no_grad():
        eye = torch.eye(4)
        experts.gate_up_proj[:, :4, :] = eye
        experts.gate_up_proj[:, 4:, :] = eye
        experts.down_proj[:] = eye

    tokens = torch.randn(5, 4)
    counts_by_expert = torch.tensor([3, 2], dtype=torch.long)
    weights = torch.tensor([[1.0], [0.5], [2.0], [1.5], [0.25]])

    def fake_grouped_mm(input_, weight, offs):
        outputs = []
        start = 0
        for expert_idx, end_t in enumerate(offs.tolist()):
            end = int(end_t)
            outputs.append(input_[start:end] @ weight[expert_idx])
            start = end
        return torch.cat(outputs, dim=0)

    original_grouped_mm = moe_ep_module.grouped_mm
    moe_ep_module.grouped_mm = fake_grouped_mm
    try:
        output = moe_ep_module.compute_local_experts(
            experts,
            tokens,
            counts_by_expert,
            weights,
        )
    finally:
        moe_ep_module.grouped_mm = original_grouped_mm

    expected = torch.nn.functional.silu(tokens) * tokens * weights
    assert torch.allclose(output, expected)


def test_token_unpermutation_unsorts_before_single_reverse_a2a():
    dispatcher = moe_ep_module.EPTokenDispatcher(
        n_local_experts=2,
        local_expert_indices=[0, 1],
        n_experts=4,
        top_k=1,
        ep_size=2,
        ep_group=object(),
    )
    dispatcher.input_splits = [3, 3]
    dispatcher.output_splits = [3, 3]
    dispatcher.tokens_per_local_expert_by_sender = torch.tensor(
        [[2, 1], [1, 2]],
        dtype=torch.long,
    )
    dispatcher._reverse_map = torch.arange(6, dtype=torch.long)
    dispatcher._n_tokens = 6
    dispatcher._hidden_size = 1
    expert_output = torch.tensor([[0.0], [1.0], [3.0], [2.0], [4.0], [5.0]])

    a2a_inputs = []
    original_a2a = moe_ep_module.all_to_all_single_autograd

    def fake_a2a(tensor, output_split_sizes, input_split_sizes, group):
        a2a_inputs.append(tensor.clone())
        return tensor

    moe_ep_module.all_to_all_single_autograd = fake_a2a
    try:
        output = dispatcher.token_unpermutation(expert_output)
    finally:
        moe_ep_module.all_to_all_single_autograd = original_a2a

    assert torch.equal(a2a_inputs[0].squeeze(-1), torch.arange(6, dtype=torch.float32))
    assert torch.equal(output.squeeze(-1), torch.arange(6, dtype=torch.float32))


def test_token_unpermutation_uses_single_reverse_a2a():
    dispatcher = moe_ep_module.EPTokenDispatcher(
        n_local_experts=1,
        local_expert_indices=[0],
        n_experts=1,
        top_k=1,
        ep_size=2,
        ep_group=object(),
    )
    dispatcher.input_splits = [8, 0]
    dispatcher.output_splits = [4, 4]
    dispatcher._reverse_map = torch.arange(8, dtype=torch.long)
    dispatcher._n_tokens = 8
    dispatcher._hidden_size = 2

    expert_output = torch.zeros(8, 2, dtype=torch.float32)
    calls = []

    def fake_a2a(tensor, output_split_sizes, input_split_sizes, group):
        calls.append((list(output_split_sizes), list(input_split_sizes)))
        return torch.zeros(sum(output_split_sizes), 2, dtype=torch.float32)

    original_a2a = moe_ep_module.all_to_all_single_autograd
    moe_ep_module.all_to_all_single_autograd = fake_a2a
    try:
        dispatcher.token_unpermutation(expert_output)
    finally:
        moe_ep_module.all_to_all_single_autograd = original_a2a

    assert calls == [([8, 0], [4, 4])]


def test_token_permutation_exchanges_weights_and_sorts_to_expert_major():
    dispatcher = moe_ep_module.EPTokenDispatcher(
        n_local_experts=2,
        local_expert_indices=[0, 1],
        n_experts=4,
        top_k=1,
        ep_size=2,
        ep_group=object(),
    )
    dispatcher.input_splits = [3, 3]
    dispatcher.output_splits = [3, 3]
    dispatcher.tokens_per_local_expert_by_sender = torch.tensor(
        [[2, 1], [1, 2]],
        dtype=torch.long,
    )

    permuted = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    weights = torch.arange(10, 16, dtype=torch.float32).unsqueeze(-1)
    reverse_map = torch.arange(6, dtype=torch.long)
    calls = []

    def fake_permute_tokens(
        hidden_states, selected_experts, routing_weights, n_experts
    ):
        return permuted, weights, reverse_map, torch.tensor([3, 3])

    def fake_a2a(tensor, output_split_sizes, input_split_sizes, group):
        calls.append((list(output_split_sizes), list(input_split_sizes)))
        return tensor

    original_permute = moe_ep_module.permute_tokens
    original_a2a = moe_ep_module.all_to_all_single_autograd
    moe_ep_module.permute_tokens = fake_permute_tokens
    moe_ep_module.all_to_all_single_autograd = fake_a2a
    try:
        output, output_weights = dispatcher.token_permutation(
            torch.zeros(6, 1),
            torch.zeros(6, 1, dtype=torch.long),
            torch.ones(6, 1),
        )
    finally:
        moe_ep_module.permute_tokens = original_permute
        moe_ep_module.all_to_all_single_autograd = original_a2a

    assert calls == [([3, 3], [3, 3]), ([3, 3], [3, 3])]
    assert output.squeeze(-1).tolist() == [0, 1, 3, 2, 4, 5]
    assert output_weights.squeeze(-1).tolist() == [10, 11, 13, 12, 14, 15]


def test_moe_sft_length_grouping_uses_dp_seed_for_ep():
    trainer = LFMMoeSFTTrainer.__new__(LFMMoeSFTTrainer)
    trainer.train_dataset = LengthDataset([8, 3, 5, 2])
    trainer._train_batch_size = 2
    trainer.data_collator = lambda features: features
    trainer.ep_config = {"dp_rank": 3}

    dataloader = trainer.get_train_dataloader()

    assert dataloader.sampler is not None
    assert dataloader.sampler.generator is not None
    assert dataloader.sampler.generator.initial_seed() == 45


def test_moe_sft_length_grouping_can_be_disabled():
    trainer = LFMMoeSFTTrainer.__new__(LFMMoeSFTTrainer)
    trainer.train_dataset = LengthDataset([8, 3, 5, 2])
    trainer._train_batch_size = 2
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=2,
        group_by_length=False,
        seed=7,
    )
    trainer.data_collator = lambda features: features
    trainer.ep_config = None

    dataloader = trainer.get_train_dataloader()

    assert type(dataloader.sampler).__name__ == "RandomSampler"


def test_ray_dataloader_length_grouping_is_toggleable():
    trainer = RayDataLoaderMixin()
    trainer.train_dataset = LengthDataset([8, 3, 5, 2])
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=2,
        group_by_length=True,
        seed=17,
    )
    trainer.data_collator = lambda features: features

    dataloader = trainer.get_train_dataloader()

    assert dataloader.sampler is not None
    assert dataloader.sampler.generator.initial_seed() == 17


def test_ray_dataloader_honors_hf_worker_controls():
    trainer = RayDataLoaderMixin()
    trainer.train_dataset = LengthDataset([8, 3, 5, 2])
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=2,
        group_by_length=False,
        seed=17,
        dataloader_num_workers=2,
        dataloader_prefetch_factor=3,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
    )
    trainer.data_collator = lambda features: features

    dataloader = trainer.get_train_dataloader()

    assert dataloader.num_workers == 2
    assert dataloader.prefetch_factor == 3
    assert dataloader.persistent_workers is True
    assert dataloader.pin_memory is True
    assert dataloader.drop_last is True


def test_ray_dataloader_omits_worker_only_options_without_workers():
    trainer = RayDataLoaderMixin()
    trainer.train_dataset = LengthDataset([8, 3, 5, 2])
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=2,
        group_by_length=False,
        seed=17,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=3,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=False,
    )
    trainer.data_collator = lambda features: features

    dataloader = trainer.get_train_dataloader()

    assert dataloader.num_workers == 0
