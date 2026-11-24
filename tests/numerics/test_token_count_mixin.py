"""The Trainer's supervised-token count must match the targets the causal loss scores."""

import pytest
import torch
from transformers import Lfm2Config, Lfm2ForCausalLM, Trainer, TrainingArguments

from liquid_finetune.training.utils.trainer_mixins import CausalLMLossTokenCountMixin


class _Trainer(CausalLMLossTokenCountMixin, Trainer):
    pass


@pytest.fixture
def trainer(tmp_path):
    config = Lfm2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        max_position_embeddings=64,
    )
    args = TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True)
    return _Trainer(model=Lfm2ForCausalLM(config), args=args)


def test_count_matches_the_targets_the_causal_loss_scores(trainer):
    labels = torch.tensor([[-100, 5, 6, 7], [-100, -100, 9, 10]])
    # The loss predicts labels[..., 1:]: three targets in row 0, two in row 1.
    # Shifting twice would count labels[..., 2:] and give 4.
    count = trainer._get_num_items_in_batch([{"labels": labels}], torch.device("cpu"))
    assert int(count) == 5


def test_count_sums_across_accumulated_batches(trainer):
    batches = [
        {"labels": torch.tensor([[1, 2, 3]])},
        {"labels": torch.tensor([[4, -100, 6]])},
    ]
    assert int(trainer._get_num_items_in_batch(batches, torch.device("cpu"))) == 3
