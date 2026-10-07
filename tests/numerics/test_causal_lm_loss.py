import torch
import torch.nn as nn

import liquid_finetune.training.utils.causal_lm_loss as causal_lm_loss_module
from liquid_finetune.training.utils.causal_lm_loss import (
    install_memory_efficient_causal_lm_loss,
)

VOCAB_SIZE = 16
HIDDEN_SIZE = 8
IGNORE_INDEX = -100


class _FakeBaseModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _FakeBaseModel(nn.Module):
    """Stand-in for the transformers base model returning hidden states."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> _FakeBaseModelOutput:
        del kwargs
        return _FakeBaseModelOutput(self.embed(input_ids))


class _FakeCausalLM(nn.Module):
    """Mimics a transformers causal LM: forwards extra kwargs (including
    ``num_items_in_batch``) to ``self.loss_function``, as stock model
    forwards do."""

    def __init__(self):
        super().__init__()
        self.model = _FakeBaseModel()
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> dict:
        del logits_to_keep
        outputs = self.model(input_ids=input_ids)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=VOCAB_SIZE, **kwargs
            )
        return {"loss": loss, "logits": logits}


def _make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 6))
    labels = input_ids.clone()
    labels[0, :3] = IGNORE_INDEX
    return input_ids, labels


def _reference_sum_ce(model: _FakeCausalLM, input_ids, labels) -> torch.Tensor:
    """Summed cross-entropy over valid (shifted) label positions."""
    logits = model.lm_head(model.model(input_ids=input_ids).last_hidden_state)
    shift_labels = nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    return nn.functional.cross_entropy(
        logits.view(-1, VOCAB_SIZE).float(),
        shift_labels.reshape(-1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def _local_valid_tokens(labels: torch.Tensor) -> torch.Tensor:
    shift_labels = nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    return (shift_labels != IGNORE_INDEX).sum()


def test_eval_forward_normalizes_by_global_token_count():
    """Eval-mode forward must divide by the trainer-provided global token
    count (gathered across processes), not the local per-rank mean.
    Otherwise the trainer's ``loss *= num_processes`` correction inflates
    eval_loss by the world size."""
    model = _FakeCausalLM()
    install_memory_efficient_causal_lm_loss(model)
    model.eval()

    input_ids, labels = _make_batch()
    local_items = _local_valid_tokens(labels)

    # Simulate a 4-process run: the gathered global count is 4x the local one.
    world_size = 4
    num_items_in_batch = local_items * world_size

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            labels=labels,
            num_items_in_batch=num_items_in_batch,
        )

    expected = _reference_sum_ce(model, input_ids, labels) / num_items_in_batch
    torch.testing.assert_close(output["loss"], expected)


def test_eval_forward_without_num_items_uses_local_mean():
    model = _FakeCausalLM()
    install_memory_efficient_causal_lm_loss(model)
    model.eval()

    input_ids, labels = _make_batch()
    local_items = _local_valid_tokens(labels)

    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels)

    expected = _reference_sum_ce(model, input_ids, labels) / local_items
    torch.testing.assert_close(output["loss"], expected)


def test_train_forward_normalizes_by_global_token_count(monkeypatch):
    monkeypatch.setattr(causal_lm_loss_module, "_LOSS_BACKEND", "eager")
    model = _FakeCausalLM()
    install_memory_efficient_causal_lm_loss(model)
    model.train()

    input_ids, labels = _make_batch()
    local_items = _local_valid_tokens(labels)

    world_size = 4
    num_items_in_batch = local_items * world_size

    output = model(
        input_ids=input_ids,
        labels=labels,
        num_items_in_batch=num_items_in_batch,
    )

    expected = _reference_sum_ce(model, input_ids, labels) / num_items_in_batch
    torch.testing.assert_close(output["loss"], expected)


def test_train_and_eval_forward_agree(monkeypatch):
    """Same inputs and same global token count must yield the same loss in
    training and eval mode."""
    monkeypatch.setattr(causal_lm_loss_module, "_LOSS_BACKEND", "eager")
    model = _FakeCausalLM()
    install_memory_efficient_causal_lm_loss(model)

    input_ids, labels = _make_batch()
    num_items_in_batch = _local_valid_tokens(labels) * 8

    model.train()
    train_loss = model(
        input_ids=input_ids,
        labels=labels,
        num_items_in_batch=num_items_in_batch,
    )["loss"]

    model.eval()
    with torch.no_grad():
        eval_loss = model(
            input_ids=input_ids,
            labels=labels,
            num_items_in_batch=num_items_in_batch,
        )["loss"]

    torch.testing.assert_close(eval_loss, train_loss.detach())
