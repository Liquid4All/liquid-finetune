from types import SimpleNamespace

import torch

from liquid_finetune.training.utils.vlm_parameter_participation import (
    VLMParameterParticipationMixin,
)


class _BaseTrainer:
    def compute_loss(self, model, inputs, *args, **kwargs):
        return model.language_model(inputs["input_ids"]).sum()


class _Trainer(VLMParameterParticipationMixin, _BaseTrainer):
    pass


def test_text_only_vlm_loss_participates_with_zero_media_gradients():
    model = torch.nn.Module()
    model.language_model = torch.nn.Linear(2, 2)
    model.vision_tower = torch.nn.Linear(2, 2)
    model.multi_modal_projector = torch.nn.Linear(2, 2)

    loss = _Trainer().compute_loss(
        model,
        {"input_ids": torch.ones(1, 2)},
    )
    loss.backward()

    assert model.language_model.weight.grad.abs().sum() > 0
    assert model.vision_tower.weight.grad is not None
    assert model.vision_tower.weight.grad.count_nonzero() == 0
    assert model.multi_modal_projector.weight.grad is not None
    assert model.multi_modal_projector.weight.grad.count_nonzero() == 0


def test_loss_output_tuple_is_preserved():
    class _TupleBaseTrainer:
        def compute_loss(self, model, inputs, *args, **kwargs):
            return torch.tensor(1.0), SimpleNamespace(logits=None)

    class _TupleTrainer(VLMParameterParticipationMixin, _TupleBaseTrainer):
        pass

    model = torch.nn.Module()
    model.vision_tower = torch.nn.Linear(1, 1)
    loss, outputs = _TupleTrainer().compute_loss(model, {})
    assert loss.item() == 1.0
    assert outputs.logits is None
