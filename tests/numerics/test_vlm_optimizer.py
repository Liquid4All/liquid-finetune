import torch

from leap_finetune.training.utils.vlm_optimizer import freeze_vlm_modules


class _ToyVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.vision_tower = torch.nn.Module()
        self.model.vision_tower.fc1 = torch.nn.Linear(2, 2)
        self.model.vision_tower.fc1.lora_A = torch.nn.Linear(2, 1, bias=False)
        self.model.vision_tower.fc1.lora_B = torch.nn.Linear(1, 2, bias=False)
        self.model.multi_modal_projector = torch.nn.Linear(2, 2)
        self.model.language_model = torch.nn.Linear(2, 2)


def test_freeze_vision_tower_includes_injected_lora_parameters():
    model = _ToyVLM()

    freeze_vlm_modules(model, ["model.vision_tower"])

    vision_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("model.vision_tower.")
    ]
    assert vision_parameters
    assert all(not parameter.requires_grad for parameter in vision_parameters)
    assert model.model.multi_modal_projector.weight.requires_grad
    assert model.model.language_model.weight.requires_grad
