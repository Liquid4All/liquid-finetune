from __future__ import annotations

from typing import Any


_MEDIA_MODULE_NAMES = ("vision_tower", "multi_modal_projector")


class VLMParameterParticipationMixin:
    """Keep conditional VLM modules in every distributed backward graph."""

    def compute_loss(self, model, inputs, *args: Any, **kwargs: Any):
        result = super().compute_loss(model, inputs, *args, **kwargs)
        loss = result[0] if isinstance(result, tuple) else result

        zero = None
        for name, parameter in model.named_parameters():
            if (
                parameter.requires_grad
                and parameter.numel()
                and any(module_name in name for module_name in _MEDIA_MODULE_NAMES)
            ):
                term = parameter.reshape(-1)[0] * 0
                zero = term if zero is None else zero + term

        if zero is None:
            return result
        loss = loss + zero
        if isinstance(result, tuple):
            return (loss, *result[1:])
        return loss
