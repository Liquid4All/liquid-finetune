from __future__ import annotations

from typing import Any
from unittest.mock import patch

from leap_finetune.quantization.qat.controller import prepare_model_for_qat


class QATGRPOReferenceMixin:
    """Apply the policy QAT profile to TRL's internally-created reference."""

    def __init__(self, *args: Any, qat_config: dict | None = None, **kwargs: Any):
        if qat_config is None:
            super().__init__(*args, **kwargs)
            return

        # TRL creates the full-weight reference in GRPOTrainer.__init__, then
        # prepares it with FSDP/DeepSpeed/Accelerate before returning. Wrap its
        # factory so QAT is installed at creation time, while parameters are
        # still unwrapped and unsharded. PEFT references use adapter disabling
        # on the already-prepared policy and do not call this factory.
        import trl.trainer.grpo_trainer as grpo_module

        original_factory = grpo_module.create_model_from_path

        def create_qat_reference(*factory_args: Any, **factory_kwargs: Any):
            reference = original_factory(*factory_args, **factory_kwargs)
            prepare_model_for_qat(
                reference,
                {"qat": qat_config},
                is_vlm=bool(getattr(self, "_is_vlm", False)),
            )
            return reference

        with patch.object(
            grpo_module, "create_model_from_path", new=create_qat_reference
        ):
            super().__init__(*args, **kwargs)
