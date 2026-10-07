from __future__ import annotations

from unittest.mock import Mock

from liquid_finetune.training import grpo


def test_existing_adapter_takes_precedence_over_fresh_peft(monkeypatch) -> None:
    model = Mock()
    loaded_model = Mock()
    load_adapter = Mock(return_value=loaded_model)
    apply_fresh = Mock()
    monkeypatch.setattr(grpo, "load_peft_adapter", load_adapter)
    monkeypatch.setattr(grpo, "apply_peft_to_model", apply_fresh)

    result = grpo._apply_grpo_peft(
        model,
        peft_config=Mock(),
        adapter_path="/models/best-adapter",
    )

    assert result is loaded_model
    load_adapter.assert_called_once_with(model, "/models/best-adapter")
    apply_fresh.assert_not_called()


def test_fresh_adapter_is_applied_without_adapter_path(monkeypatch) -> None:
    model = Mock()
    peft_config = Mock()
    wrapped_model = Mock()
    apply_fresh = Mock(return_value=wrapped_model)
    monkeypatch.setattr(grpo, "apply_peft_to_model", apply_fresh)

    result = grpo._apply_grpo_peft(
        model,
        peft_config=peft_config,
        adapter_path=None,
    )

    assert result is wrapped_model
    apply_fresh.assert_called_once_with(model, peft_config)


def test_model_is_unchanged_without_peft() -> None:
    model = Mock()

    result = grpo._apply_grpo_peft(
        model,
        peft_config=None,
        adapter_path=None,
    )

    assert result is model
