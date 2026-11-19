"""Config handling that has to survive the Transformers 5.x training-args changes."""

import pytest
from transformers import TrainingArguments

from liquid_finetune.training.utils.config_filter import filter_runtime_config_kwargs

pytestmark = pytest.mark.configs


class _ArgsWithoutWarmupRatio:
    def __init__(self, learning_rate=1e-5, warmup_steps=0):
        pass


class _ArgsWithWarmupRatio:
    def __init__(self, learning_rate=1e-5, warmup_steps=0, warmup_ratio=0.0):
        pass


def test_warmup_ratio_becomes_fractional_warmup_steps():
    filtered, dropped = filter_runtime_config_kwargs(
        {"learning_rate": 2e-5, "warmup_ratio": 0.05},
        excluded_keys=set(),
        config_cls=_ArgsWithoutWarmupRatio,
    )
    assert filtered == {"learning_rate": 2e-5, "warmup_steps": 0.05}
    assert dropped == []


def test_explicit_warmup_steps_wins_over_warmup_ratio():
    filtered, _ = filter_runtime_config_kwargs(
        {"warmup_ratio": 0.05, "warmup_steps": 10},
        excluded_keys=set(),
        config_cls=_ArgsWithoutWarmupRatio,
    )
    assert filtered == {"warmup_steps": 10}


def test_warmup_ratio_kept_when_config_class_accepts_it():
    filtered, _ = filter_runtime_config_kwargs(
        {"warmup_ratio": 0.1},
        excluded_keys=set(),
        config_cls=_ArgsWithWarmupRatio,
    )
    assert filtered == {"warmup_ratio": 0.1}


def test_out_of_range_warmup_ratio_is_rejected():
    with pytest.raises(ValueError, match="warmup_ratio"):
        filter_runtime_config_kwargs(
            {"warmup_ratio": 1.5},
            excluded_keys=set(),
            config_cls=_ArgsWithoutWarmupRatio,
        )


def test_default_warmup_ratio_reaches_installed_training_arguments(tmp_path):
    filtered, dropped = filter_runtime_config_kwargs(
        {"warmup_ratio": 0.1, "max_steps": 200},
        excluded_keys=set(),
        config_cls=TrainingArguments,
    )
    assert dropped == []
    args = TrainingArguments(output_dir=str(tmp_path), report_to=[], **filtered)
    assert args.get_warmup_steps(200) == 20
