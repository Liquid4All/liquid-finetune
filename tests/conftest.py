import math
import os
import pathlib
import re
import shutil

import pytest
import torch
import yaml

from liquid_finetune import LIQUID_FINETUNE_DIR

# === Ray temp dir ===
_RAY_TMPDIR = pathlib.Path(f"/tmp/{os.environ.get('USER', 'default')}/ray")
_RAY_TMPDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("RAY_TMPDIR", str(_RAY_TMPDIR))

# === E2E test output dir ===

_DEFAULT_TEST_RESULTS_DIR = LIQUID_FINETUNE_DIR / ".test-results" / "e2e"


def _e2e_test_results_dir() -> pathlib.Path:
    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir:
        return pathlib.Path(output_dir).expanduser().resolve()
    return _DEFAULT_TEST_RESULTS_DIR


# === CLI flag registration ===


def pytest_addoption(parser):
    parser.addoption("--configs", action="store_true", help="Run only config tests")
    parser.addoption("--dense", action="store_true", help="Run only dense GPU tests")
    parser.addoption("--vlm", action="store_true", help="Run only VLM GPU tests")
    parser.addoption("--moe", action="store_true", help="Run only MoE GPU tests")
    parser.addoption(
        "--retrieval", action="store_true", help="Run only retrieval GPU tests"
    )


def pytest_collection_modifyitems(config, items):
    flag_mark_map = {
        "configs": "configs",
        "dense": "dense",
        "vlm": "vlm",
        "moe": "moe",
        "retrieval": "retrieval",
    }
    active = [
        mark
        for flag, mark in flag_mark_map.items()
        if config.getoption(flag, default=False)
    ]

    if not active:
        return

    skip = pytest.mark.skip(reason="Not selected by CLI flags")
    for item in items:
        item_marks = {m.name for m in item.iter_markers()}
        if not item_marks.intersection(active):
            item.add_marker(skip)


# === Skip markers ===

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No GPU available",
)

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires 2+ GPUs",
)

requires_single_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() != 1,
    reason="Requires exactly 1 GPU",
)


# === Shared fixtures ===


@pytest.fixture
def job_configs_dir():
    return LIQUID_FINETUNE_DIR / "job_configs"


@pytest.fixture
def sft_config_path(job_configs_dir):
    return str(job_configs_dir / "sft_example.yaml")


@pytest.fixture
def dpo_config_path(job_configs_dir):
    return str(job_configs_dir / "dpo_example.yaml")


@pytest.fixture
def kto_config_path(job_configs_dir):
    return str(job_configs_dir / "kto_example.yaml")


@pytest.fixture
def vlm_config_path(job_configs_dir):
    return str(job_configs_dir / "vlm_sft_example.yaml")


@pytest.fixture
def moe_sft_config_path(job_configs_dir):
    return str(job_configs_dir / "moe_sft_example.yaml")


@pytest.fixture
def moe_dpo_config_path(job_configs_dir):
    return str(job_configs_dir / "moe_dpo_example.yaml")


BASE_SFT_DATASET = {
    "path": "HuggingFaceTB/smoltalk",
    "type": "sft",
    "limit": 10,
    "test_size": 0.2,
    "subset": "all",
}

BASE_DPO_DATASET = {
    "path": "mlabonne/orpo-dpo-mix-40k",
    "type": "dpo",
    "limit": 10,
    "test_size": 0.2,
    "subset": "default",
}


def write_config(config: dict, tmp_path: pathlib.Path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


# === E2E training helper ===


@pytest.fixture
def e2e_output_dir():
    """Provide the configured e2e output dir, cleaned up after each test."""
    test_results_dir = _e2e_test_results_dir()
    test_results_dir.mkdir(parents=True, exist_ok=True)
    yield test_results_dir
    shutil.rmtree(test_results_dir, ignore_errors=True)


def run_local_e2e_training(
    config_path: str, output_dir: pathlib.Path, *, max_steps: int = 2
):
    """Run a short job through the automatic local single-GPU dispatcher."""
    previous_output_dir = os.environ.get("OUTPUT_DIR")
    os.environ["OUTPUT_DIR"] = str(output_dir)
    try:
        from liquid_finetune.config.parser import materialize_job_config, parse_job_config
        from liquid_finetune.distribution.local_trainer import (
            local_trainer,
            should_use_local,
        )

        job_config = materialize_job_config(parse_job_config(config_path))
        job_config_dict = job_config.to_dict()
        job_config_dict["training_config"]["max_steps"] = max_steps
        assert should_use_local(job_config_dict), "Expected local single-GPU dispatch"
        local_trainer(job_config_dict)
    finally:
        if previous_output_dir is None:
            os.environ.pop("OUTPUT_DIR", None)
        else:
            os.environ["OUTPUT_DIR"] = previous_output_dir


def assert_local_model_saved(output_dir: pathlib.Path):
    assert any(output_dir.rglob("config.json")), (
        f"No merged local model found under {output_dir}"
    )


def run_e2e_training(config_path: str, output_dir: pathlib.Path):
    """Parse config, override output_dir, run training, return Result."""
    previous_output_dir = os.environ.get("OUTPUT_DIR")
    os.environ["OUTPUT_DIR"] = str(output_dir)
    try:
        from liquid_finetune.cli.main import run_config

        return run_config(config_path)
    finally:
        if previous_output_dir is None:
            os.environ.pop("OUTPUT_DIR", None)
        else:
            os.environ["OUTPUT_DIR"] = previous_output_dir


def assert_training_result(
    result, max_eval_loss=5.0, check_loss_trend=True, check_dpo_preference=False
):
    """Verify training completed, produced finite loss, and optionally learning signals.

    Args:
        result: Ray Train Result object.
        max_eval_loss: Upper bound on eval_loss. Random cross-entropy for
            vocab=65536 is ~11.1, so anything above max_eval_loss indicates
            the model didn't learn. Default 5.0 is generous for 1-epoch runs.
        check_loss_trend: Whether to assert loss trends downward. Should be
            False for DPO — DPO loss measures preference margin, not
            cross-entropy, so it often stays flat or fluctuates even when
            the model is learning (eval_rewards/accuracies is the real signal).
        check_dpo_preference: Whether to assert DPO reward accuracy > 0.5 and
            positive reward margins. Use for DPO full fine-tune where the model
            has enough capacity and steps to learn preferences.
    """
    assert result is not None, "Training returned no result"
    metrics = result.metrics
    assert metrics is not None, "No metrics in training result"

    # Training must have run at least 1 epoch
    assert "epoch" in metrics, f"No epoch in metrics: {metrics}"

    # eval_loss must exist, be finite, and show the model learned
    assert "eval_loss" in metrics, f"No eval_loss in metrics: {metrics}"
    eval_loss = metrics["eval_loss"]
    assert math.isfinite(eval_loss), f"eval_loss is not finite: {eval_loss}"
    assert eval_loss < max_eval_loss, (
        f"eval_loss {eval_loss:.4f} >= {max_eval_loss} — model did not learn. "
        f"Random baseline for vocab=65536 is ~11.1"
    )

    # Check for loss trend downward from first and last quarter of training.
    if check_loss_trend:
        loss_history = metrics.get("loss_history", [])
        if len(loss_history) >= 4:
            q = max(1, len(loss_history) // 4)
            early_avg = sum(loss_history[:q]) / q
            late_avg = sum(loss_history[-q:]) / q
            assert late_avg < early_avg, (
                f"Loss did not trend down: "
                f"first quarter avg={early_avg:.4f} → last quarter avg={late_avg:.4f}"
            )

    # DPO-specific -- check for reward margin / acc over the loss trend.
    if check_dpo_preference:
        if "eval_rewards/accuracies" in metrics:
            acc = metrics["eval_rewards/accuracies"]
            assert acc > 0.5, (
                f"DPO eval reward accuracy {acc:.2f} <= 0.5 — "
                f"model is not preferring chosen over rejected"
            )
        if "eval_rewards/margins" in metrics:
            margin = metrics["eval_rewards/margins"]
            assert margin > 0.0, (
                f"DPO eval reward margin {margin:.4f} <= 0 — "
                f"chosen reward is not higher than rejected"
            )

    # train_loss should also be present and finite
    if "train_loss" in metrics:
        assert math.isfinite(metrics["train_loss"]), (
            f"train_loss is not finite: {metrics['train_loss']}"
        )


def assert_grpo_optimization(result, min_steps=4, min_reward_logs=2):
    """Require reward variation and nonzero gradients over several updates.

    Short stochastic GRPO jobs cannot reliably require a monotonic reward
    curve. These checks instead prove that the policy received a non-constant
    learning signal and performed optimizer steps with a positive LR.
    """
    assert result is not None, "Training returned no result"
    metrics = result.metrics
    assert metrics is not None, "No metrics in training result"
    assert metrics.get("epoch", 0) > 0, f"No epoch progress: {metrics}"

    if "train_loss" in metrics:
        assert math.isfinite(metrics["train_loss"]), (
            f"train_loss is not finite: {metrics['train_loss']}"
        )

    histories = {
        key: metrics.get(key, [])
        for key in (
            "reward_history",
            "reward_std_history",
            "grad_norm_history",
            "learning_rate_history",
        )
    }
    minimum_entries = {
        "reward_history": min_reward_logs,
        "reward_std_history": min_reward_logs,
        "grad_norm_history": min_steps,
        "learning_rate_history": min_steps,
    }
    for key, values in histories.items():
        assert len(values) >= minimum_entries[key], f"Too few {key} entries: {values}"
        assert all(math.isfinite(value) for value in values), (
            f"Non-finite {key}: {values}"
        )

    assert any(value > 0 for value in histories["reward_std_history"]), (
        f"Rewards never varied within a generation group: {histories}"
    )
    assert any(value > 0 for value in histories["grad_norm_history"]), (
        f"GRPO produced no nonzero gradients: {histories}"
    )
    assert any(value > 0 for value in histories["learning_rate_history"]), (
        f"GRPO never used a positive learning rate: {histories}"
    )


def assert_eval_callback_logged(result):
    metrics = result.metrics or {}
    benchmark_keys = [key for key in metrics if key.startswith("benchmark/")]
    assert benchmark_keys, f"No benchmark/eval metrics found in result: {metrics}"


def assert_checkpoints_exist(output_dir: pathlib.Path):
    """Verify at least one checkpoint directory exists (original or renamed)."""
    checkpoint_dirs = list(output_dir.rglob("checkpoint-*"))
    renamed_dirs = [
        d
        for d in output_dir.rglob("*")
        if d.is_dir() and re.search(r"-e\d+s\d+-", d.name)
    ]
    assert len(checkpoint_dirs) + len(renamed_dirs) > 0, (
        f"No checkpoint directories found under {output_dir}. "
        f"Contents: {[p.name for p in output_dir.rglob('*') if p.is_dir()]}"
    )
