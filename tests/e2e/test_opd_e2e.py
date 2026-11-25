"""End-to-end on-policy distillation smoke tests (GPU required)."""

import math
import pathlib

import pytest
import yaml

from conftest import requires_gpu, run_e2e_training

pytestmark = pytest.mark.dense

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _write_opd_config(tmp_path, mode: str, **overrides) -> str:
    config = yaml.safe_load((FIXTURES / "e2e_opd.yaml").read_text())
    config["project_name"] = f"e2e_opd_{mode}"
    config["training_config"].update(
        {
            "use_vllm": mode != "transformers",
            "vllm_mode": "colocate" if mode == "transformers" else mode,
            **overrides,
        }
    )
    if mode == "server":
        config["grpo_rollout"] = {"server_gpus": 1}
    config_path = tmp_path / f"opd_{mode}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return str(config_path)


@pytest.mark.parametrize("mode", ["transformers", "colocate", "server"])
@requires_gpu
def test_opd_generation_modes_train(mode, e2e_output_dir, tmp_path):
    result = run_e2e_training(_write_opd_config(tmp_path, mode), e2e_output_dir)

    metrics = result.metrics
    assert metrics is not None and metrics.get("epoch", 0) > 0, metrics
    assert math.isfinite(metrics["train_loss"]), metrics


@requires_gpu
def test_opd_top_k_loss_trains(e2e_output_dir, tmp_path):
    config_path = _write_opd_config(tmp_path, "colocate", loss_top_k=32)
    result = run_e2e_training(config_path, e2e_output_dir)
    assert math.isfinite(result.metrics["train_loss"]), result.metrics
