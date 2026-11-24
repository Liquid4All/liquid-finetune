"""End-to-end GRPO training smoke tests (GPU required).

These tests verify the whole pipeline: dataset loading, reward resolution,
vLLM colocate instantiation, multiple optimizer steps, and learning signals.
They deliberately use tiny settings (num_generations=2, 16 samples,
max_completion_length=16) so they finish in under ~5 minutes on 1 H100.

Run an individual case on a local GPU with:
    uv run pytest tests/e2e/test_grpo_e2e.py::TestDenseGRPO::test_text_grpo_optimizes -v

Submit the complete E2E matrix with:
    tests/e2e/slurm/submit_e2e_tests.sh
"""

import pathlib

import pytest
import yaml

from conftest import (
    assert_grpo_optimization,
    requires_gpu,
    requires_single_gpu,
    require_vllm_architecture,
    run_e2e_training,
    run_local_e2e_training,
)

pytestmark = pytest.mark.dense

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _write_lfm3_grpo_config(tmp_path, *, use_peft: bool, mode: str) -> str:
    config = yaml.safe_load((FIXTURES / "e2e_grpo.yaml").read_text())
    config["project_name"] = f"e2e_lfm3_grpo_{mode}"
    config["model_name"] = "LiquidAI/LFM3-1.7B"
    reward_path = FIXTURES / "grpo_reward.py"
    config["rewards"]["funcs"] = [f"{reward_path}::alternating_reward"]
    config["training_config"].update(
        {
            "max_steps": 4,
            "use_vllm": mode != "transformers",
            "vllm_mode": "colocate" if mode == "transformers" else mode,
        }
    )
    config["peft_config"] = (
        {"extends": "DEFAULT_LORA", "use_peft": True}
        if use_peft
        else {"use_peft": False}
    )
    if mode == "server":
        config["grpo_rollout"] = {"server_gpus": 1, "training_gpus": 3}
    config_path = tmp_path / f"lfm3_grpo_{mode}_{use_peft}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return str(config_path)


class TestDenseGRPO:
    @requires_gpu
    def test_text_grpo_optimizes(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_grpo.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_grpo_optimization(result)

    @pytest.mark.single_gpu
    @requires_single_gpu
    def test_local_single_gpu_training(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_grpo.yaml")
        run_local_e2e_training(config_path, e2e_output_dir)

    @pytest.mark.parametrize("use_peft", [True, False], ids=["lora", "full"])
    @pytest.mark.parametrize("mode", ["transformers", "colocate", "server"])
    @requires_gpu
    def test_lfm3_grpo_generation_modes_optimize(
        self, use_peft, mode, e2e_output_dir, tmp_path
    ):
        if mode != "transformers":
            require_vllm_architecture("Lfm3ForCausalLM")
        config_path = _write_lfm3_grpo_config(tmp_path, use_peft=use_peft, mode=mode)
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_grpo_optimization(result, min_steps=4)
