import pytest
import yaml

from conftest import (
    assert_checkpoints_exist,
    assert_eval_callback_logged,
    assert_local_model_saved,
    assert_training_result,
    requires_gpu,
    requires_single_gpu,
    run_e2e_training,
    run_local_e2e_training,
)

pytestmark = pytest.mark.dense

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


# === Dense SFT with LoRA ===


class TestDenseSFTLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_sft_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, max_eval_loss=7.0, check_loss_trend=False)
        assert_eval_callback_logged(result)

    @pytest.mark.single_gpu
    @requires_single_gpu
    def test_local_single_gpu_training(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_sft_lora.yaml")
        run_local_e2e_training(config_path, e2e_output_dir)
        assert_local_model_saved(e2e_output_dir)


# === Dense SFT full fine-tune ===


class TestDenseSFTFull:
    @requires_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_sft_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)
        assert_eval_callback_logged(result)

        assert_checkpoints_exist(e2e_output_dir)


# === Dense DPO with LoRA ===


class TestDenseDPOLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_dpo_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, check_loss_trend=False)
        assert_eval_callback_logged(result)

    @pytest.mark.single_gpu
    @requires_single_gpu
    def test_local_single_gpu_training(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_dpo_lora.yaml")
        run_local_e2e_training(config_path, e2e_output_dir)
        assert_local_model_saved(e2e_output_dir)


class TestLFM3:
    @requires_gpu
    def test_training_completes_and_checkpoints(self, e2e_output_dir, tmp_path):
        config = yaml.safe_load((FIXTURES / "e2e_sft_lora.yaml").read_text())
        config["project_name"] = "e2e_lfm3_sft"
        config["model_name"] = "LiquidAI/LFM3-1.7B"
        config["dataset"]["limit"] = 40
        config["evals"]["benchmarks"][0]["path"] = str(FIXTURES / "tiny_eval.jsonl")
        config["training_config"].update(
            {
                "assistant_only_loss": True,
                "logging_steps": 2,
                "save_strategy": "epoch",
            }
        )
        config_path = tmp_path / "lfm3_sft.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_e2e_training(str(config_path), e2e_output_dir)
        assert_training_result(result, max_eval_loss=7.0, check_loss_trend=False)
        assert_eval_callback_logged(result)
        assert_checkpoints_exist(e2e_output_dir)


# === Dense DPO full fine-tune ===


class TestDenseDPOFull:
    @requires_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_dpo_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, check_loss_trend=False)
        assert_eval_callback_logged(result)

        assert_checkpoints_exist(e2e_output_dir)
