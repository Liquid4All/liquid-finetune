import pytest

from conftest import (
    assert_checkpoints_exist,
    assert_local_model_saved,
    assert_training_result,
    requires_multi_gpu,
    requires_single_gpu,
    run_e2e_training,
    run_local_e2e_training,
)

pytestmark = pytest.mark.moe

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


# === MoE SFT with LoRA (DeepSpeed stage 0 path) ===


class TestMoESFTLoRA:
    @requires_multi_gpu
    def test_training_completes_and_learns(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_moe_sft_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, max_eval_loss=7.0, check_loss_trend=False)

    @pytest.mark.single_gpu
    @requires_single_gpu
    def test_local_single_gpu_training(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_moe_sft_lora.yaml")
        run_local_e2e_training(config_path, e2e_output_dir)
        assert_local_model_saved(e2e_output_dir)


# === MoE SFT full fine-tune (FSDP path) ===


class TestMoESFTFull:
    @requires_multi_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_moe_sft_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)

        assert_checkpoints_exist(e2e_output_dir)
