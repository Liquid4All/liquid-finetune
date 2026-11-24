"""End-to-end KTO training smoke tests (GPU required)."""

import pytest
import yaml

from conftest import assert_training_result, requires_gpu, run_e2e_training

pytestmark = pytest.mark.dense

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


@requires_gpu
def test_kto_training_completes_and_evaluates(e2e_output_dir):
    result = run_e2e_training(str(FIXTURES / "e2e_kto.yaml"), e2e_output_dir)
    assert_training_result(result, max_eval_loss=5.0, check_loss_trend=False)


@pytest.mark.parametrize("use_peft", [True, False])
@requires_gpu
def test_lfm3_kto_training_completes_and_evaluates(use_peft, e2e_output_dir, tmp_path):
    config = yaml.safe_load((FIXTURES / "e2e_kto.yaml").read_text())
    config["project_name"] = "e2e_lfm3_kto"
    config["model_name"] = "LiquidAI/LFM3-1.7B"
    config["peft_config"] = (
        {"extends": "DEFAULT_LORA", "use_peft": True}
        if use_peft
        else {"use_peft": False}
    )
    config["training_config"]["save_strategy"] = "epoch"
    config_path = tmp_path / "lfm3_kto.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_e2e_training(str(config_path), e2e_output_dir)
    assert_training_result(result, max_eval_loss=7.0, check_loss_trend=False)
