"""End-to-end knowledge distillation smoke tests (GPU required)."""

import pathlib

import pytest
import yaml

from conftest import assert_training_result, requires_gpu, run_e2e_training

pytestmark = pytest.mark.dense

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="forward-kl"),
        pytest.param({"beta": 1.0, "loss_top_k": 64}, id="reverse-kl-top-k"),
    ],
)
@requires_gpu
def test_distill_training_completes_and_evaluates(overrides, e2e_output_dir, tmp_path):
    config = yaml.safe_load((FIXTURES / "e2e_distill.yaml").read_text())
    config["training_config"].update(overrides)
    config_path = tmp_path / "distill.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_e2e_training(str(config_path), e2e_output_dir)
    # eval_loss is the per-token divergence to the teacher, not cross-entropy.
    assert_training_result(result, max_eval_loss=5.0, check_loss_trend=False)
