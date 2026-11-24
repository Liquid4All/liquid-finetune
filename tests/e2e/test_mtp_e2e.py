import json
import shutil

import pytest
import torch
import yaml
from safetensors import safe_open

from conftest import (
    assert_checkpoints_exist,
    assert_hf_export_reloads_and_generates,
    assert_training_result,
    requires_gpu,
    run_e2e_training,
)
from liquid_finetune.models.lfm3_mtp import prepare_lfm3_mtp_checkpoint

pytestmark = pytest.mark.dense

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def _checkpoint_tensor(model_dir, name):
    index_path = model_dir / "model.safetensors.index.json"
    shard = (
        json.loads(index_path.read_text())["weight_map"][name]
        if index_path.exists()
        else "model.safetensors"
    )
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


class TestLFM3MTP:
    @requires_gpu
    def test_heads_train_export_and_reload(self, e2e_output_dir, tmp_path):
        start = e2e_output_dir / "mtp-start"
        prepare_lfm3_mtp_checkpoint("LiquidAI/LFM3-1.7B", str(start))
        backbone_before = _checkpoint_tensor(
            start, "model.layers.0.operator_norm.weight"
        )
        mtp_before = _checkpoint_tensor(start, "mtp.fc.weight")

        config = yaml.safe_load((FIXTURES / "e2e_sft_full.yaml").read_text())
        config["project_name"] = "e2e_lfm3_mtp_heads"
        config["model_name"] = str(start)
        config["evals"]["benchmarks"][0]["path"] = str(FIXTURES / "tiny_eval.jsonl")
        config["training_config"].update(
            {
                "assistant_only_loss": True,
                "learning_rate": 1e-4,
                "logging_steps": 10,
                "save_strategy": "epoch",
            }
        )
        config_path = tmp_path / "lfm3_mtp_heads.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_e2e_training(str(config_path), e2e_output_dir)
        assert_training_result(result, max_eval_loss=20.0)
        assert_checkpoints_exist(e2e_output_dir)

        # Remove the immutable start point so the generic export assertion can
        # only select a checkpoint produced by this training run.
        shutil.rmtree(start)
        exports = sorted(
            config.parent
            for config in e2e_output_dir.rglob("config.json")
            if (config.parent / "model.safetensors").exists()
            or (config.parent / "model.safetensors.index.json").exists()
        )
        assert exports
        exported = exports[-1]
        torch.testing.assert_close(
            _checkpoint_tensor(exported, "model.layers.0.operator_norm.weight"),
            backbone_before,
            rtol=0,
            atol=0,
        )
        assert not torch.equal(
            _checkpoint_tensor(exported, "mtp.fc.weight"), mtp_before
        )
        assert_hf_export_reloads_and_generates(e2e_output_dir, modality="text")
