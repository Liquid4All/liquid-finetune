import json

import numpy as np
import pytest
from PIL import Image

from liquid_finetune.config import materialize_job_config, parse_job_config
from liquid_finetune.data_loading.validate_dataset_format import (
    get_row_filter,
    normalize_columns,
)
from liquid_finetune.rl.rewards import resolve_reward_specs
from liquid_finetune.training.default_configs import TRAINING_DEFAULTS

from conftest import write_config

pytestmark = pytest.mark.configs


GRPO_DATASET = {
    "path": "trl-lib/DeepMath-103K",
    "type": "grpo",
    "limit": 10,
}

VLM_GRPO_DATASET = {
    "path": "alay2shah/example-vlm-sft-dataset",
    "type": "vlm_grpo",
    "limit": 10,
    "image_root": "/tmp/images",
}


class TestGRPODefaultProfiles:
    def test_grpo_defaults_discovered(self):
        names = set(TRAINING_DEFAULTS)
        assert "DEFAULT_GRPO" in names
        assert "DEFAULT_VLM_GRPO" in names
        assert "MOE_GRPO" in names

    def test_default_grpo_has_expected_fields(self):
        cfg = TRAINING_DEFAULTS["DEFAULT_GRPO"]
        assert cfg["training_type"] == "grpo"
        assert cfg["loss_type"] == "dapo"
        assert cfg["beta"] == 0.0
        assert cfg["vllm_mode"] == "colocate"


class TestGRPOSmoke:
    def test_minimal_grpo_materializes(self, tmp_path):
        config = {
            "project_name": "test_grpo",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": GRPO_DATASET,
            "training_config": {"extends": "DEFAULT_GRPO"},
            "rewards": {
                "funcs": ["./rewards/length.py::length_reward"],
                "weights": [1.0],
            },
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        jc = materialize_job_config(parsed)
        assert jc.training_type == "grpo"
        assert jc.dataset.test_size == 0.01
        assert jc.rewards["weights"] == [1.0]
        assert jc.config_dir == str(tmp_path.resolve())

    def test_reward_paths_are_absolutized(self, tmp_path):
        reward_file = tmp_path / "local_reward.py"
        reward_file.write_text(
            "def local_reward(completions, **kwargs):\n"
            "    return [1.0 for _ in completions]\n"
        )
        config = {
            "project_name": "test_grpo",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": GRPO_DATASET,
            "training_config": {"extends": "DEFAULT_GRPO"},
            "rewards": {
                "funcs": ["./local_reward.py::local_reward"],
                "weights": [1.0],
            },
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        jc = materialize_job_config(parsed)
        assert jc.rewards["funcs"] == [f"{reward_file.resolve()}::local_reward"]

    def test_judge_reward_config_persists(self, tmp_path):
        config = {
            "project_name": "t",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": GRPO_DATASET,
            "training_config": {"extends": "DEFAULT_GRPO"},
            "rewards": {
                "judge": {
                    "model": "LFM2-1.2B",
                    "base_url": "http://judge:8001",
                    "weight": 0.5,
                }
            },
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        jc = materialize_job_config(parsed)
        funcs, weights = resolve_reward_specs(jc.rewards, tmp_path)
        assert [f.__name__ for f in funcs] == ["judge_llm_reward"]
        assert weights == [0.5]

    def test_vlm_grpo_materializes(self, tmp_path):
        config = {
            "project_name": "vlm_grpo_test",
            "model_name": "LFM2-VL-1.6B",
            "training_type": "vlm_grpo",
            "dataset": VLM_GRPO_DATASET,
            "training_config": {"extends": "DEFAULT_VLM_GRPO"},
            "rl_env": {
                "source": "liquidai/vlm-grounding-bbox-env",
                "max_turns": 1,
            },
            "grpo_rollout": {"server_gpus": 1, "tensor_parallel_size": 1},
            "rewards": {
                "funcs": ["./rewards/length.py::length_reward"],
                "weights": [0.2],
            },
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        jc = materialize_job_config(parsed)
        assert jc.training_type == "vlm_grpo"
        assert jc.rl_env["source"] == "liquidai/vlm-grounding-bbox-env"
        assert jc.grpo_rollout["server_gpus"] == 1
        assert jc.training_config.value["lr_multipliers"]["model.vision_tower"] == 0.1

    def test_vlm_grpo_normalizes_nested_ray_arrays(self, tmp_path):
        image_path = tmp_path / "square.png"
        Image.new("RGB", (8, 8), color="red").save(image_path)
        row = {
            "prompt": np.array(
                [
                    {
                        "role": "user",
                        "content": np.array(
                            [
                                json.dumps({"type": "image", "image": str(image_path)}),
                                json.dumps({"type": "text", "text": "Describe it."}),
                            ],
                            dtype=object,
                        ),
                    }
                ],
                dtype=object,
            )
        }

        normalized = normalize_columns("vlm_grpo")(row)

        assert isinstance(normalized["prompt"][0]["content"], list)
        assert get_row_filter("vlm_grpo")(normalized)

    def test_grpo_keys_rejected_on_sft(self, tmp_path):
        config = {
            "project_name": "t",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": {
                "path": "HuggingFaceTB/smoltalk",
                "type": "sft",
                "limit": 10,
                "test_size": 0.2,
            },
            "rewards": ["./rewards/length.py::length_reward"],
        }
        with pytest.raises(ValueError, match="only valid for training_type"):
            parse_job_config(write_config(config, tmp_path))
