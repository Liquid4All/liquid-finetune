"""End-to-end VLM GRPO smoke test (GPU required).

Verifies the VLM GRPO training loop works end-to-end including:
- VLM model loading via load_vlm_model
- Per-component LR param groups (vision_tower at 0.1x base LR)
- Dataset validation and rollout with two images in every prompt
- GRPO rollout + reward computation
- Multiple optimizer steps with nonzero gradients

Run an individual case on a local GPU with:
    uv run pytest tests/e2e/test_vlm_grpo_e2e.py::TestVLMGRPO::test_vlm_grpo_multi_image_optimizes -v

Submit the complete E2E matrix with:
    tests/e2e/slurm/submit_e2e_tests.sh
"""

import json
import pathlib

import pytest
import yaml
from PIL import Image

from conftest import (
    assert_grpo_optimization,
    requires_gpu,
    requires_single_gpu,
    run_e2e_training,
    run_local_e2e_training,
)

pytestmark = pytest.mark.vlm

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _write_multi_image_grpo_config(tmp_path):
    fixture_dir = tmp_path / "grpo_multi_image_fixture"
    image_dir = fixture_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for name, color in (("red", "red"), ("blue", "blue")):
        path = image_dir / f"{name}.png"
        Image.new("RGB", (48, 48), color=color).save(path)
        image_paths.append(str(path))

    rows = []
    for index in range(8):
        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_paths[index % 2]},
                            {
                                "type": "image",
                                "image": image_paths[(index + 1) % 2],
                            },
                            {
                                "type": "text",
                                "text": "Describe how these two colored squares differ.",
                            },
                        ],
                    }
                ]
            }
        )

    dataset_dir = fixture_dir / "data"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "multi_image_grpo.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    config = yaml.safe_load((FIXTURES / "e2e_vlm_grpo.yaml").read_text())
    config["dataset"]["path"] = str(dataset_path)
    config["dataset"]["limit"] = len(rows)
    reward_path = FIXTURES.parents[2] / "rewards" / "length.py"
    config["rewards"]["funcs"] = [f"{reward_path}::length_reward"]
    config_dir = fixture_dir / "config"
    config_dir.mkdir()
    config_path = config_dir / "e2e_vlm_grpo_multi_image.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return str(config_path)


class TestVLMGRPO:
    @requires_gpu
    def test_vlm_grpo_multi_image_optimizes(self, e2e_output_dir, tmp_path):
        config_path = _write_multi_image_grpo_config(tmp_path)
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_grpo_optimization(result)
        metrics = result.metrics

        # Per-component LR metrics should be logged because
        # LFMVLMGRPOTrainer.log() injects lr/<component> entries.
        lr_keys = [k for k in metrics if k.startswith("lr/")]
        assert lr_keys, (
            f"No per-component LR metrics logged. Expected lr/vision_tower etc. "
            f"All metrics: {list(metrics)}"
        )

    @pytest.mark.single_gpu
    @requires_single_gpu
    def test_local_single_gpu_training(self, e2e_output_dir, tmp_path):
        config_path = _write_multi_image_grpo_config(tmp_path)
        run_local_e2e_training(config_path, e2e_output_dir)
