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
    require_vllm_architecture,
    run_e2e_training,
    run_local_e2e_training,
)

pytestmark = pytest.mark.vlm

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _write_multi_image_grpo_config(
    tmp_path, *, model_name=None, use_peft=False, mode="colocate"
):
    fixture_dir = tmp_path / "grpo_multi_image_fixture"
    image_dir = fixture_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for index, (name, color) in enumerate((("red", "red"), ("blue", "blue"))):
        path = image_dir / f"{name}.png"
        Image.new("RGB", (48 + index * 24, 48), color=color).save(path)
        image_paths.append(str(path))

    rows = []
    for index in range(8):
        content = (
            [{"type": "text", "text": "Reply with a short greeting."}]
            if index % 4 == 0
            else [
                {"type": "image", "image": image_paths[index % 2]},
                {
                    "type": "image",
                    "image": image_paths[(index + 1) % 2],
                },
                {
                    "type": "text",
                    "text": "Describe how these two colored squares differ.",
                },
            ]
        )
        rows.append({"prompt": [{"role": "user", "content": content}]})

    dataset_dir = fixture_dir / "data"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "multi_image_grpo.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    config = yaml.safe_load((FIXTURES / "e2e_vlm_grpo.yaml").read_text())
    config["dataset"]["path"] = str(dataset_path)
    config["dataset"]["limit"] = len(rows)
    if model_name is not None:
        reward_path = FIXTURES / "grpo_reward.py"
        config["rewards"]["funcs"] = [f"{reward_path}::alternating_reward"]
        config["project_name"] = f"e2e_lfm3_vlm_grpo_{mode}"
        config["model_name"] = model_name
        config["training_config"].update(
            {
                "max_steps": 2,
                "steps_per_generation": 1,
                "per_device_train_batch_size": 2,
                "max_completion_length": 4,
                "max_image_tokens": 64,
                "use_vllm": mode != "transformers",
                "vllm_mode": "colocate" if mode == "transformers" else mode,
            }
        )
        config["peft_config"] = (
            {"extends": "DEFAULT_VLM_LORA", "use_peft": True}
            if use_peft
            else {"use_peft": False}
        )
        if mode == "server":
            config["grpo_rollout"] = {"server_gpus": 1, "training_gpus": 3}
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

    @pytest.mark.parametrize("use_peft", [True, False], ids=["lora", "full"])
    @pytest.mark.parametrize("mode", ["transformers", "colocate", "server"])
    @requires_gpu
    def test_lfm3_vlm_grpo_generation_modes_optimize(
        self, use_peft, mode, e2e_output_dir, tmp_path
    ):
        if mode != "transformers":
            require_vllm_architecture("Lfm3VlForConditionalGeneration")
        config_path = _write_multi_image_grpo_config(
            tmp_path,
            model_name="LiquidAI/LFM3-VL-2B",
            use_peft=use_peft,
            mode=mode,
        )
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_grpo_optimization(result, min_steps=2)
