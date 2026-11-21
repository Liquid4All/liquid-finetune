import pytest
import torch
from PIL import Image

from conftest import (
    assert_checkpoints_exist,
    assert_local_model_saved,
    assert_training_result,
    requires_gpu,
    requires_single_gpu,
    run_local_e2e_training,
    run_e2e_training,
)
from liquid_finetune.checkpointing.model_loading import load_vlm_model
from liquid_finetune.data_loading.tokenize_data import create_vlm_collate_fn
from liquid_finetune.training.peft.peft import apply_peft_to_model
from liquid_finetune.training.peft.peft_configs import DEFAULT_VLM_LORA

pytestmark = pytest.mark.vlm

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


# === VLM SFT with LoRA ===


class TestVLMLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, e2e_output_dir, tmp_path):
        config_path = _write_vlm_sft_config(tmp_path, "e2e_vlm_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)

    @pytest.mark.single_gpu
    @requires_single_gpu
    def test_local_single_gpu_training(self, e2e_output_dir, tmp_path):
        config_path = _write_vlm_sft_config(tmp_path, "e2e_vlm_lora.yaml")
        run_local_e2e_training(config_path, e2e_output_dir)
        assert_local_model_saved(e2e_output_dir)


class TestLFM3VL:
    @requires_gpu
    def test_checkpoint_collation_and_lora_backward(self):
        model, processor = load_vlm_model("LiquidAI/LFM3-VL-2B", max_image_tokens=256)
        samples = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": Image.new("RGB", (96, 64), "red"),
                            },
                            {"type": "text", "text": "What color is this?"},
                        ],
                    },
                    {"role": "assistant", "content": "Red."},
                ]
            }
        ]
        batch = create_vlm_collate_fn(processor, max_length=1024)(samples)
        assert torch.all(
            batch["labels"][batch["input_ids"] == processor.image_token_id] == -100
        )

        model = apply_peft_to_model(model.cuda(), DEFAULT_VLM_LORA)
        loss = model(**{key: value.cuda() for key, value in batch.items()}).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )


# === VLM SFT full fine-tune ===


class TestVLMFull:
    @requires_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir, tmp_path):
        config_path = _write_vlm_sft_config(tmp_path, "e2e_vlm_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)

        assert_checkpoints_exist(e2e_output_dir)


# === VLM SFT with string assistant content + multi-image ===


def _write_vlm_sft_config(tmp_path, fixture_name, *, string_assistant=False):
    """Create a self-contained VLM SFT fixture from a shipped config."""
    import json

    import pandas as pd
    import yaml
    from PIL import Image

    imgdir = tmp_path / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    paths = []
    for i, c in enumerate(colors):
        p = imgdir / f"img{i}.png"
        Image.new("RGB", (48, 48), c).save(p)
        paths.append(str(p))

    rows = []
    for i in range(40):
        if i % 7 == 0:  # multi-image rows
            content = [
                {"type": "image", "image": paths[i % 4]},
                {"type": "image", "image": paths[(i + 1) % 4]},
                {"type": "text", "text": "Compare the two images."},
            ]
            answer = "The two images differ in color."
        else:
            content = [
                {"type": "image", "image": paths[i % 4]},
                {"type": "text", "text": "What color is this?"},
            ]
            answer = "It is a colored square."
        assistant_content = (
            answer if string_assistant else [{"type": "text", "text": answer}]
        )
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": assistant_content},
        ]
        rows.append({"messages": json.dumps(messages)})

    ds_path = tmp_path / "data.parquet"
    pd.DataFrame(rows).to_parquet(ds_path)

    config = yaml.safe_load((FIXTURES / fixture_name).read_text())
    config["dataset"]["path"] = str(ds_path)
    if string_assistant:
        config["project_name"] = "e2e_vlm_string"
        config["training_config"]["gradient_checkpointing"] = True
        config["training_config"]["save_strategy"] = "epoch"
        config["training_config"]["logging_steps"] = 2
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    return str(cfg_path)


class TestVLMStringContent:
    @requires_gpu
    def test_string_assistant_and_multi_image_train(self, e2e_output_dir, tmp_path):
        """Regression: VLM datasets with plain-string assistant content (and
        multi-image turns) must train, not be silently dropped at the row filter."""
        config_path = _write_vlm_sft_config(
            tmp_path, "e2e_vlm_lora.yaml", string_assistant=True
        )
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)
        assert_checkpoints_exist(e2e_output_dir)
