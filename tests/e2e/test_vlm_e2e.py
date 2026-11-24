import pytest

from conftest import (
    assert_checkpoints_exist,
    assert_hf_export_reloads_and_generates,
    assert_local_model_saved,
    assert_training_result,
    requires_gpu,
    requires_single_gpu,
    run_local_e2e_training,
    run_e2e_training,
)

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
    @pytest.mark.parametrize("fixture_name", ["e2e_vlm_lora.yaml", "e2e_vlm_full.yaml"])
    @requires_gpu
    def test_training_completes_and_checkpoints(
        self, fixture_name, e2e_output_dir, tmp_path
    ):
        config_path = _write_vlm_sft_config(
            tmp_path,
            fixture_name,
            model_name="LiquidAI/LFM3-VL-2B",
            save_checkpoints=True,
        )
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, max_eval_loss=7.0, check_loss_trend=False)
        assert_checkpoints_exist(e2e_output_dir)
        assert_hf_export_reloads_and_generates(
            e2e_output_dir,
            modality="vlm",
            image_path=str(tmp_path / "images" / "img0.png"),
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


def _write_vlm_sft_config(
    tmp_path,
    fixture_name,
    *,
    string_assistant=False,
    model_name=None,
    save_checkpoints=False,
):
    """Create a self-contained VLM SFT fixture from a shipped config."""
    import json

    import pandas as pd
    import yaml
    from PIL import Image

    imgdir = tmp_path / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    paths = []
    sizes = [(48, 48), (64, 40), (40, 72), (80, 56)]
    for i, c in enumerate(colors):
        p = imgdir / f"img{i}.png"
        Image.new("RGB", sizes[i], c).save(p)
        paths.append(str(p))

    rows = []
    for i in range(40):
        if i % 11 == 0:  # text-only rows mixed into the VLM batch
            content = [{"type": "text", "text": "Reply with a short greeting."}]
            answer = "Hello."
        elif i % 7 == 0:  # multi-image rows
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
    if model_name is not None:
        config["project_name"] = "e2e_lfm3_vlm"
        config["model_name"] = model_name
        config["training_config"].update(
            {
                "logging_steps": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-5,
                "max_image_tokens": 256,
                "max_length": 1024,
            }
        )
    if save_checkpoints:
        config["training_config"]["save_strategy"] = "epoch"
    if string_assistant:
        config["project_name"] = "e2e_vlm_string"
        config["training_config"]["gradient_checkpointing"] = True
        config["training_config"]["save_strategy"] = "epoch"
        config["training_config"]["logging_steps"] = 2
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    return str(cfg_path)


def _write_lfm3_vlm_dpo_config(tmp_path, *, use_peft: bool) -> str:
    import json

    from PIL import Image

    image_dir = tmp_path / "dpo_images"
    image_dir.mkdir()
    image_paths = []
    for index, (size, color) in enumerate(
        [((48, 48), "red"), ((72, 40), "blue"), ((40, 80), "green")]
    ):
        path = image_dir / f"{index}.png"
        Image.new("RGB", size, color).save(path)
        image_paths.append(str(path))

    rows = []
    for index in range(24):
        rows.append(
            {
                "prompt": [{"role": "user", "content": "Describe the colored image."}],
                "chosen": [
                    {"role": "assistant", "content": "It is a colored rectangle."}
                ],
                "rejected": [{"role": "assistant", "content": "There is no image."}],
                "images": [
                    image_paths[index % len(image_paths)],
                    *(
                        [image_paths[(index + 1) % len(image_paths)]]
                        if index % 5 == 0
                        else []
                    ),
                ],
            }
        )
    dataset_path = tmp_path / "vlm_dpo.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    config = {
        "project_name": "e2e_lfm3_vlm_dpo",
        "model_name": "LiquidAI/LFM3-VL-2B",
        "training_type": "vlm_dpo",
        "dataset": {
            "path": str(dataset_path),
            "type": "vlm_dpo",
            "limit": len(rows),
            "test_size": 0.25,
        },
        "training_config": {
            "extends": "DEFAULT_VLM_DPO",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-6,
            "max_image_tokens": 256,
            "max_length": 1024,
            "logging_steps": 1,
            "save_strategy": "no",
            "eval_strategy": "epoch",
        },
        "peft_config": (
            {"extends": "DEFAULT_VLM_LORA", "use_peft": True}
            if use_peft
            else {"use_peft": False}
        ),
    }
    config_path = tmp_path / f"lfm3_vlm_dpo_{use_peft}.yaml"
    config_path.write_text(__import__("yaml").safe_dump(config))
    return str(config_path)


class TestLFM3VLMDPO:
    @pytest.mark.parametrize("use_peft", [True, False], ids=["lora", "full"])
    @requires_gpu
    def test_training_completes(self, use_peft, e2e_output_dir, tmp_path):
        config_path = _write_lfm3_vlm_dpo_config(tmp_path, use_peft=use_peft)
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, max_eval_loss=7.0, check_loss_trend=False)


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
