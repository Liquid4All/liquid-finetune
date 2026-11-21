import pytest
import torch

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
from liquid_finetune.checkpointing.model_loading import load_model
from liquid_finetune.training.peft.peft import apply_peft_to_model
from liquid_finetune.training.peft.peft_configs import DEFAULT_LORA

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
    def test_checkpoint_generation_and_lora_backward(self):
        model, tokenizer = load_model("LiquidAI/LFM3-1.7B")
        model.cuda()

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Count to three."}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )["input_ids"].cuda()
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=2, do_sample=False)
        assert generated.shape[-1] > prompt.shape[-1]

        encoded = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "What is two plus two?"},
                {"role": "assistant", "content": "Four."},
            ],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[~encoded.pop("assistant_masks").bool()] = -100
        batch = {key: value.cuda() for key, value in encoded.items()}
        batch["labels"] = labels.cuda()

        model = apply_peft_to_model(model, DEFAULT_LORA)
        loss = model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )


# === Dense DPO full fine-tune ===


class TestDenseDPOFull:
    @requires_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_dpo_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, check_loss_trend=False)
        assert_eval_callback_logged(result)

        assert_checkpoints_exist(e2e_output_dir)
