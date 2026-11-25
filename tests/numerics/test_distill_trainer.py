import os
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from trl.experimental.gkd import GKDConfig, GKDTrainer  # noqa: E402

from liquid_finetune.training.distill import (  # noqa: E402
    LFMDistillDataCollator,
    LFMDistillTrainer,
)
from liquid_finetune.training.utils.distillation import (  # noqa: E402
    IGNORE_INDEX,
    freeze_teacher_model,
    prepare_fsdp_teacher,
    report_vocab_size,
    resolve_shared_vocab_size,
)

VOCAB_SIZE = 32
PAD, BOS, EOS = 0, 1, 2


def _tokenizer(rename: dict[str, str] | None = None) -> PreTrainedTokenizerFast:
    vocab = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<unk>": 3}
    vocab.update({f"w{index}": index + 4 for index in range(VOCAB_SIZE - 4)})
    for old, new in (rename or {}).items():
        vocab[new] = vocab.pop(old)
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def _model(hidden_size: int, *, vocab_size: int = VOCAB_SIZE, seed: int = 0):
    torch.manual_seed(seed)
    config = Lfm2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        block_auto_adjust_ff_dim=False,
        layer_types=["full_attention"],
        pad_token_id=PAD,
        bos_token_id=BOS,
        eos_token_id=EOS,
    )
    return Lfm2ForCausalLM(config)


def _rows(count: int = 8) -> list[dict]:
    """Pretokenized SFT rows: BOS + prompt, then a supervised completion + EOS."""
    rows = []
    for index in range(count):
        prompt = [4 + (index + offset) % 28 for offset in range(3 + index % 3)]
        completion = [4 + (2 * index + offset) % 28 for offset in range(4)] + [EOS]
        input_ids = [BOS, *prompt, *completion]
        mask = [0] * (1 + len(prompt)) + [1] * len(completion)
        rows.append(
            {"input_ids": input_ids, "assistant_masks": mask, "length": len(input_ids)}
        )
    return rows


def _trainer(tmp_path, *, teacher_vocab_size=VOCAB_SIZE, loss_top_k=None, **config):
    args = GKDConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=3,
        learning_rate=5e-2,
        report_to=[],
        use_cpu=True,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        max_new_tokens=3,
        seed=0,
        **config,
    )
    teacher = freeze_teacher_model(_model(32, vocab_size=teacher_vocab_size, seed=1))
    collator = LFMDistillDataCollator(PAD, return_prompts=args.lmbda > 0 or args.seq_kd)
    return LFMDistillTrainer(
        model=_model(16),
        teacher_model=teacher,
        args=args,
        data_collator=collator,
        train_dataset=Dataset.from_list(_rows()),
        processing_class=_tokenizer(),
        loss_top_k=loss_top_k,
    )


# === Collation ===


def test_collator_masks_labels_and_left_pads_final_turn_prompts():
    rows = [
        {"input_ids": [1, 5, 6, 7, 8, 9, 2], "assistant_masks": [0, 0, 1, 0, 0, 1, 1]},
        {"input_ids": [1, 5, 6, 2], "assistant_masks": [0, 0, 1, 1]},
    ]

    batch = LFMDistillDataCollator(PAD, return_prompts=True)(rows)

    assert batch["labels"].tolist() == [
        [IGNORE_INDEX, IGNORE_INDEX, 6, IGNORE_INDEX, IGNORE_INDEX, 9, 2],
        [IGNORE_INDEX, IGNORE_INDEX, 6, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
    ]
    # Each prompt ends where the final supervised span starts.
    assert batch["prompts"].tolist() == [[1, 5, 6, 7, 8], [PAD, PAD, PAD, 1, 5]]
    assert batch["prompt_attention_mask"].tolist() == [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ]


def test_collator_skips_prompts_for_off_policy_runs():
    batch = LFMDistillDataCollator(PAD)(_rows(2))
    assert "prompts" not in batch
    assert set(batch) == {"input_ids", "attention_mask", "labels"}


def test_collator_uses_full_row_when_completion_was_truncated():
    batch = LFMDistillDataCollator(PAD, return_prompts=True)(
        [{"input_ids": [1, 5, 6], "assistant_masks": [0, 0, 0]}]
    )
    assert batch["prompts"].tolist() == [[1, 5, 6]]


def test_collator_rejects_rows_without_a_prompt():
    collator = LFMDistillDataCollator(PAD, return_prompts=True)
    with pytest.raises(ValueError, match="assistant_only_loss"):
        collator([{"input_ids": [1, 5, 6]}])


# === Teacher validation and preparation ===


def test_shared_vocabulary_allows_padded_embeddings():
    tokenizer = _tokenizer()
    assert (
        resolve_shared_vocab_size(
            tokenizer, _tokenizer(), student_vocab_size=32, teacher_vocab_size=48
        )
        == 32
    )


def test_shared_vocabulary_rejects_different_token_ids():
    with pytest.raises(ValueError, match="share a tokenizer"):
        resolve_shared_vocab_size(
            _tokenizer(),
            _tokenizer(rename={"w7": "other"}),
            student_vocab_size=VOCAB_SIZE,
            teacher_vocab_size=VOCAB_SIZE,
        )


def test_shared_vocabulary_rejects_models_smaller_than_tokenizer():
    with pytest.raises(ValueError, match="teacher model has vocab_size 16"):
        resolve_shared_vocab_size(
            _tokenizer(), _tokenizer(), student_vocab_size=32, teacher_vocab_size=16
        )


def test_report_vocab_size_restores_config():
    model = _model(16)
    with report_vocab_size(model, 99):
        assert model.config.vocab_size == 99
    assert model.config.vocab_size == VOCAB_SIZE


def test_freeze_teacher_model_disables_gradients():
    teacher = freeze_teacher_model(_model(16).train())
    assert not teacher.training
    assert not any(parameter.requires_grad for parameter in teacher.parameters())


def test_prepare_fsdp_teacher_undoes_evaluation_prepare(monkeypatch):
    import trl.models

    teacher = _model(16)
    stock_forward = teacher.forward
    teacher._original_forward = stock_forward
    teacher.forward = lambda *args, **kwargs: None
    other = object()
    accelerator = SimpleNamespace(_models=[other, teacher])
    monkeypatch.setattr(
        trl.models, "prepare_fsdp", lambda model, acc: ("sharded", model, acc)
    )

    prepared = prepare_fsdp_teacher(accelerator, teacher)

    assert prepared == ("sharded", teacher, accelerator)
    assert accelerator._models == [other]
    assert teacher.forward == stock_forward
    assert "_original_forward" not in teacher.__dict__


# === Trainer ===


def test_globally_normalized_loss_is_scaled_by_world_size():
    trainer = LFMDistillTrainer.__new__(LFMDistillTrainer)
    trainer.args = SimpleNamespace(average_tokens_across_devices=True)
    trainer.accelerator = SimpleNamespace(num_processes=4)
    loss = torch.tensor(1.5)

    assert trainer._scale_for_data_parallel(loss, torch.tensor(10)).item() == 6.0
    assert trainer._scale_for_data_parallel(loss, None).item() == 1.5
    trainer.args.average_tokens_across_devices = False
    assert trainer._scale_for_data_parallel(loss, torch.tensor(10)).item() == 1.5


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_full_vocabulary_loss_matches_gkd_trainer(tmp_path, beta):
    trainer = _trainer(tmp_path, beta=beta)
    batch = trainer._prepare_inputs(trainer.data_collator(_rows(4)))
    num_items = (batch["labels"][:, 1:] != IGNORE_INDEX).sum()

    for items in (None, num_items):
        expected = GKDTrainer.compute_loss(
            trainer, trainer.model, dict(batch), num_items_in_batch=items
        )
        actual = trainer.compute_loss(
            trainer.model, dict(batch), num_items_in_batch=items
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"beta": 0.0}, id="forward-kl"),
        pytest.param({"beta": 1.0, "loss_top_k": 4}, id="reverse-kl-top-k"),
        pytest.param({"beta": 0.5, "loss_top_k": 4, "lmbda": 1.0}, id="on-policy-jsd"),
        pytest.param({"beta": 0.0, "teacher_vocab_size": 40}, id="padded-teacher"),
    ],
)
def test_tiny_models_train_and_leave_teacher_frozen(tmp_path, config):
    trainer = _trainer(tmp_path, **config)
    student_before = [p.detach().clone() for p in trainer.model.parameters()]
    teacher_before = [p.detach().clone() for p in trainer.teacher_model.parameters()]

    result = trainer.train()

    assert result.global_step == 3
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(student_before, trainer.model.parameters())
    )
    for before, after in zip(teacher_before, trainer.teacher_model.parameters()):
        assert torch.equal(before, after)
        assert not after.requires_grad
