import os

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from trl.experimental.distillation import (  # noqa: E402
    DistillationConfig,
    DistillationTrainer,
)

from liquid_finetune.training.opd import LFMOPDTrainer, to_prompt_dataset  # noqa: E402
from liquid_finetune.training.utils.distillation import (  # noqa: E402
    conversation_prompt,
    freeze_teacher_model,
)

VOCAB_SIZE = 32
PAD, BOS, EOS = 0, 1, 2
# Minimal template over whitespace-separated vocabulary words.
CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}w0 {% endif %}"
)


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<unk>": 3}
    vocab.update({f"w{index}": index + 4 for index in range(VOCAB_SIZE - 4)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


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


def _prompts(count: int = 8) -> Dataset:
    rows = [
        {
            "prompt": " ".join(
                f"w{(index + offset) % 28}" for offset in range(2 + index % 3)
            )
        }
        for index in range(count)
    ]
    return to_prompt_dataset(Dataset.from_list(rows))


def _trainer(tmp_path, *, teacher_vocab_size=VOCAB_SIZE, loss_top_k=None, **config):
    config.setdefault("beta", 1.0)
    args = DistillationConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=3,
        learning_rate=5e-2,
        report_to=[],
        use_cpu=True,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        max_length=32,
        max_completion_length=4,
        top_p=1.0,
        seed=0,
        **config,
    )
    return LFMOPDTrainer(
        model=_model(16),
        teacher_model=freeze_teacher_model(
            _model(32, vocab_size=teacher_vocab_size, seed=1)
        ),
        args=args,
        train_dataset=_prompts(),
        processing_class=_tokenizer(),
        loss_top_k=loss_top_k,
    )


# === Prompts ===


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"prompt": "w1 w2"}, [{"role": "user", "content": "w1 w2"}]),
        (
            {
                "prompt": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                ]
            },
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]
            },
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ],
        ),
        (
            {"messages": [{"role": "user", "content": "q"}]},
            [{"role": "user", "content": "q"}],
        ),
    ],
)
def test_conversation_prompt(row, expected):
    assert conversation_prompt(row) == expected


@pytest.mark.parametrize(
    "row",
    [
        {"prompt": " "},
        {"prompt": []},
        {"messages": [{"role": "assistant", "content": "a"}]},
        {},
    ],
)
def test_conversation_prompt_rejects_rows_without_a_prompt(row):
    with pytest.raises(ValueError, match="prompt"):
        conversation_prompt(row)


def test_prompt_dataset_drops_final_assistant_turn_and_messages_column():
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                "source": "x",
            }
        ]
    )
    converted = to_prompt_dataset(dataset)
    assert converted.column_names == ["source", "prompt"]
    assert converted[0]["prompt"] == [{"role": "user", "content": "q"}]


# === Trainer ===


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_top_k_path_with_full_support_matches_trl_loss(tmp_path, beta):
    trainer = _trainer(tmp_path, beta=beta, loss_top_k=4)
    trainer.model.eval()
    rows = [
        {
            "messages": [
                {"role": "user", "content": "w1 w2 w3"[: 2 + 3 * index]},
                {"role": "assistant", "content": "w4 w5 w6"},
            ]
        }
        for index in range(3)
    ]
    batch = trainer._prepare_inputs(trainer.data_collator(rows))
    expected = DistillationTrainer.compute_loss(trainer, trainer.model, dict(batch))

    # Exercise the top-k code path with a support covering the vocabulary.
    trainer.loss_top_k, trainer.loss_add_tail = VOCAB_SIZE, False
    actual = trainer.compute_loss(trainer.model, dict(batch))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({}, id="reverse-kl"),
        pytest.param({"loss_top_k": 4}, id="reverse-kl-top-k"),
        pytest.param({"beta": 0.5, "num_generations": 2}, id="jsd-two-generations"),
        pytest.param({"teacher_vocab_size": 40}, id="padded-teacher"),
        pytest.param(
            {"teacher_vocab_size": 40, "loss_top_k": 4}, id="padded-teacher-top-k"
        ),
    ],
)
def test_tiny_models_train_on_student_samples(tmp_path, config):
    trainer = _trainer(tmp_path, **config)
    student_before = [p.detach().clone() for p in trainer.model.parameters()]
    teacher_before = [p.detach().clone() for p in trainer.teacher_model.parameters()]

    result = trainer.train()

    assert result.global_step == 3
    assert torch.isfinite(torch.tensor(result.training_loss))
    logged = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert logged and all("completions/mean_length" in entry for entry in logged)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(student_before, trainer.model.parameters())
    )
    for before, after in zip(teacher_before, trainer.teacher_model.parameters()):
        assert torch.equal(before, after)
        assert not after.requires_grad
