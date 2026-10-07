import torch
from liquid_finetune.data_loading.tokenize_data import (
    _final_assistant_span_mask,
    tokenize_and_pack_sft,
    tokenize_dpo,
    tokenize_dpo_dataset,
    tokenize_sft,
)
from liquid_finetune.training.sft import build_sft_data_collator


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        truncation=True,
        max_length=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        tools=None,
    ):
        del messages, tokenize, truncation, max_length, tools
        output = {"input_ids": [10, 11, 12, 13, 14, 15]}
        if return_assistant_tokens_mask:
            output["assistant_masks"] = [0, 1, 1, 0, 1, 1]
        return output if return_dict else output["input_ids"]

    def __call__(self, text, truncation=True, max_length=None):
        del truncation, max_length
        return {"input_ids": [1, 2, 3] if text else []}


def test_final_assistant_span_mask_keeps_only_last_assistant_segment():
    mask = [0, 1, 1, 0, 1, 1, 1, 0]
    assert _final_assistant_span_mask(mask) == [0, 0, 0, 0, 1, 1, 1, 0]


def test_tokenize_sft_emits_assistant_masks():
    tokenizer = _FakeTokenizer()
    row = {"messages": [{"role": "user", "content": "x"}]}

    tokenized = tokenize_sft(
        row,
        tokenizer,
        max_length=128,
        assistant_only_loss=True,
    )

    assert tokenized["input_ids"] == [10, 11, 12, 13, 14, 15]
    assert tokenized["assistant_masks"] == [0, 1, 1, 0, 1, 1]


def test_tokenize_sft_emits_completion_mask_for_final_assistant_span():
    tokenizer = _FakeTokenizer()
    row = {"messages": [{"role": "user", "content": "x"}]}

    tokenized = tokenize_sft(
        row,
        tokenizer,
        max_length=128,
        completion_only_loss=True,
    )

    assert tokenized["completion_mask"] == [0, 0, 0, 0, 1, 1]


def test_tokenize_sft_rejects_text_rows_for_masked_loss():
    tokenizer = _FakeTokenizer()
    row = {"text": "plain text"}

    try:
        tokenize_sft(
            row,
            tokenizer,
            max_length=128,
            assistant_only_loss=True,
        )
    except ValueError as exc:
        assert "require conversational SFT rows" in str(exc)
    else:
        raise AssertionError("Expected tokenize_sft to reject text rows")


class _NoGenerationMarkerTokenizer(_FakeTokenizer):
    """A template without {% generation %}: transformers warns and returns zeros."""

    def __init__(self, length: int = 6) -> None:
        self._length = length

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        truncation=True,
        max_length=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        tools=None,
    ):
        del messages, tokenize, truncation, max_length, tools
        output = {"input_ids": list(range(10, 10 + self._length))}
        if return_assistant_tokens_mask:
            output["assistant_masks"] = [0] * self._length
        return output if return_dict else output["input_ids"]


def test_tokenize_sft_rejects_all_zero_assistant_mask():
    """Without this the run supervises no tokens and the loss still looks fine."""
    tokenizer = _NoGenerationMarkerTokenizer()
    row = {"messages": [{"role": "user", "content": "x"}]}

    try:
        tokenize_sft(row, tokenizer, max_length=128, assistant_only_loss=True)
    except ValueError as exc:
        assert "all zeros" in str(exc)
    else:
        raise AssertionError("Expected tokenize_sft to reject an all-zero mask")


def test_tokenize_sft_rejects_all_zero_completion_mask():
    tokenizer = _NoGenerationMarkerTokenizer()
    row = {"messages": [{"role": "user", "content": "x"}]}

    try:
        tokenize_sft(row, tokenizer, max_length=128, completion_only_loss=True)
    except ValueError as exc:
        assert "all zeros" in str(exc)
    else:
        raise AssertionError("Expected tokenize_sft to reject an all-zero mask")


def test_tokenize_sft_allows_all_zero_mask_when_row_was_truncated():
    """Truncation can legitimately cut the assistant span off a long row."""
    tokenizer = _NoGenerationMarkerTokenizer(length=4)
    row = {"messages": [{"role": "user", "content": "x"}]}

    tokenized = tokenize_sft(row, tokenizer, max_length=4, assistant_only_loss=True)

    assert tokenized["assistant_masks"] == [0, 0, 0, 0]


def test_collator_applies_intersection_of_completion_and_assistant_masks():
    tokenizer = _FakeTokenizer()
    collator = build_sft_data_collator(
        tokenizer,
        {
            "assistant_only_loss": True,
            "completion_only_loss": True,
            "padding_free": False,
        },
    )

    batch = collator(
        [
            {
                "input_ids": [10, 11, 12, 13, 14, 15],
                "assistant_masks": [0, 1, 1, 0, 1, 1],
                "completion_mask": [0, 0, 0, 0, 1, 1],
            }
        ]
    )

    assert torch.equal(
        batch["labels"],
        torch.tensor([[-100, -100, -100, -100, 14, 15]]),
    )


def test_collator_applies_masks_in_padding_free_mode():
    tokenizer = _FakeTokenizer()
    collator = build_sft_data_collator(
        tokenizer,
        {
            "assistant_only_loss": True,
            "completion_only_loss": True,
            "padding_free": True,
        },
    )

    batch = collator(
        [
            {
                "input_ids": [10, 11, 12, 13],
                "assistant_masks": [0, 1, 1, 0],
                "completion_mask": [0, 1, 1, 0],
            },
            {
                "input_ids": [20, 21],
                "assistant_masks": [1, 1],
                "completion_mask": [0, 1],
            },
        ]
    )

    assert torch.equal(
        batch["labels"],
        torch.tensor([[-100, 11, 12, -100, -100, 21]]),
    )


class _PackingTokenizer(_FakeTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        truncation=True,
        max_length=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        tools=None,
    ):
        del tokenize, truncation, max_length, tools
        key = messages[0]["content"]
        if key == "one":
            output = {"input_ids": [1, 2, 3]}
            if return_assistant_tokens_mask:
                output["assistant_masks"] = [0, 1, 1]
        elif key == "two":
            output = {"input_ids": [4, 5]}
            if return_assistant_tokens_mask:
                output["assistant_masks"] = [0, 1]
        else:
            raise AssertionError(f"Unexpected key: {key}")
        return output if return_dict else output["input_ids"]


class _FakeRayDataset:
    def __init__(self, rows):
        self._rows = rows

    def map(self, fn, fn_kwargs=None):
        fn_kwargs = fn_kwargs or {}
        return _FakeRayDataset([fn(row, **fn_kwargs) for row in self._rows])

    def filter(self, fn):
        return _FakeRayDataset([row for row in self._rows if fn(row)])

    def iter_rows(self):
        yield from self._rows


class _FakePackedDataset:
    def __init__(self, rows):
        self._rows = rows

    def iter_rows(self):
        yield from self._rows


def test_tokenize_and_pack_sft_preserves_assistant_masks(monkeypatch):
    tokenizer = _PackingTokenizer()
    ds = _FakeRayDataset(
        [
            {"messages": [{"role": "user", "content": "one"}]},
            {"messages": [{"role": "user", "content": "two"}]},
        ]
    )

    monkeypatch.setattr(
        "liquid_finetune.data_loading.tokenize_data.ray.data.from_arrow",
        lambda table: _FakePackedDataset(table.to_pylist()),
    )

    packed = tokenize_and_pack_sft(
        ds,
        tokenizer,
        max_length=8,
        packing=True,
        assistant_only_loss=True,
    )

    rows = list(packed.iter_rows())
    assert len(rows) == 1
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["assistant_masks"] == [0, 1, 1, 0, 1]


class _VariableLengthTokenizer(_FakeTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        truncation=True,
        max_length=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        tools=None,
    ):
        del tokenize, tools
        length = int(messages[0]["content"])
        input_ids = list(range(length))
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        output = {"input_ids": input_ids}
        if return_assistant_tokens_mask:
            output["assistant_masks"] = [1] * len(input_ids)
        return output if return_dict else output["input_ids"]


def test_tokenize_and_pack_sft_can_drop_overlength_without_truncating():
    tokenizer = _VariableLengthTokenizer()
    ds = _FakeRayDataset(
        [
            {"messages": [{"role": "user", "content": "3"}]},
            {"messages": [{"role": "user", "content": "6"}]},
        ]
    )

    filtered = tokenize_and_pack_sft(
        ds,
        tokenizer,
        max_length=4,
        packing=False,
        assistant_only_loss=True,
        drop_overlength=True,
    )

    rows = list(filtered.iter_rows())
    assert len(rows) == 1
    assert rows[0]["input_ids"] == [0, 1, 2]
    assert rows[0]["assistant_masks"] == [1, 1, 1]


def test_tokenize_dpo_emits_collated_pair_length(monkeypatch):
    import liquid_finetune.data_loading.tokenize_data as tokenize_data

    monkeypatch.setattr(tokenize_data, "maybe_extract_prompt", lambda row: row)
    monkeypatch.setattr(
        tokenize_data, "maybe_apply_chat_template", lambda row, tokenizer: row
    )

    class _DpoTokenizer:
        eos_token_id = 99

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text)))}

    tokenized = tokenize_dpo(
        {"prompt": "p", "chosen": "ccc", "rejected": "rrrr"},
        _DpoTokenizer(),
        None,
        None,
    )

    assert tokenized["length"] == 6


def test_tokenize_dpo_dataset_stays_lazy():
    class _LazyDpoDataset:
        def __init__(self):
            self.mapped = False

        def map(self, fn, fn_kwargs=None):
            self.mapped = True
            assert fn is tokenize_dpo
            assert fn_kwargs["max_prompt_length"] == 32
            return self

        def to_arrow_refs(self):
            raise AssertionError("DPO tokenization must not gather the full dataset")

    class _DpoTokenizer:
        eos_token_id = 99

    dataset = _LazyDpoDataset()
    result = tokenize_dpo_dataset(
        dataset,
        _DpoTokenizer(),
        max_prompt_length=32,
        max_completion_length=64,
    )

    assert result is dataset
    assert dataset.mapped is True
