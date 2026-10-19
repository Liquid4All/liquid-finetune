from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from PIL import Image

from leap_finetune.data_loading.vlm_embedding import (
    build_candidate_mask,
    create_vlm_embedding_collate_fn,
)
from leap_finetune.data_loading.validate_dataset_format import (
    normalize_columns,
    validate_dataset_format,
)
from leap_finetune.training.vlm_embedding import (
    contrastive_embedding_loss,
    pool_last_attended_token,
)


def test_pool_last_attended_token_supports_left_padding():
    hidden = torch.tensor(
        [
            [[9.0, 9.0], [3.0, 4.0], [0.0, 2.0]],
            [[1.0, 0.0], [5.0, 12.0], [8.0, 15.0]],
        ]
    )
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    pooled = pool_last_attended_token(hidden, mask)
    torch.testing.assert_close(pooled[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(pooled[1], torch.tensor([5.0 / 13.0, 12.0 / 13.0]))


def test_contrastive_embedding_loss_prefers_aligned_pairs():
    query = torch.eye(2)
    positive = torch.eye(2)
    negative = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
    aligned = contrastive_embedding_loss(query, positive, negative, temperature=0.1)
    swapped = contrastive_embedding_loss(
        query, positive.flip(0), negative, temperature=0.1
    )
    assert aligned < swapped
    with pytest.raises(ValueError, match="positive"):
        contrastive_embedding_loss(query, positive, negative, temperature=0)


def test_candidate_mask_uses_task_semantics_and_masks_duplicate_positives():
    samples = [
        {
            "loss_kind": "retrieval",
            "loss_group": "text",
            "positive_key": "same",
            "negative_key": "n0",
        },
        {
            "loss_kind": "retrieval",
            "loss_group": "text",
            "positive_key": "same",
            "negative_key": "n1",
        },
        {
            "loss_kind": "explicit",
            "loss_group": "text",
            "positive_key": "p2",
            "negative_key": "n2",
        },
    ]

    mask = build_candidate_mask(samples)

    assert mask.tolist() == [
        [True, False, True, True, True, True],
        [False, True, True, True, True, True],
        [False, False, True, False, False, True],
    ]


def test_contrastive_loss_honors_candidate_mask():
    query = torch.eye(2)
    positive = torch.eye(2)
    negative = torch.eye(2)
    explicit = torch.tensor([[True, False, True, False], [False, True, False, True]])

    loss = contrastive_embedding_loss(
        query,
        positive,
        negative,
        temperature=0.1,
        candidate_mask=explicit,
    )

    torch.testing.assert_close(loss, torch.tensor(0.6931472))


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
        return {
            "input_ids": torch.tensor([[0, 4, 5], [2, 3, 5]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }


def test_vlm_embedding_collator_uses_assistant_prefix_and_left_padding():
    processor = _FakeProcessor()
    collator = create_vlm_embedding_collate_fn(processor)
    item = {"instruction": "Represent it.", "text": "content"}
    batch = collator(
        [
            {"query": item, "positive": item, "negative": item},
            {"query": item, "positive": item, "negative": item},
        ]
    )
    assert processor.tokenizer.padding_side == "left"
    assert set(batch) == {"query", "positive", "negative", "candidate_mask"}
    assert len(processor.calls) == 3
    assert all(call[1]["add_generation_prompt"] for call in processor.calls)
    assert all(call[1]["padding"] for call in processor.calls)


def test_vlm_embedding_dataset_normalizes_relative_images(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "white").save(image_path)
    row = {
        "query": {"text": "question"},
        "positive": {"image": "image.png"},
        "negative": {"text": "unrelated"},
    }
    dataset = Dataset.from_list(
        [normalize_columns("vlm_embedding", image_root=str(tmp_path))(row)]
    )
    validated = validate_dataset_format(dataset, "vlm_embedding")
    assert validated[0]["positive"]["image"] == str(image_path)
    assert "instruction" in validated[0]["query"]
