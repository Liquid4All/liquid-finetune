from __future__ import annotations

import copy
from typing import Any

import torch

from leap_finetune.data_loading.image_loader import load_image


def _messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image = item.get("image")
    if image is not None:
        content.append({"type": "image", "image": image})
    text = item.get("text")
    if text is not None:
        content.append({"type": "text", "text": text})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": item["instruction"]}],
        },
        {"role": "user", "content": content},
    ]


def create_vlm_embedding_collate_fn(processor):
    """Collate three assistant-prefix VLM batches with explicit left padding."""

    processor.tokenizer.padding_side = "left"

    def encode_role(samples: list[dict[str, Any]], role: str):
        conversations = []
        loaded_images = []
        try:
            for sample in samples:
                item = copy.deepcopy(sample[role])
                if isinstance(item.get("image"), str):
                    image = load_image(item["image"])
                    item["image"] = image
                    loaded_images.append(image)
                conversations.append(_messages(item))
            return processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        finally:
            for image in loaded_images:
                if hasattr(image, "close"):
                    image.close()

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty VLM embedding batch")
        batch = {
            role: encode_role(samples, role)
            for role in ("query", "positive", "negative")
        }
        batch["candidate_mask"] = build_candidate_mask(samples)
        return batch

    return collate


def build_candidate_mask(samples: list[dict[str, Any]]) -> torch.Tensor:
    """Select semantically valid positives and negatives for each query.

    Retrieval rows use in-batch candidates from the same modality group while
    closed-set and QA rows use only their explicit positive/negative pair.
    Duplicate positives are masked rather than treated as false negatives.
    """

    size = len(samples)
    mask = torch.zeros((size, 2 * size), dtype=torch.bool)
    for query_index, sample in enumerate(samples):
        mask[query_index, query_index] = True
        mask[query_index, size + query_index] = True
        if sample.get("loss_kind", "retrieval") != "retrieval":
            continue
        group = sample.get("loss_group", "all")
        positive_key = sample.get("positive_key", f"positive:{query_index}")
        for candidate_index, candidate in enumerate(samples):
            if candidate.get("loss_group", "all") != group:
                continue
            candidate_positive = candidate.get(
                "positive_key", f"positive:{candidate_index}"
            )
            candidate_negative = candidate.get(
                "negative_key", f"negative:{candidate_index}"
            )
            if candidate_index == query_index or candidate_positive != positive_key:
                mask[query_index, candidate_index] = True
            if candidate_negative != positive_key:
                mask[query_index, size + candidate_index] = True
    return mask


__all__ = ["build_candidate_mask", "create_vlm_embedding_collate_fn"]
