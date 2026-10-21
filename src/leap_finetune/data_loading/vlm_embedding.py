from __future__ import annotations

import copy
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import torch

from leap_finetune.data_loading.image_loader import load_image


@lru_cache(maxsize=8)
def _load_reference_dataset(path: str):
    from datasets import load_from_disk

    return load_from_disk(path, keep_in_memory=False)


def _load_referenced_image(reference: dict[str, Any]) -> Any:
    dataset_path = str(reference["dataset_path"])
    row_index = int(reference["row_index"])
    column = str(reference["column"])
    dataset = _load_reference_dataset(dataset_path)
    if row_index < 0 or row_index >= len(dataset):
        raise IndexError(f"image_ref row {row_index} is outside {dataset_path}")
    row = dataset[row_index]
    if column not in row:
        raise KeyError(f"image_ref column {column!r} is absent from {dataset_path}")
    return row[column]


def _messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instruction = item.get("instruction")
    if instruction:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}],
            }
        )
    content: list[dict[str, Any]] = []
    image = item.get("image")
    if image is not None:
        content.append({"type": "image", "image": image})
    text = item.get("text")
    if text is not None:
        content.append({"type": "text", "text": text})
    messages.append({"role": "user", "content": content})
    return messages


def create_vlm_embedding_collate_fn(processor):
    """Collate three assistant-prefix VLM batches with explicit left padding."""

    processor.tokenizer.padding_side = "left"

    def encode_role(samples: list[dict[str, Any]], role: str):
        conversations = []
        loaded_images = []
        try:
            for sample in samples:
                item = copy.deepcopy(sample[role])
                image_reference = item.pop("image_ref", None)
                if image_reference is not None:
                    image = _load_referenced_image(image_reference)
                    item["image"] = image
                    loaded_images.append(image)
                elif isinstance(item.get("image"), str):
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

    Retrieval rows use in-batch candidates from the same candidate group.
    Fixed-ontology classification rows use all label candidates from the same
    ontology. Multiple-choice rows use only their explicit choices. Duplicate
    positives are masked rather than treated as false negatives.
    """

    size = len(samples)
    group_tokens: list[str | None] = []
    positive_keys: list[str] = []
    negative_keys: list[str] = []
    for query_index, sample in enumerate(samples):
        loss_kind = sample.get("loss_kind", "retrieval")
        if loss_kind == "explicit":
            loss_kind = "multiple_choice"
        if loss_kind == "multiple_choice":
            group_tokens.append(None)
            positive_keys.append(
                str(sample.get("positive_key", f"positive:{query_index}"))
            )
            negative_keys.append(
                str(sample.get("negative_key", f"negative:{query_index}"))
            )
            continue
        if loss_kind == "classification":
            group_field = "candidate_set_id"
        elif loss_kind == "retrieval":
            group_field = "candidate_group"
        else:
            raise ValueError(f"unsupported loss_kind: {loss_kind!r}")
        group = sample.get(group_field)
        if loss_kind == "retrieval" and not group:
            group = sample.get("loss_group") or sample.get("modality") or "legacy"
        if not group:
            raise ValueError(f"{loss_kind} rows require {group_field}")
        group_tokens.append(f"{loss_kind}:{group}")
        positive_keys.append(str(sample.get("positive_key", f"positive:{query_index}")))
        negative_keys.append(str(sample.get("negative_key", f"negative:{query_index}")))

    def integer_ids(values: Sequence[str | None]) -> torch.Tensor:
        lookup: dict[str | None, int] = {}
        return torch.tensor(
            [lookup.setdefault(value, len(lookup)) for value in values],
            dtype=torch.int64,
        )

    group_ids = integer_ids(group_tokens)
    candidate_ids = integer_ids([*positive_keys, *negative_keys])
    positive_ids = candidate_ids[:size]
    negative_ids = candidate_ids[size:]
    same_group = group_ids[:, None].eq(group_ids[None, :])
    explicit = torch.tensor([value is None for value in group_tokens], dtype=torch.bool)
    same_group &= ~explicit[:, None]
    identity = torch.eye(size, dtype=torch.bool)
    positive_mask = same_group & (
        ~positive_ids[:, None].eq(positive_ids[None, :]) | identity
    )
    negative_mask = same_group & ~positive_ids[:, None].eq(negative_ids[None, :])
    positive_mask |= identity
    negative_mask |= identity
    mask = torch.cat((positive_mask, negative_mask), dim=1)
    return mask


def collate_raw_vlm_embedding_samples(
    samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep a logical batch raw so cached loss can tokenize bounded chunks."""

    if not samples:
        raise ValueError("cannot collate an empty VLM embedding batch")
    return {"samples": samples}


__all__ = [
    "build_candidate_mask",
    "collate_raw_vlm_embedding_samples",
    "create_vlm_embedding_collate_fn",
]
