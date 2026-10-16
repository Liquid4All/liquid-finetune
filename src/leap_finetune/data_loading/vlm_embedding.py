from __future__ import annotations

import copy
from typing import Any

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
        return {
            role: encode_role(samples, role)
            for role in ("query", "positive", "negative")
        }

    return collate


__all__ = ["create_vlm_embedding_collate_fn"]
