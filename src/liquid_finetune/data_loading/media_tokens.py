from __future__ import annotations

from typing import Any


MEDIA_TOKEN_CANDIDATES = {
    "image": ("<|image|>", "<|image_pad|>"),
    "video": ("<|video|>", "<|video_pad|>"),
}


def resolve_media_tokens(processor: Any) -> dict[str, dict[str, set]]:
    """Resolve model-specific image/video placeholder strings and token IDs."""
    tokenizer = processor.tokenizer
    vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    resolved = {
        media_type: {"ids": set(), "tokens": set()}
        for media_type in MEDIA_TOKEN_CANDIDATES
    }

    for media_type, candidates in MEDIA_TOKEN_CANDIDATES.items():
        token_id = getattr(processor, f"{media_type}_token_id", None)
        if isinstance(token_id, int) and token_id >= 0:
            resolved[media_type]["ids"].add(token_id)
            token = (
                tokenizer.convert_ids_to_tokens(token_id)
                if hasattr(tokenizer, "convert_ids_to_tokens")
                else None
            )
            if isinstance(token, str) and token:
                resolved[media_type]["tokens"].add(token)

        for token in candidates:
            candidate_id = vocabulary.get(token)
            if (
                isinstance(candidate_id, int)
                and candidate_id >= 0
                and candidate_id != unknown_token_id
            ):
                resolved[media_type]["ids"].add(candidate_id)
                resolved[media_type]["tokens"].add(token)

    return resolved
