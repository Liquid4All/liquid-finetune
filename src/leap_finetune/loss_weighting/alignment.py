from __future__ import annotations

import logging
from collections.abc import Sequence

from .config import LossWeightingConfig
from .selectors import select_spans

logger = logging.getLogger(__name__)


def assistant_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    return "".join(
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )


def _assistant_offset_regions(offsets, mask) -> list[tuple[int, int]]:
    """Return rendered-text bounds for contiguous supervised assistant runs."""
    regions = []
    current = []
    for token_span, supervised in zip(offsets, mask, strict=True):
        if supervised:
            if token_span[1] > token_span[0]:
                current.append(token_span)
        elif current:
            regions.append(
                (min(span[0] for span in current), max(span[1] for span in current))
            )
            current = []
    if current:
        regions.append(
            (min(span[0] for span in current), max(span[1] for span in current))
        )
    return regions


def _assistant_regions(
    messages: list[dict],
    rendered_text: str,
    offsets: list[tuple[int, int]],
    mask: list[bool],
) -> list[tuple[str, int]]:
    """Locate assistant bodies using the template's supervised token regions."""
    expected_regions = _assistant_offset_regions(offsets, mask)
    regions = []
    cursor = 0
    assistant_index = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        text = assistant_text(message)
        if not text:
            continue

        variants = [text]
        stripped = text.strip()
        if stripped and stripped != text:
            variants.append(stripped)
        candidates = []
        for variant in variants:
            start = rendered_text.find(variant, cursor)
            while start >= 0:
                candidates.append((start, variant))
                start = rendered_text.find(variant, start + 1)
        if not candidates:
            raise ValueError(
                "Could not align assistant text with the rendered chat template. "
                "Ensure preprocessing produces the exact text consumed by the template."
            )

        if assistant_index < len(expected_regions):
            expected_start, expected_end = expected_regions[assistant_index]

            def overlap(candidate):
                start, variant = candidate
                end = start + len(variant)
                return max(0, min(end, expected_end) - max(start, expected_start))

            start, text = max(candidates, key=overlap)
        else:
            start, text = candidates[0]
        regions.append((text, start))
        cursor = start + len(text)
        assistant_index += 1
    return regions


def _token_touches_span(
    token_span: tuple[int, int],
    selected_span: tuple[int, int],
    boundary: str,
) -> bool:
    token_start, token_end = token_span
    selected_start, selected_end = selected_span
    if token_end <= token_start:
        return False
    if boundary == "contained":
        return token_start >= selected_start and token_end <= selected_end
    return token_start < selected_end and token_end > selected_start


def build_token_loss_weights(
    *,
    messages: list[dict],
    rendered_text: str,
    offset_mapping: Sequence[Sequence[int]],
    assistant_mask: Sequence[int | bool],
    config: LossWeightingConfig | dict,
    diagnostics: dict | None = None,
) -> list[float]:
    """Align semantic character selectors to final model tokens.

    Tokens with any character overlap are selected by default. Rules are
    applied in order and later rules overwrite earlier ones. Tokens outside
    assistant spans always receive weight zero.
    """
    config = (
        config
        if isinstance(config, LossWeightingConfig)
        else LossWeightingConfig.model_validate(config)
    )
    offsets = [tuple(int(value) for value in pair) for pair in offset_mapping]
    mask = [bool(value) for value in assistant_mask]
    if len(offsets) != len(mask):
        raise ValueError("offset_mapping and assistant mask lengths must match")

    weights = [config.default_weight if supervised else 0.0 for supervised in mask]
    matched_rules = {rule.name: False for rule in config.rules}

    for text, rendered_start in _assistant_regions(
        messages, rendered_text, offsets, mask
    ):
        for rule in config.rules:
            local_spans = select_spans(text, rule.selector)
            if local_spans:
                matched_rules[rule.name] = True
            global_spans = [
                (rendered_start + start, rendered_start + end)
                for start, end in local_spans
            ]
            for token_index, token_span in enumerate(offsets):
                if not mask[token_index]:
                    continue
                if any(
                    _token_touches_span(
                        token_span, selected_span, config.token_boundary
                    )
                    for selected_span in global_spans
                ):
                    weights[token_index] = rule.weight

    unmatched = [name for name, matched in matched_rules.items() if not matched]
    if unmatched:
        logger.debug("Loss-weight rules unmatched for row: %s", ", ".join(unmatched))

    positive_count = sum(weight > 0 for weight in weights)
    if diagnostics is not None:
        diagnostics.update(
            weighted_token_count=positive_count,
            effective_weight_sum=float(sum(weights)),
            rule_matches=dict(matched_rules),
        )

    if positive_count == 0:
        message = "row has no positive-weight supervised token after loss weighting"
        if config.zero_weight_action == "error":
            raise ValueError(message)
        if config.zero_weight_action == "warn":
            logger.warning(message)
    return weights
