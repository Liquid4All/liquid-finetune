"""Pad LFM2-VL image batches to their true maximum patch count.

``Lfm2VlImageProcessorFast.smart_resize`` rounds both sides up when it
upsamples an image to ``min_image_tokens``, which can exceed
``max_image_tokens``; ``_preprocess`` still pads every image group to the
patch count implied by ``max_image_tokens``, so a mixed-shape batch fails to
concatenate. Single images (as served by vLLM) keep the overshoot, so the
batch is padded to the largest image instead of clamping it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import transformers.models.lfm2_vl.image_processing_lfm2_vl_fast as lfm2_vl_fast

_PAD_TARGET: ContextVar[int] = ContextVar("lfm2_vl_pad_target", default=0)
_ORIGINAL_PAD = lfm2_vl_fast.pad_along_first_dim


def _pad_to_batch_max(images, target_length: int, pad_value: int = 0):
    return _ORIGINAL_PAD(images, max(target_length, _PAD_TARGET.get()), pad_value)


lfm2_vl_fast.pad_along_first_dim = _pad_to_batch_max


def max_patches(image_processor: Any, sizes: Iterable[tuple[int, int]]) -> int:
    """Largest single-crop patch count the processor produces for these sizes."""

    patch = int(image_processor.encoder_patch_size)
    largest = 0
    for height, width in sizes:
        new_width, new_height = image_processor.smart_resize(
            height=height,
            width=width,
            downsample_factor=int(image_processor.downsample_factor),
            min_image_tokens=int(image_processor.min_image_tokens),
            max_image_tokens=int(image_processor.max_image_tokens),
            encoder_patch_size=patch,
        )
        largest = max(largest, (new_height // patch) * (new_width // patch))
    return largest


@contextmanager
def pad_to_batch_max(image_processor: Any, images: Iterable[Any]) -> Iterator[None]:
    """Pad every image group in this processor call to the batch's largest image."""

    sizes = [(image.height, image.width) for image in images]
    lfm2_vl = isinstance(image_processor, lfm2_vl_fast.Lfm2VlImageProcessorFast)
    target = max_patches(image_processor, sizes) if sizes and lfm2_vl else 0
    token = _PAD_TARGET.set(target)
    try:
        yield
    finally:
        _PAD_TARGET.reset(token)
