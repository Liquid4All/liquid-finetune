import pytest
from PIL import Image
from transformers.models.lfm2_vl.image_processing_lfm2_vl_fast import (
    Lfm2VlImageProcessorFast,
)

from leap_finetune.data_loading.lfm2_vl_padding import pad_to_batch_max


def test_upsampled_images_pad_to_batch_max():
    processor = Lfm2VlImageProcessorFast(min_image_tokens=256, max_image_tokens=256)
    images = [Image.new("RGB", (300, 200)), Image.new("RGB", (256, 256))]
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        processor([[image] for image in images], return_tensors="pt")
    with pad_to_batch_max(processor, images):
        out = processor([[image] for image in images], return_tensors="pt")
    assert out["pixel_values"].shape[1] == 1120
    assert out["pixel_attention_mask"].sum(dim=1).tolist() == [1120, 1024]


def test_default_bounds_keep_fixed_padding():
    processor = Lfm2VlImageProcessorFast()
    images = [Image.new("RGB", (300, 200)), Image.new("RGB", (64, 64))]
    with pad_to_batch_max(processor, images):
        out = processor([[image] for image in images], return_tensors="pt")
    assert out["pixel_values"].shape[1] == 1024
