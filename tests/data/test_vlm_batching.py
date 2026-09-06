from types import SimpleNamespace

import torch
from datasets import Dataset
from PIL import Image

from liquid_finetune.data_loading import image_loader
from liquid_finetune.data_loading.length_grouping import get_tile_count_grouped_sampler
from liquid_finetune.data_loading.vlm_batching import (
    add_vlm_tile_counts,
    estimate_vlm_tile_count,
)


class _FakeImageProcessor:
    do_image_splitting = True
    max_image_tokens = 256
    encoder_patch_size = 16
    downsample_factor = 2
    max_pixels_tolerance = 2.0
    min_tiles = 2
    max_tiles = 10
    tile_size = 512
    use_thumbnail = True

    @staticmethod
    def _is_image_too_large(height, width, **kwargs):
        del kwargs
        return height * width > 512 * 512

    @staticmethod
    def _get_grid_layout(height, width, **kwargs):
        del height, width, kwargs
        return 1, 2, 512, 1024, 2


def _processor():
    return SimpleNamespace(image_processor=_FakeImageProcessor())


def test_estimate_vlm_tile_count_handles_multi_image_rows(tmp_path):
    small = tmp_path / "small.png"
    large = tmp_path / "large.png"
    Image.new("RGB", (64, 64)).save(small)
    Image.new("RGB", (1024, 512)).save(large)
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(small)},
                    {"type": "image", "image": str(large)},
                ],
            }
        ]
    }
    assert estimate_vlm_tile_count(row, _processor()) == 4


def test_add_vlm_tile_counts_preserves_rows(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64)).save(image)
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": str(image)}],
                    }
                ]
            }
        ]
    )
    result = add_vlm_tile_counts(dataset, _processor())
    assert len(result) == len(dataset)
    assert result["_vlm_tile_count"] == [1]


def test_tile_count_sampler_covers_each_row_once():
    dataset = Dataset.from_dict({"_vlm_tile_count": [1, 1, 4, 4, 8, 8]})
    sampler = get_tile_count_grouped_sampler(
        dataset,
        batch_size=2,
        generator=torch.Generator().manual_seed(7),
    )
    assert sorted(sampler) == list(range(len(dataset)))


def test_tile_count_sampler_skips_uniform_counts():
    dataset = Dataset.from_dict({"_vlm_tile_count": [4, 4, 4]})
    assert get_tile_count_grouped_sampler(dataset, batch_size=2) is None


def test_image_cache_returns_closeable_copies(tmp_path, monkeypatch):
    image_path = tmp_path / "cached.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image_path)
    original_open = image_loader.Image.open
    open_count = 0

    def counting_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(image_loader.Image, "open", counting_open)
    first = image_loader.load_image(image_path)
    first.close()
    second = image_loader.load_image(image_path)
    assert second.getpixel((0, 0)) == (1, 2, 3)
    assert open_count == 1
    second.close()
