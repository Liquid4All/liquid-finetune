from types import SimpleNamespace


from datasets import Dataset
import pytest
import torch
from PIL import Image
from leap_finetune.data_loading.length_grouping import get_tile_count_grouped_sampler
from leap_finetune.data_loading.vlm_batching import (
    add_vlm_tile_counts,
    estimate_vlm_tile_count,
)

from leap_finetune.data_loading import image_loader
from leap_finetune.data_loading.tokenize_data import create_vlm_collate_fn


@pytest.fixture(autouse=True)
def clean_image_cache(monkeypatch):
    monkeypatch.delenv("LEAP_IMAGE_CACHE_MAX_BYTES", raising=False)
    image_loader.clear_image_cache()
    yield
    image_loader.clear_image_cache()


def _write_image(path, size=(10, 10)):
    Image.new("RGB", size, color=(1, 2, 3)).save(path)


def test_image_cache_returns_closeable_copies(tmp_path, monkeypatch):
    image_path = tmp_path / "cached.png"
    _write_image(image_path, size=(8, 8))
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


def test_image_cache_evicts_oldest_entries_by_decoded_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAP_IMAGE_CACHE_MAX_BYTES", "600")
    paths = [tmp_path / f"image-{index}.png" for index in range(3)]
    for path in paths:
        _write_image(path)

    closed_ids = []
    original_close = Image.Image.close

    def tracking_close(image):
        closed_ids.append(id(image))
        original_close(image)

    monkeypatch.setattr(Image.Image, "close", tracking_close)
    for path in paths:
        image_loader.load_image(path).close()

    assert len(image_loader._IMAGE_CACHE) == 2
    assert image_loader._IMAGE_CACHE_BYTES == 600
    assert id(next(iter(image_loader._IMAGE_CACHE.values()))) not in closed_ids

    oldest_cached = next(iter(image_loader._IMAGE_CACHE.values()))
    image_loader.load_image(paths[0]).close()
    assert id(oldest_cached) in closed_ids


def test_image_cache_does_not_retain_image_larger_than_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAP_IMAGE_CACHE_MAX_BYTES", "299")
    image_path = tmp_path / "too-large.png"
    _write_image(image_path)
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
    second.close()

    assert open_count == 2
    assert not image_loader._IMAGE_CACHE
    assert image_loader._IMAGE_CACHE_BYTES == 0


def test_image_loadability_does_not_populate_cache(tmp_path):
    image_path = tmp_path / "validated.png"
    _write_image(image_path)

    assert image_loader.is_image_loadable(str(image_path)) is True
    assert not image_loader._IMAGE_CACHE
    assert image_loader._IMAGE_CACHE_BYTES == 0


def test_zero_image_cache_budget_disables_retention(tmp_path, monkeypatch):
    image_path = tmp_path / "uncached.png"
    _write_image(image_path)
    monkeypatch.setenv("LEAP_IMAGE_CACHE_MAX_BYTES", "0")
    original_open = image_loader.Image.open
    open_count = 0

    def counting_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(image_loader.Image, "open", counting_open)
    image_loader.load_image(image_path).close()
    image_loader.load_image(image_path).close()

    assert open_count == 2
    assert not image_loader._IMAGE_CACHE


def test_image_cache_keeps_item_limit(tmp_path):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LEAP_IMAGE_CACHE_MAX_BYTES", "1000000")
    try:
        paths = [tmp_path / f"image-{index}.png" for index in range(33)]
        for path in paths:
            _write_image(path, size=(1, 1))
        for path in paths:
            image_loader.load_image(path).close()

        assert len(image_loader._IMAGE_CACHE) == 32
        assert paths[0].as_posix() not in {key[0] for key in image_loader._IMAGE_CACHE}
    finally:
        monkeypatch.undo()


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [1, 2]

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 3


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def __init__(self):
        self.images_seen = []

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        self.images_seen.append(messages[0][0]["content"][0]["image"])
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def test_vlm_collator_closes_batch_image_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAP_IMAGE_CACHE_MAX_BYTES", "0")
    image_path = tmp_path / "batch.png"
    _write_image(image_path)
    processor = _FakeProcessor()
    collate = create_vlm_collate_fn(processor)
    sample = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "describe"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
    }

    batch = collate([sample])

    assert batch["labels"].tolist() == [[-100, -100, 3]]
    with pytest.raises(ValueError, match="closed image"):
        processor.images_seen[0].getpixel((0, 0))


def test_load_image_closes_http_response(tmp_path, monkeypatch):
    image_path = tmp_path / "remote.png"
    _write_image(image_path)

    class Response:
        def __init__(self, raw):
            self.raw = raw
            self.closed = False

        def raise_for_status(self):
            pass

        def close(self):
            self.closed = True
            self.raw.close()

    response = Response(image_path.open("rb"))
    monkeypatch.setattr(image_loader.requests, "get", lambda *args, **kwargs: response)

    image = image_loader.load_image("https://example.test/image.png")
    image.close()

    assert response.closed is True


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
