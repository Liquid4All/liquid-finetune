import io
import logging
import os
from collections import OrderedDict
from threading import RLock

import requests
from PIL import Image, ImageFile

# PIL safety: prevent crashes on large/truncated images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

_IMAGE_CACHE_MAX_ITEMS = 32
_IMAGE_CACHE: OrderedDict[tuple[str, int, int], Image.Image] = OrderedDict()
_IMAGE_CACHE_LOCK = RLock()


def _load_cached_path_image(src) -> Image.Image:
    """Load a local image with a small per-process cache.

    Return a copy so callers may close their image without invalidating the
    cached object. File size and mtime invalidate entries when a path changes.
    """
    path = os.fspath(src)
    try:
        stat = os.stat(path)
    except OSError:
        with Image.open(path) as image:
            return image.convert("RGB")

    key = (path, stat.st_mtime_ns, stat.st_size)
    with _IMAGE_CACHE_LOCK:
        cached = _IMAGE_CACHE.pop(key, None)
        if cached is not None:
            _IMAGE_CACHE[key] = cached
            return cached.copy()

    with Image.open(path) as image:
        decoded = image.convert("RGB")

    with _IMAGE_CACHE_LOCK:
        for old_key in list(_IMAGE_CACHE):
            if old_key[0] == path:
                _IMAGE_CACHE.pop(old_key).close()
        _IMAGE_CACHE[key] = decoded
        while len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX_ITEMS:
            _, evicted = _IMAGE_CACHE.popitem(last=False)
            evicted.close()
        return decoded.copy()


def load_image(src) -> Image.Image:
    """Load image from various sources and return PIL Image in RGB."""
    # bytes -> PIL
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    # URL -> PIL
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        resp = requests.get(
            src, stream=True, headers={"User-Agent": "liquid-finetune"}, timeout=15
        )
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")
    # file path -> PIL
    return _load_cached_path_image(src)


def get_image_size(src) -> tuple[int, int]:
    """Read image dimensions without decoding the full local image."""
    if isinstance(src, (bytes, bytearray)):
        with Image.open(io.BytesIO(src)) as image:
            return image.size
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        resp = requests.get(
            src, stream=True, headers={"User-Agent": "liquid-finetune"}, timeout=15
        )
        try:
            resp.raise_for_status()
            with Image.open(resp.raw) as image:
                return image.size
        finally:
            resp.close()
    with Image.open(src) as image:
        return image.size


def is_image_loadable(src: str) -> bool:
    """Check if an image source can be loaded without error."""
    try:
        img = load_image(src)
        img.close()
        return True
    except Exception:
        return False
