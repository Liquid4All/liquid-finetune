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
_DEFAULT_IMAGE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_IMAGE_CACHE_MAX_BYTES = _DEFAULT_IMAGE_CACHE_MAX_BYTES
_IMAGE_CACHE: OrderedDict[tuple[str, int, int], Image.Image] = OrderedDict()
_IMAGE_CACHE_LOCK = RLock()
_IMAGE_CACHE_BYTES = 0


def _image_cache_max_bytes() -> int:
    """Return the configured per-process decoded-image cache budget."""
    configured = os.getenv("LEAP_IMAGE_CACHE_MAX_BYTES")
    if configured is None:
        return _IMAGE_CACHE_MAX_BYTES
    try:
        return max(0, int(configured))
    except ValueError:
        logger.warning(
            "Invalid LEAP_IMAGE_CACHE_MAX_BYTES=%r; using default %d",
            configured,
            _DEFAULT_IMAGE_CACHE_MAX_BYTES,
        )
        return _IMAGE_CACHE_MAX_BYTES


def _decoded_image_bytes(image: Image.Image) -> int:
    """Estimate the decoded pixel memory held by an image."""
    return image.width * image.height * len(image.getbands())


def _evict_image_cache_entry(key) -> None:
    global _IMAGE_CACHE_BYTES
    image = _IMAGE_CACHE.pop(key)
    _IMAGE_CACHE_BYTES -= _decoded_image_bytes(image)
    image.close()


def clear_image_cache() -> None:
    """Close and remove all retained decoded images."""
    global _IMAGE_CACHE_BYTES
    with _IMAGE_CACHE_LOCK:
        for image in _IMAGE_CACHE.values():
            image.close()
        _IMAGE_CACHE.clear()
        _IMAGE_CACHE_BYTES = 0


def _load_cached_path_image(src) -> Image.Image:
    """Load a local image with bounded per-process decoded-image caching.

    Return a copy so callers may close their image without invalidating the
    cached object. File size and mtime invalidate entries when a path changes.
    """
    global _IMAGE_CACHE_BYTES

    path = os.fspath(src)
    try:
        stat = os.stat(path)
    except OSError:
        with Image.open(path) as image:
            return image.convert("RGB")

    key = (path, stat.st_mtime_ns, stat.st_size)
    max_bytes = _image_cache_max_bytes()
    with _IMAGE_CACHE_LOCK:
        if max_bytes == 0 and _IMAGE_CACHE:
            clear_image_cache()
        while _IMAGE_CACHE and (
            len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX_ITEMS or _IMAGE_CACHE_BYTES > max_bytes
        ):
            _evict_image_cache_entry(next(iter(_IMAGE_CACHE)))
        cached = _IMAGE_CACHE.pop(key, None)
        if cached is not None:
            _IMAGE_CACHE_BYTES -= _decoded_image_bytes(cached)
            _IMAGE_CACHE[key] = cached
            _IMAGE_CACHE_BYTES += _decoded_image_bytes(cached)
            return cached.copy()

    with Image.open(path) as image:
        decoded = image.convert("RGB")

    decoded_bytes = _decoded_image_bytes(decoded)
    with _IMAGE_CACHE_LOCK:
        # A changed file must not leave its old decoded copy retained.
        for old_key in list(_IMAGE_CACHE):
            if old_key[0] == path:
                _evict_image_cache_entry(old_key)

        # Zero disables retention. Images larger than the complete budget are
        # still processed for the current batch, but are never cached.
        if max_bytes == 0 or decoded_bytes > max_bytes:
            return decoded

        _IMAGE_CACHE[key] = decoded
        _IMAGE_CACHE_BYTES += decoded_bytes
        while (
            len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX_ITEMS or _IMAGE_CACHE_BYTES > max_bytes
        ):
            _evict_image_cache_entry(next(iter(_IMAGE_CACHE)))
        return decoded.copy()


def _load_uncached_path_image(src) -> Image.Image:
    with Image.open(src) as image:
        return image.convert("RGB")


def load_image(src, *, cache: bool = True) -> Image.Image:
    """Load image from various sources and return PIL Image in RGB."""
    # bytes -> PIL
    if isinstance(src, (bytes, bytearray)):
        with Image.open(io.BytesIO(src)) as image:
            return image.convert("RGB")
    # URL -> PIL
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        resp = requests.get(
            src, stream=True, headers={"User-Agent": "leap-finetune"}, timeout=15
        )
        try:
            resp.raise_for_status()
            with Image.open(resp.raw) as image:
                return image.convert("RGB")
        finally:
            resp.close()
    # file path -> PIL
    return _load_cached_path_image(src) if cache else _load_uncached_path_image(src)


def get_image_size(src) -> tuple[int, int]:
    """Read image dimensions without decoding the full local image."""
    if isinstance(src, (bytes, bytearray)):
        with Image.open(io.BytesIO(src)) as image:
            return image.size
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        resp = requests.get(
            src, stream=True, headers={"User-Agent": "leap-finetune"}, timeout=15
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
        img = load_image(src, cache=False)
        img.close()
        return True
    except Exception:
        return False
