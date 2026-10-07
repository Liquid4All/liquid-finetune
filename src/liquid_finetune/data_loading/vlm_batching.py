import logging
from collections.abc import Iterator

from datasets import Dataset

from liquid_finetune.data_loading.image_loader import get_image_size

logger = logging.getLogger(__name__)
VLM_TILE_COUNT_COLUMN = "_vlm_tile_count"


def _image_sources(value) -> Iterator[str]:
    if isinstance(value, dict):
        if value.get("type") == "image" and isinstance(value.get("image"), str):
            yield value["image"]
            return
        for child in value.values():
            yield from _image_sources(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _image_sources(child)


def _row_image_sources(row: dict) -> Iterator[str]:
    for key in ("messages", "prompt", "chosen", "rejected"):
        if key in row:
            yield from _image_sources(row[key])
    if isinstance(row.get("image"), str):
        yield row["image"]
    if isinstance(row.get("images"), (list, tuple)):
        yield from (image for image in row["images"] if isinstance(image, str))


def _tile_count(image_processor, height: int, width: int) -> int:
    """Match LFM2-VL's grid decision without running pixel preprocessing."""
    if not getattr(image_processor, "do_image_splitting", False):
        return 1

    is_too_large = getattr(image_processor, "_is_image_too_large", None)
    get_grid_layout = getattr(image_processor, "_get_grid_layout", None)
    if is_too_large is None or get_grid_layout is None:
        return 1

    kwargs = {
        "max_image_tokens": int(getattr(image_processor, "max_image_tokens", 256)),
        "encoder_patch_size": int(getattr(image_processor, "encoder_patch_size", 16)),
        "downsample_factor": int(getattr(image_processor, "downsample_factor", 2)),
        "max_pixels_tolerance": float(
            getattr(image_processor, "max_pixels_tolerance", 2.0)
        ),
    }
    if not is_too_large(height=height, width=width, **kwargs):
        return 1

    _, _, _, _, tiles = get_grid_layout(
        height=height,
        width=width,
        min_tiles=int(getattr(image_processor, "min_tiles", 2)),
        max_tiles=int(getattr(image_processor, "max_tiles", 10)),
        tile_size=int(getattr(image_processor, "tile_size", 512)),
    )
    if getattr(image_processor, "use_thumbnail", True) and tiles > 1:
        tiles += 1
    return int(tiles)


def estimate_vlm_tile_count(row: dict, processor) -> int:
    """Estimate processed visual tiles for one normalized VLM row."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return 0

    count = 0
    for source in _row_image_sources(row):
        try:
            width, height = get_image_size(source)
            count += _tile_count(image_processor, height, width)
        except Exception:
            logger.debug(
                "Could not estimate VLM tile count for %s", source, exc_info=True
            )
            count += 1
    return count


def add_vlm_tile_counts(dataset: Dataset | None, processor) -> Dataset | None:
    """Add local, non-training metadata used by the VLM batch sampler."""
    if dataset is None or VLM_TILE_COUNT_COLUMN in dataset.column_names:
        return dataset
    counts = [estimate_vlm_tile_count(row, processor) for row in dataset]
    return dataset.add_column(VLM_TILE_COUNT_COLUMN, counts)
