from .dataset_loader import DatasetLoader
from .length_grouping import get_length_grouped_sampler
from .validate_dataset_format import (
    get_row_filter,
    normalize_columns,
    quick_validate_schema,
    validate_dataset_format,
)


def create_ray_datasets(*args, **kwargs):
    """Import the Ray-backed loader only when distributed loading is requested."""
    from .ray_data_utils import create_ray_datasets as _create_ray_datasets

    return _create_ray_datasets(*args, **kwargs)


def ray_dataset_to_hf(*args, **kwargs):
    """Import Ray only for callers that materialize a Ray dataset."""
    from .ray_data_utils import ray_dataset_to_hf as _ray_dataset_to_hf

    return _ray_dataset_to_hf(*args, **kwargs)


__all__ = [
    "DatasetLoader",
    "get_length_grouped_sampler",
    "quick_validate_schema",
    "get_row_filter",
    "normalize_columns",
    "validate_dataset_format",
    "create_ray_datasets",
    "ray_dataset_to_hf",
]


def __getattr__(name):
    """Load Ray-backed helpers only when a caller actually needs them."""
    if name in {"create_ray_datasets", "ray_dataset_to_hf"}:
        from . import ray_data_utils

        return getattr(ray_data_utils, name)
    raise AttributeError(name)
