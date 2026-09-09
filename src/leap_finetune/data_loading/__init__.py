from .dataset_loader import DatasetLoader
from .length_grouping import get_length_grouped_sampler
from .validate_dataset_format import (
    get_row_filter,
    normalize_columns,
    quick_validate_schema,
    validate_dataset_format,
)


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
