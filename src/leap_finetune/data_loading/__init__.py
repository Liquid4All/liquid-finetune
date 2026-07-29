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
    "create_ray_datasets",
    "ray_dataset_to_hf",
    "quick_validate_schema",
    "get_row_filter",
    "normalize_columns",
    "validate_dataset_format",
]
