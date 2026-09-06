import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LengthGroupedSampler


def get_length_grouped_sampler(
    dataset: Dataset | None,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> LengthGroupedSampler | None:
    """Build a local length-grouped sampler for tokenized HF datasets.

    Prefer a precomputed ``length`` column when present. Otherwise mirror HF's
    default behavior and let ``LengthGroupedSampler`` infer lengths from
    ``input_ids`` once during sampler construction.
    """
    if dataset is None:
        return None

    column_names = getattr(dataset, "column_names", None)
    if column_names is None:
        return None

    if "length" in column_names:
        lengths = list(dataset["length"])
        if not lengths:
            return None
        return LengthGroupedSampler(
            batch_size=batch_size,
            lengths=lengths,
            generator=generator,
        )

    if "input_ids" not in column_names:
        return None

    return LengthGroupedSampler(
        batch_size=batch_size,
        dataset=dataset,
        model_input_name="input_ids",
        generator=generator,
    )


def get_tile_count_grouped_sampler(
    dataset: Dataset | None,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
):
    """Group VLM examples with similar processed image tile counts."""
    if dataset is None:
        return None
    column_names = getattr(dataset, "column_names", None)
    if column_names is None or "_vlm_tile_count" not in column_names:
        return None
    counts = list(dataset["_vlm_tile_count"])
    if not counts:
        return None
    if len(set(counts)) == 1:
        return None
    return LengthGroupedSampler(
        batch_size=batch_size,
        lengths=[max(1, int(count)) for count in counts],
        generator=generator,
    )
