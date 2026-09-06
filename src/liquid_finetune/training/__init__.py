from collections.abc import Iterator, Mapping
from importlib import import_module
from types import MappingProxyType
from typing import Callable

from liquid_finetune.distribution.ray_runtime import (
    normalize_visible_devices,
    patch_ray_rocm_torch_device_helpers,
)

normalize_visible_devices()
patch_ray_rocm_torch_device_helpers()

_TRAINING_LOOP_TARGETS = MappingProxyType(
    {
        "sft": ("liquid_finetune.training.sft", "sft_run"),
        "dpo": ("liquid_finetune.training.dpo", "dpo_run"),
        "kto": ("liquid_finetune.training.kto", "kto_run"),
        "embedding": ("liquid_finetune.training.embedding", "embedding_run"),
        "colbert": ("liquid_finetune.training.colbert", "colbert_run"),
        "vlm_sft": ("liquid_finetune.training.vlm_sft", "vlm_sft_run"),
        "vlm_dpo": ("liquid_finetune.training.vlm_dpo", "vlm_dpo_run"),
        "grpo": ("liquid_finetune.training.grpo", "grpo_run"),
        "vlm_grpo": ("liquid_finetune.training.vlm_grpo", "vlm_grpo_run"),
        "moe_sft": ("liquid_finetune.training.moe_sft", "moe_sft_run"),
        "moe_dpo": ("liquid_finetune.training.moe_dpo", "moe_dpo_run"),
    }
)


PRETOKENIZED_TRAINING_TYPES = frozenset({"sft", "dpo", "moe_sft", "moe_dpo"})

# Ray splits these datasets across workers. Datasets omitted from each tuple are
# replicated, which lets rank zero report metrics over the complete eval set.
TRAINING_DATASETS_TO_SPLIT = MappingProxyType(
    {
        "embedding": ("train",),
        "colbert": ("train",),
        "grpo": (),
        "vlm_grpo": (),
    }
)


class _TrainingLoopRegistry(Mapping[str, Callable[[dict], None]]):
    """Lazy training loop registry.

    Keeping GRPO imports lazy matters on ROCm: importing TRL's GRPO module
    imports vLLM, whose ROCm platform validates GPU visibility immediately.
    Ray may inject per-worker visibility env vars before our worker setup hook
    runs, so non-GRPO loops must not import that path at package import time.
    """

    def __getitem__(self, key: str) -> Callable[[dict], None]:
        module_name, attr = _TRAINING_LOOP_TARGETS[key]
        return getattr(import_module(module_name), attr)

    def __iter__(self) -> Iterator[str]:
        return iter(_TRAINING_LOOP_TARGETS)

    def __len__(self) -> int:
        return len(_TRAINING_LOOP_TARGETS)


TRAINING_LOOPS = _TrainingLoopRegistry()


def __getattr__(name: str):
    loop_names = {
        "sft_run": "sft",
        "dpo_run": "dpo",
        "kto_run": "kto",
        "embedding_run": "embedding",
        "colbert_run": "colbert",
        "vlm_sft_run": "vlm_sft",
        "vlm_dpo_run": "vlm_dpo",
        "grpo_run": "grpo",
        "vlm_grpo_run": "vlm_grpo",
        "moe_sft_run": "moe_sft",
        "moe_dpo_run": "moe_dpo",
    }
    if name in loop_names:
        return TRAINING_LOOPS[loop_names[name]]
    raise AttributeError(name)


__all__ = [
    "TRAINING_LOOPS",
    "PRETOKENIZED_TRAINING_TYPES",
    "TRAINING_DATASETS_TO_SPLIT",
    "sft_run",
    "dpo_run",
    "kto_run",
    "embedding_run",
    "colbert_run",
    "vlm_sft_run",
    "vlm_dpo_run",
    "grpo_run",
    "vlm_grpo_run",
    "moe_sft_run",
    "moe_dpo_run",
]
