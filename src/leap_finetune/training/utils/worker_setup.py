import os

import leap_finetune.distribution.ray_runtime  # noqa: F401

import ray.train
import torch
from ray.train.torch import get_device as get_ray_torch_device

from leap_finetune.data_loading.ray_data_utils import ray_dataset_to_hf
from leap_finetune.checkpointing.model_loading import load_model
from leap_finetune.distribution.ray_runtime import is_single_visible_rocm_worker
from leap_finetune.training.utils.logging import init_tracker, is_rank_zero


def _setup_ray_worker_logging(is_rank_zero_worker: bool) -> None:
    """Disable noisy progress bars on non-rank-0 Ray Train workers."""
    if is_rank_zero_worker:
        return

    import datasets

    datasets.disable_progress_bars()


def _pin_ray_worker_cuda_device() -> None:
    """Pin this worker to the CUDA device assigned by Ray Train."""
    if not torch.cuda.is_available():
        return

    if is_single_visible_rocm_worker():
        os.environ["LOCAL_RANK"] = "0"
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda:0"
        torch.cuda.set_device(0)
        return

    worker_device = get_ray_torch_device()
    if worker_device.type != "cuda":
        return

    torch.cuda.set_device(worker_device)


def setup_training_worker() -> None:
    """Run common per-worker setup before model/data initialization."""
    _setup_ray_worker_logging(is_rank_zero())
    _pin_ray_worker_cuda_device()


def get_ray_train_eval_datasets():
    """Fetch Ray train/eval shards and convert them to HF datasets."""
    train_ds_ray = ray.train.get_dataset_shard("train")
    try:
        eval_ds_ray = ray.train.get_dataset_shard("eval")
    except KeyError:
        eval_ds_ray = None
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray) if eval_ds_ray is not None else None
    return train_dataset, eval_dataset


def resolve_train_eval_datasets(train_dataset=None, eval_dataset=None):
    """Use supplied datasets locally, otherwise resolve the Ray worker shard."""
    if train_dataset is not None:
        return train_dataset, eval_dataset, lambda trainer: trainer

    setup_training_worker()
    train_dataset, eval_dataset = get_ray_train_eval_datasets()
    from ray.train.huggingface.transformers import prepare_trainer

    return train_dataset, eval_dataset, prepare_trainer


def init_tracking_from_config(
    job_name: str,
    train_config: dict,
    *,
    output_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> str:
    """Initialize configured tracking and return the normalized tracker name."""
    tracker = train_config.get("tracker", "none")
    if tracker == "none" and train_config.get("wandb_logging", False):
        tracker = "wandb"

    init_tracker(
        job_name,
        tracker,
        train_config.get("trackio_space_id"),
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return tracker


def default_eval_batch_size(train_config_filtered: dict) -> None:
    """Default eval batch size to train batch size when unspecified."""
    if "per_device_eval_batch_size" not in train_config_filtered:
        train_config_filtered["per_device_eval_batch_size"] = train_config_filtered.get(
            "per_device_train_batch_size", 1
        )


def load_causal_lm_for_training(
    training_config: dict,
    *,
    model_name: str,
    train_config: dict,
    **load_kwargs,
):
    """Load a causal LM with the repo-standard config/template overrides."""
    return load_model(
        model_name,
        model_config=training_config.get("model_config"),
        chat_template=train_config.get("chat_template"),
        chat_template_path=train_config.get("chat_template_path"),
        **load_kwargs,
    )
