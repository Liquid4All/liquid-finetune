from dataclasses import dataclass, field
from typing import Literal

from datasets import Dataset, load_dataset

from .validate_dataset_format import (
    find_local_files,
    get_source_type,
    quick_validate_schema,
    validate_dataset_format,
)


@dataclass
class DatasetLoader:
    """Dataset loader for training and testing datasets."""

    dataset_path: str
    dataset_type: Literal[
        "sft",
        "dpo",
        "kto",
        "embedding",
        "colbert",
        "vlm_sft",
        "vlm_dpo",
        "grpo",
        "vlm_grpo",
    ]
    model_name: str | None = None
    limit: int | None = None
    split: str = "train"
    test_size: float | None = 0.2
    subset: str | None = None
    val_dataset_path: str | None = None
    val_split: str | None = None
    val_subset: str | None = None
    # Prepended to relative image paths in VLM datasets (e.g. "/data/images")
    image_root: str | None = None
    cache_dataset: bool = False
    hf_streaming_batch_size: int = 10000
    _validated: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.test_size is not None and not (0 < self.test_size < 1):
            raise ValueError(
                f"test_size must be between 0 and 1 (exclusive), got {self.test_size}"
            )
        if self.test_size is not None and (
            self.val_dataset_path is not None or self.val_split is not None
        ):
            raise ValueError(
                "test_size cannot be combined with explicit validation dataset settings"
            )

    def has_eval_dataset(self) -> bool:
        return (
            self.test_size is not None
            or self.val_dataset_path is not None
            or self.val_split is not None
        )

    def get_train_path(self) -> str:
        return self.dataset_path

    def get_eval_path(self) -> str | None:
        if self.val_dataset_path is not None:
            return self.val_dataset_path
        if self.val_split is not None:
            return self.dataset_path
        return None

    def quick_validate(self) -> None:
        """Fast validation on ~10 samples. Raises ValueError on issues. No-ops if already called."""
        if self._validated:
            return
        quick_validate_schema(
            dataset_path=self.get_train_path(),
            dataset_type=self.dataset_type,
            subset=self.subset,
            split=self.split,
            num_samples=10,
            image_root=self.image_root,
            model_name=self.model_name,
        )
        eval_path = self.get_eval_path()
        if eval_path is not None:
            quick_validate_schema(
                dataset_path=eval_path,
                dataset_type=self.dataset_type,
                subset=self.val_subset if self.val_split is not None else self.subset,
                split=self.val_split or "train",
                num_samples=10,
                image_root=self.image_root,
                model_name=self.model_name,
            )
        self._validated = True

    def to_ray_dataset(
        self,
        *,
        dataset_path: str | None = None,
        subset: str | None = None,
        split: str | None = None,
    ):
        """Create a lazy Ray Dataset from the source."""
        import ray.data
        from rich.console import Console

        console = Console()
        path = dataset_path or self.dataset_path
        subset = self.subset if subset is None else subset
        split = self.split if split is None else split
        source_type = get_source_type(path)

        if source_type == "parquet":
            local_parquet_files = find_local_files(path, "*.parquet", "*.pq")
            if local_parquet_files:
                parquet_files = [str(file) for file in local_parquet_files]
                console.print(
                    f"[dim]Reading {len(parquet_files)} parquet shards from: {path}[/dim]"
                )
                ds = ray.data.read_parquet(parquet_files)
            else:
                console.print(f"[dim]Reading parquet: {path}[/dim]")
                ds = ray.data.read_parquet(path)
        elif source_type in {"s3", "gcs", "azure", "cloud"}:
            cloud_type = self._get_cloud_type(path)
            if path.lower().endswith((".parquet", ".pq")):
                console.print(f"[dim]Reading parquet from {cloud_type}: {path}[/dim]")
                ds = ray.data.read_parquet(path)
            else:
                console.print(f"[dim]Reading JSON from {cloud_type}: {path}[/dim]")
                ds = ray.data.read_json(path)
        else:
            ds = self._load_dataset_via_hf(path=path, subset=subset, split=split)

        if self.limit:
            ds = ds.limit(self.limit)

        return ds

    def _get_cloud_type(self, path: str) -> str:
        """Get human-readable cloud provider name."""
        if path.startswith("s3://"):
            return "S3"
        elif path.startswith("gs://"):
            return "GCS"
        elif path.startswith(("az://", "abfs://", "abfss://")):
            return "Azure"
        return "cloud"

    def _load_huggingface_as_ray(self, path: str, subset: str | None, split: str):
        """Load a HuggingFace Hub dataset into Ray Data via streaming."""
        import ray
        import ray.data
        import pandas as pd
        from rich.console import Console
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
        )

        console = Console()
        console.print(f"[dim]Streaming from HuggingFace: {path}[/dim]")
        split_parts = [part.strip() for part in split.split("+") if part.strip()]
        if not split_parts:
            raise ValueError("Dataset split cannot be empty")

        # Stream batches and put into Ray object store
        refs = []
        batch_size = self.hf_streaming_batch_size
        total_items = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[dim]{task.fields[items]} items[/dim]"),
            transient=True,
        ) as progress:
            task = progress.add_task("Streaming dataset...", total=None, items=0)

            for split_part in split_parts:
                # HF streaming does not accept split expressions like
                # "train+validation", so stream each named split sequentially.
                hf_stream = load_dataset(
                    path,
                    subset,
                    split=split_part,
                    streaming=True,
                )

                for batch in hf_stream.batch(batch_size=batch_size):
                    # Object dtype preserves heterogeneous messages[].content until
                    # normalize uniformizes it; pa.Table.from_pydict would reject it here.
                    df = pd.DataFrame(batch)
                    if self.limit is not None:
                        remaining = self.limit - total_items
                        if remaining <= 0:
                            break
                        df = df.iloc[:remaining]
                    refs.append(ray.put(df))
                    total_items += len(df)
                    progress.update(task, items=f"{total_items:,}")

                    if self.limit and total_items >= self.limit:
                        break

                if self.limit and total_items >= self.limit:
                    break

        return ray.data.from_pandas_refs(refs)

    def _load_dataset_via_hf(self, *, path: str, subset: str | None, split: str):
        """Load non-parquet sources via Hugging Face datasets and convert to Ray."""
        import ray.data
        from rich.console import Console

        console = Console()
        source_type = get_source_type(path)

        if source_type == "huggingface":
            return self._load_huggingface_as_ray(path, subset, split)

        console.print(f"[dim]Loading via datasets: {path}[/dim]")

        if source_type == "directory":
            dataset = load_dataset(path, subset, split=split)
        else:
            builder_by_type = {
                "json": "json",
                "csv": "csv",
                "arrow": "arrow",
            }
            builder_name = builder_by_type.get(source_type)
            if builder_name is None:
                raise ValueError(f"Unsupported dataset source for path '{path}'")
            dataset = load_dataset(builder_name, data_files=path, split=split)

        return ray.data.from_arrow(dataset.data.table)

    def load(self) -> tuple[Dataset, Dataset]:
        """Load and return validated (train, test) dataset"""
        split_str = self.split
        if self.limit:
            split_str = f"{self.split}[:{self.limit}]"

        # Load dataset from either local file or HuggingFace
        source_type = get_source_type(self.dataset_path)
        if source_type != "huggingface":
            try:
                if source_type == "directory":
                    dataset = load_dataset(
                        self.dataset_path, self.subset, split=split_str
                    )
                else:
                    file_type = {
                        "parquet": "parquet",
                        "json": "json",
                        "csv": "csv",
                        "arrow": "arrow",
                    }.get(source_type, "json")
                    dataset = load_dataset(
                        file_type, data_files=self.dataset_path, split=split_str
                    )
            except Exception as e:
                raise ValueError(
                    f"Failed to load local dataset '{self.dataset_path}': {e}"
                )
        else:
            # Try HuggingFace dataset
            try:
                dataset = load_dataset(self.dataset_path, self.subset, split=split_str)
            except Exception as e:
                raise ValueError(
                    f"Failed to load HuggingFace dataset '{self.dataset_path}': {e}"
                )

        # Validate dataset format
        model_family = "lfm2"
        if self.model_name and self.dataset_type in ("sft", "dpo"):
            from leap_finetune.checkpointing.model_info import get_model_family

            model_family = get_model_family(self.model_name)
        dataset = validate_dataset_format(
            dataset, self.dataset_type, model_family=model_family
        )

        if self.test_size is None:
            raise ValueError(
                "DatasetLoader.load() requires test_size; use to_ray_dataset()/create_ray_datasets() "
                "for configurations without eval splits."
            )

        # Split dataset
        split_dataset = dataset.train_test_split(test_size=self.test_size)
        return split_dataset["train"], split_dataset["test"]
