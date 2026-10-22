from __future__ import annotations

import copy
import itertools
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
import threading
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import torch
from PIL import Image

from leap_finetune.data_loading.image_loader import load_image


@lru_cache(maxsize=8)
def _load_reference_dataset(path: str):
    from datasets import load_from_disk

    return load_from_disk(path, keep_in_memory=False)


def _load_referenced_image(reference: dict[str, Any]) -> Any:
    dataset_path = str(reference["dataset_path"])
    row_index = int(reference["row_index"])
    column = str(reference["column"])
    dataset = _load_reference_dataset(dataset_path)
    if row_index < 0 or row_index >= len(dataset):
        raise IndexError(f"image_ref row {row_index} is outside {dataset_path}")
    row = dataset[row_index]
    if column not in row:
        raise KeyError(f"image_ref column {column!r} is absent from {dataset_path}")
    image = row[column]
    if isinstance(image, Image.Image) and image.mode != "RGB":
        converted = image.convert("RGB")
        image.close()
        return converted
    return image


def _messages(
    item: dict[str, Any], generator: torch.Generator | None = None
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    variants = item.pop("instruction_variants", None)
    if variants:
        if not isinstance(variants, list) or not all(
            isinstance(value, str) and value for value in variants
        ):
            raise ValueError("instruction_variants must be non-empty strings")
        instruction = variants[
            int(torch.randint(len(variants), size=(), generator=generator).item())
        ]
    else:
        instruction = item.get("instruction")
    if instruction:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}],
            }
        )
    content: list[dict[str, Any]] = []
    image = item.get("image")
    if image is not None:
        content.append({"type": "image", "image": image})
    text = item.get("text")
    if text is not None:
        content.append({"type": "text", "text": text})
    messages.append({"role": "user", "content": content})
    return messages


def create_vlm_embedding_collate_fn(processor, max_text_tokens: int | None = None):
    """Collate three assistant-prefix VLM batches with explicit left padding.

    ``max_text_tokens`` truncates each role's text field to that many tokens
    before templating, bounding sequence length for long text documents.
    """

    processor.tokenizer.padding_side = "left"
    if max_text_tokens is not None and max_text_tokens <= 0:
        max_text_tokens = None

    def truncate(text: str) -> str:
        if max_text_tokens is None or len(text) <= max_text_tokens:
            return text
        ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_text_tokens:
            return text
        return processor.tokenizer.decode(ids[:max_text_tokens])

    def encode_role(
        samples: list[dict[str, Any]],
        role: str,
        generator: torch.Generator | None,
    ):
        conversations = []
        loaded_images = []
        try:
            for sample in samples:
                item = copy.deepcopy(sample[role])
                if isinstance(item.get("text"), str):
                    item["text"] = truncate(item["text"])
                image_reference = item.pop("image_ref", None)
                if image_reference is not None:
                    image = _load_referenced_image(image_reference)
                    item["image"] = image
                    loaded_images.append(image)
                elif isinstance(item.get("image"), str):
                    image = load_image(item["image"])
                    item["image"] = image
                    loaded_images.append(image)
                conversations.append(_messages(item, generator))
            return processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        finally:
            for image in loaded_images:
                if hasattr(image, "close"):
                    image.close()

    def collate(
        samples: list[dict[str, Any]], *, seed: int | None = None
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty VLM embedding batch")
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        batch = {
            role: encode_role(samples, role, generator)
            for role in ("query", "positive", "negative")
        }
        batch["candidate_mask"] = build_candidate_mask(samples)
        return batch

    return collate


def build_candidate_mask(samples: list[dict[str, Any]]) -> torch.Tensor:
    """Select semantically valid positives and negatives for each query.

    Retrieval rows use in-batch candidates from the same candidate group.
    Fixed-ontology classification rows use all label candidates from the same
    ontology. Multiple-choice rows use only their explicit choices. Duplicate
    positives are masked rather than treated as false negatives.
    """

    size = len(samples)
    group_tokens: list[str | None] = []
    positive_keys: list[str] = []
    negative_keys: list[str] = []
    for query_index, sample in enumerate(samples):
        loss_kind = sample.get("loss_kind", "retrieval")
        if loss_kind == "explicit":
            loss_kind = "multiple_choice"
        if loss_kind == "multiple_choice":
            group_tokens.append(None)
            positive_keys.append(
                str(sample.get("positive_key", f"positive:{query_index}"))
            )
            negative_keys.append(
                str(sample.get("negative_key", f"negative:{query_index}"))
            )
            continue
        if loss_kind == "classification":
            group_field = "candidate_set_id"
        elif loss_kind == "retrieval":
            group_field = "candidate_group"
        else:
            raise ValueError(f"unsupported loss_kind: {loss_kind!r}")
        group = sample.get(group_field)
        if loss_kind == "retrieval" and not group:
            group = sample.get("loss_group") or sample.get("modality") or "legacy"
        if not group:
            raise ValueError(f"{loss_kind} rows require {group_field}")
        group_tokens.append(f"{loss_kind}:{group}")
        positive_keys.append(str(sample.get("positive_key", f"positive:{query_index}")))
        negative_keys.append(str(sample.get("negative_key", f"negative:{query_index}")))

    def integer_ids(values: Sequence[str | None]) -> torch.Tensor:
        lookup: dict[str | None, int] = {}
        return torch.tensor(
            [lookup.setdefault(value, len(lookup)) for value in values],
            dtype=torch.int64,
        )

    group_ids = integer_ids(group_tokens)
    candidate_ids = integer_ids([*positive_keys, *negative_keys])
    positive_ids = candidate_ids[:size]
    negative_ids = candidate_ids[size:]
    same_group = group_ids[:, None].eq(group_ids[None, :])
    explicit = torch.tensor([value is None for value in group_tokens], dtype=torch.bool)
    same_group &= ~explicit[:, None]
    identity = torch.eye(size, dtype=torch.bool)
    positive_mask = same_group & (
        ~positive_ids[:, None].eq(positive_ids[None, :]) | identity
    )
    negative_mask = same_group & ~positive_ids[:, None].eq(negative_ids[None, :])
    positive_mask |= identity
    negative_mask |= identity
    mask = torch.cat((positive_mask, negative_mask), dim=1)
    return mask


def collate_raw_vlm_embedding_samples(
    samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep a logical batch raw so cached loss can tokenize bounded chunks."""

    if not samples:
        raise ValueError("cannot collate an empty VLM embedding batch")
    return {"samples": samples}


__all__ = [
    "OrderedChunkCollator",
    "build_candidate_mask",
    "collate_raw_vlm_embedding_samples",
    "create_vlm_embedding_collate_fn",
]


CollateJob = tuple[list[dict[str, Any]], int]

_WORKER_COLLATE: Any = None


def _initialize_collate_worker(processor: Any, max_text_tokens: int | None) -> None:
    global _WORKER_COLLATE
    torch.set_num_threads(1)
    _WORKER_COLLATE = create_vlm_embedding_collate_fn(processor, max_text_tokens)


def _collate_in_worker(samples: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    if _WORKER_COLLATE is None:
        raise RuntimeError("collate worker was not initialized")
    return _WORKER_COLLATE(samples, seed=seed)


_THREAD_STATE = threading.local()


def _initialize_collate_thread(processor: Any, max_text_tokens: int | None) -> None:
    _THREAD_STATE.collate = create_vlm_embedding_collate_fn(
        copy.deepcopy(processor), max_text_tokens
    )


def _collate_in_thread(samples: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    return _THREAD_STATE.collate(samples, seed=seed)


class OrderedChunkCollator:
    """Collate seeded micro-batches in order, optionally in worker processes.

    With ``workers > 0`` a persistent pool collates up to ``prefetch`` chunks
    ahead of the consumer, so image loading and preprocessing overlap
    accelerator work. The ``thread`` backend shares collated tensors without
    copying and gives each thread its own processor; the ``process`` backend
    isolates workers but pays to move tensors between processes. Results are
    yielded in submission order and depend only on each chunk's samples and
    seed, so inline and pooled collation match.
    """

    def __init__(
        self,
        processor: Any,
        *,
        workers: int = 0,
        prefetch: int | None = None,
        backend: str = "thread",
        start_method: str = "spawn",
        max_text_tokens: int | None = None,
    ) -> None:
        if workers < 0:
            raise ValueError("collate workers must be non-negative")
        if backend not in {"thread", "process"}:
            raise ValueError("collate backend must be 'thread' or 'process'")
        if prefetch is not None and prefetch <= 0:
            raise ValueError("collate prefetch must be positive")
        self.workers = workers
        self.prefetch = prefetch if prefetch is not None else workers + 2
        self._processor = processor
        self.backend = backend
        self._start_method = start_method
        self.max_text_tokens = max_text_tokens
        self._inline = create_vlm_embedding_collate_fn(processor, max_text_tokens)
        self._executor: Executor | None = None
        self._task = _collate_in_thread if backend == "thread" else _collate_in_worker

    def _pool(self) -> Executor:
        if self._executor is None:
            if self.backend == "thread":
                # Process-wide: collation threads, not torch's intra-op pool,
                # provide the parallelism, and the pool otherwise sizes itself
                # to every core on the node rather than the job's allocation.
                torch.set_num_threads(1)
                self._executor = ThreadPoolExecutor(
                    max_workers=self.workers,
                    thread_name_prefix="vlm-collate",
                    initializer=_initialize_collate_thread,
                    initargs=(self._processor, self.max_text_tokens),
                )
            else:
                self._executor = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=torch.multiprocessing.get_context(self._start_method),
                    initializer=_initialize_collate_worker,
                    initargs=(self._processor, self.max_text_tokens),
                )
        return self._executor

    def iterate(self, jobs: Iterable[CollateJob]) -> Iterator[dict[str, Any]]:
        if self.workers == 0:
            for samples, seed in jobs:
                yield self._inline(samples, seed=seed)
            return
        pool = self._pool()
        remaining = iter(jobs)
        pending: deque[Future[dict[str, Any]]] = deque(
            pool.submit(self._task, samples, seed)
            for samples, seed in itertools.islice(remaining, self.prefetch)
        )
        try:
            while pending:
                result = pending.popleft().result()
                upcoming = next(remaining, None)
                if upcoming is not None:
                    pending.append(pool.submit(self._task, *upcoming))
                yield result
        finally:
            for future in pending:
                future.cancel()

    def submit(self, jobs: Iterable[CollateJob]) -> Iterator[dict[str, Any]]:
        """Start collating every job now and return results in order.

        Unlike ``iterate``, work begins before the first result is requested,
        so a caller can queue the next step's chunks while the current step
        runs on the accelerator.
        """

        if self.workers == 0:
            return (self._inline(samples, seed=seed) for samples, seed in jobs)
        pool = self._pool()
        futures = [pool.submit(self._task, samples, seed) for samples, seed in jobs]
        return (future.result() for future in futures)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
