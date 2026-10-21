from __future__ import annotations

import tomllib
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from PIL import Image

import leap_finetune.data_loading.vlm_embedding as vlm_data
from leap_finetune.data_loading.vlm_embedding import (
    OrderedChunkCollator,
    build_candidate_mask,
    collate_raw_vlm_embedding_samples,
    create_vlm_embedding_collate_fn,
)
from leap_finetune.data_loading.validate_dataset_format import (
    normalize_columns,
    validate_dataset_format,
)
from leap_finetune.training.vlm_embedding import (
    per_device_batch_size,
    whole_warmup_steps,
    CachedLFMVLMEmbeddingTrainer,
    configure_gradient_checkpointing,
    contrastive_embedding_loss,
    pool_last_attended_token,
)


def test_pool_last_attended_token_supports_left_padding():
    hidden = torch.tensor(
        [
            [[9.0, 9.0], [3.0, 4.0], [0.0, 2.0]],
            [[1.0, 0.0], [5.0, 12.0], [8.0, 15.0]],
        ]
    )
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    pooled = pool_last_attended_token(hidden, mask)
    torch.testing.assert_close(pooled[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(pooled[1], torch.tensor([5.0 / 13.0, 12.0 / 13.0]))


def test_contrastive_embedding_loss_prefers_aligned_pairs():
    query = torch.eye(2)
    positive = torch.eye(2)
    negative = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
    aligned = contrastive_embedding_loss(query, positive, negative, temperature=0.1)
    swapped = contrastive_embedding_loss(
        query, positive.flip(0), negative, temperature=0.1
    )
    assert aligned < swapped
    with pytest.raises(ValueError, match="positive"):
        contrastive_embedding_loss(query, positive, negative, temperature=0)


def test_candidate_mask_uses_task_semantics_and_masks_duplicate_positives():
    samples = [
        {
            "loss_kind": "retrieval",
            "candidate_group": "text",
            "positive_key": "same",
            "negative_key": "n0",
        },
        {
            "loss_kind": "retrieval",
            "candidate_group": "text",
            "positive_key": "same",
            "negative_key": "n1",
        },
        {
            "loss_kind": "multiple_choice",
            "positive_key": "p2",
            "negative_key": "n2",
        },
    ]

    mask = build_candidate_mask(samples)

    assert mask.tolist() == [
        [True, False, False, True, True, False],
        [False, True, False, True, True, False],
        [False, False, True, False, False, True],
    ]


def test_candidate_mask_shares_only_one_class_ontology():
    samples = [
        {
            "loss_kind": "classification",
            "candidate_set_id": "imagenet",
            "positive_key": "dog",
            "negative_key": "cat",
        },
        {
            "loss_kind": "classification",
            "candidate_set_id": "imagenet",
            "positive_key": "cat",
            "negative_key": "plane",
        },
        {
            "loss_kind": "classification",
            "candidate_set_id": "sun397",
            "positive_key": "kitchen",
            "negative_key": "street",
        },
    ]

    mask = build_candidate_mask(samples)

    assert mask.tolist() == [
        [True, True, False, True, True, False],
        [True, True, False, False, True, False],
        [False, False, True, False, False, True],
    ]


def test_candidate_mask_maps_legacy_objective_fields():
    samples = [
        {"loss_kind": "explicit", "modality": "image"},
        {"loss_kind": "retrieval", "loss_group": "text"},
        {"modality": "text"},
    ]

    mask = build_candidate_mask(samples)

    assert mask.tolist() == [
        [True, False, False, True, False, False],
        [False, True, True, False, True, True],
        [False, True, True, False, True, True],
    ]


def test_raw_collator_preserves_one_logical_batch():
    samples = [{"id": "a"}, {"id": "b"}]

    assert collate_raw_vlm_embedding_samples(samples) == {"samples": samples}


def test_contrastive_loss_honors_candidate_mask():
    query = torch.eye(2)
    positive = torch.eye(2)
    negative = torch.eye(2)
    explicit = torch.tensor([[True, False, True, False], [False, True, False, True]])

    loss = contrastive_embedding_loss(
        query,
        positive,
        negative,
        temperature=0.1,
        candidate_mask=explicit,
    )

    torch.testing.assert_close(loss, torch.tensor(0.6931472))


def test_cached_contrastive_replay_matches_full_batch_gradient():
    samples = [
        {
            "loss_kind": "retrieval",
            "candidate_group": "text",
            "positive_key": f"p{index}",
            "negative_key": f"n{index}",
            "query": torch.tensor([1.0 + index, 0.5, -0.25]),
            "positive": torch.tensor([0.75, 1.0 + index, 0.25]),
            "negative": torch.tensor([-0.5, 0.25, 1.0 + index]),
        }
        for index in range(4)
    ]
    cached_model = torch.nn.Linear(3, 2, bias=False)
    full_model = torch.nn.Linear(3, 2, bias=False)
    full_model.load_state_dict(cached_model.state_dict())

    trainer = object.__new__(CachedLFMVLMEmbeddingTrainer)
    trainer.micro_batch_size = 2
    trainer.embedding_temperature = 0.2
    trainer.optimizer = SimpleNamespace(train=lambda: None)
    trainer.accelerator = SimpleNamespace(backward=lambda value: value.backward())
    trainer.compute_loss_context_manager = lambda: nullcontext()
    trainer._prepare_inputs = lambda value: value

    def collate(chunk):
        return {
            role: {"x": torch.stack([sample[role] for sample in chunk])}
            for role in ("query", "positive", "negative")
        }

    def encode(self, model, batch):
        return torch.nn.functional.normalize(model(batch["x"]), dim=-1)

    trainer._chunk_collator = SimpleNamespace(
        iterate=lambda jobs: (collate(chunk) for chunk, _seed in jobs)
    )
    trainer._last_step_metrics = {}
    trainer._encode = types.MethodType(encode, trainer)
    cached_loss = trainer.training_step(cached_model, {"samples": samples})

    full_embeddings = {
        role: torch.nn.functional.normalize(
            full_model(torch.stack([sample[role] for sample in samples])), dim=-1
        )
        for role in ("query", "positive", "negative")
    }
    full_loss = contrastive_embedding_loss(
        full_embeddings["query"],
        full_embeddings["positive"],
        full_embeddings["negative"],
        temperature=0.2,
        candidate_mask=build_candidate_mask(samples),
    )
    full_loss.backward()

    torch.testing.assert_close(cached_loss, full_loss.detach())
    torch.testing.assert_close(cached_model.weight.grad, full_model.weight.grad)


def test_registered_vlm_embedding_defaults_use_cached_logical_batch():
    config = Path(__file__).parents[2] / "dittopilot" / "training.toml"
    registered = tomllib.loads(config.read_text())["vlm_embedding"]["args"]

    assert registered["batch_size"]["default"] == 8192
    assert registered["micro_batch_size"]["default"] == 32
    assert registered["cached_loss"]["default"] == "true"
    assert registered["max_steps"]["default"] == 80
    assert registered["save_steps"]["default"] == 10
    assert registered["gradient_accumulation_steps"]["default"] == 1
    assert registered["collate_workers"]["default"] == 16
    assert registered["gradient_checkpointing"]["default"] == "false"
    assert registered["gradient_checkpointing_scope"]["default"] == "auto"
    assert registered["tracker"]["default"] == "wandb"


def test_dittopilot_vlm_embedding_exposes_memory_and_tracking_controls():
    root = Path(__file__).parents[2]
    runner = (root / "dittopilot" / "run_vlm_embedding.sh").read_text()

    assert "gradient_checkpointing: ${GRADIENT_CHECKPOINTING}" in runner
    assert 'gradient_checkpointing_scope: "${GRADIENT_CHECKPOINTING_SCOPE}"' in runner
    assert 'tracker: "${TRACKER}"' in runner
    assert "micro_batch_size: ${MICRO_BATCH_SIZE}" in runner
    assert "cached_loss: ${CACHED_LOSS}" in runner
    assert "collate_workers: ${COLLATE_WORKERS}" in runner
    assert 'WANDB_MODE="${WANDB_MODE:-online}"' in runner
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-retrieval-lfm}"' in runner


class _Checkpointable:
    def __init__(self) -> None:
        self.enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.enabled = True


class _CheckpointingModel:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            language_model=_Checkpointable(),
            vision_tower=_Checkpointable(),
        )


@pytest.mark.parametrize(
    ("scope", "trainer_enabled", "language_enabled", "vision_enabled"),
    [
        ("none", False, False, False),
        ("all", True, False, False),
        ("language", False, True, False),
        ("vision", False, False, True),
    ],
)
def test_configure_gradient_checkpointing_scope(
    scope, trainer_enabled, language_enabled, vision_enabled
):
    model = _CheckpointingModel()

    assert configure_gradient_checkpointing(model, scope) is trainer_enabled
    assert model.model.language_model.enabled is language_enabled
    assert model.model.vision_tower.enabled is vision_enabled


def test_configure_gradient_checkpointing_rejects_unknown_scope():
    with pytest.raises(ValueError, match="gradient_checkpointing_scope"):
        configure_gradient_checkpointing(_CheckpointingModel(), "decoder")


def test_dittopilot_uses_pinned_rocm_project_on_amd():
    root = Path(__file__).parents[2]
    setup = (root / "dittopilot" / "setup.sh").read_text()
    runner = (root / "dittopilot" / "run_vlm_embedding.sh").read_text()

    assert "uv sync --project envs/rocm" in setup
    assert "UV_PROJECT_ARGS=(--project envs/rocm)" in runner


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
        return {
            "input_ids": torch.tensor([[0, 4, 5], [2, 3, 5]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }


def test_vlm_embedding_collator_uses_assistant_prefix_and_left_padding():
    processor = _FakeProcessor()
    collator = create_vlm_embedding_collate_fn(processor)
    item = {"instruction": "Represent it.", "text": "content"}
    batch = collator(
        [
            {"query": item, "positive": item, "negative": item},
            {"query": item, "positive": item, "negative": item},
        ]
    )
    assert processor.tokenizer.padding_side == "left"
    assert set(batch) == {"query", "positive", "negative", "candidate_mask"}
    assert len(processor.calls) == 3
    assert all(call[1]["add_generation_prompt"] for call in processor.calls)
    assert all(call[1]["padding"] for call in processor.calls)


def test_vlm_embedding_collator_omits_empty_candidate_system_prompt():
    processor = _FakeProcessor()
    collator = create_vlm_embedding_collate_fn(processor)
    query = {"instruction": "Classify the image.", "text": "query"}
    label = {"instruction": "", "text": "class label"}

    collator([{"query": query, "positive": label, "negative": label}])

    query_messages = processor.calls[0][0][0]
    positive_messages = processor.calls[1][0][0]
    assert [message["role"] for message in query_messages] == ["system", "user"]
    assert [message["role"] for message in positive_messages] == ["user"]


def test_vlm_embedding_collator_samples_instruction_variants():
    processor = _FakeProcessor()
    collator = create_vlm_embedding_collate_fn(processor)
    query = {
        "instruction": "fallback",
        "instruction_variants": ["first", "second"],
        "text": "query",
    }
    label = {"instruction": "", "text": "class label"}
    torch.manual_seed(7)

    collator([{"query": query, "positive": label, "negative": label}])

    system = processor.calls[0][0][0][0]
    assert system["content"][0]["text"] in {"first", "second"}


def test_vlm_embedding_collator_resolves_shared_dataset_image_refs(
    tmp_path, monkeypatch
):
    processor = _FakeProcessor()
    collator = create_vlm_embedding_collate_fn(processor)
    reference = {
        "dataset_path": str(tmp_path),
        "row_index": 7,
        "column": "positive_image",
    }
    resolved = []

    def load_reference(value):
        resolved.append(value)
        return Image.new("RGB", (2, 2), "white")

    monkeypatch.setattr(vlm_data, "_load_referenced_image", load_reference)
    item = {"instruction": "", "image_ref": reference}

    collator([{"query": item, "positive": item, "negative": item}])

    assert resolved == [reference, reference, reference]
    assert all(
        [part["type"] for part in call[0][0][-1]["content"]] == ["image"]
        for call in processor.calls
    )


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "LA", "RGB"])
def test_referenced_images_are_converted_to_rgb(tmp_path, mode):
    from datasets import Features
    from datasets import Image as ImageFeature
    from transformers.image_utils import infer_channel_dimension_format
    import numpy as np

    dataset_path = tmp_path / mode
    Dataset.from_dict(
        {"page": [Image.new(mode, (5, 7))]},
        features=Features({"page": ImageFeature()}),
    ).save_to_disk(str(dataset_path))
    vlm_data._load_reference_dataset.cache_clear()

    image = vlm_data._load_referenced_image(
        {"dataset_path": str(dataset_path), "row_index": 0, "column": "page"}
    )

    assert image.mode == "RGB"
    assert image.size == (5, 7)
    infer_channel_dimension_format(np.asarray(image))


def test_vlm_embedding_dataset_normalizes_relative_images(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "white").save(image_path)
    row = {
        "query": {"text": "question"},
        "positive": {"image": "image.png"},
        "negative": {"text": "unrelated"},
    }
    dataset = Dataset.from_list(
        [normalize_columns("vlm_embedding", image_root=str(tmp_path))(row)]
    )
    validated = validate_dataset_format(dataset, "vlm_embedding")
    assert validated[0]["positive"]["image"] == str(image_path)
    assert "instruction" in validated[0]["query"]


def test_vlm_embedding_dataset_accepts_shared_dataset_image_refs(tmp_path):
    row = {
        "query": {"instruction": "retrieve", "text": "question"},
        "positive": {
            "instruction": "",
            "image_ref": {
                "dataset_path": str(tmp_path),
                "row_index": 0,
                "column": "positive_image",
            },
        },
        "negative": {
            "instruction": "",
            "image_ref": {
                "dataset_path": str(tmp_path),
                "row_index": 1,
                "column": "positive_image",
            },
        },
    }

    dataset = Dataset.from_list([normalize_columns("vlm_embedding")(row)])

    assert len(validate_dataset_format(dataset, "vlm_embedding")) == 1


class _InstructionEchoProcessor:
    """Picklable processor whose token ids reveal the sampled instruction."""

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="right")

    def apply_chat_template(self, conversations, **kwargs):
        lengths = []
        for messages in conversations:
            system = [m for m in messages if m["role"] == "system"]
            lengths.append(len(system[0]["content"][0]["text"]) if system else 0)
        ids = torch.tensor(lengths).unsqueeze(1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _variant_jobs() -> list[tuple[list[dict], int]]:
    query = {"instruction_variants": ["a", "bb", "ccc", "dddd"], "text": "q"}
    label = {"instruction": "", "text": "label"}
    chunk = [{"query": query, "positive": label, "negative": label}] * 4
    return [(chunk, seed) for seed in range(12)]


def _query_ids(batches) -> list[list[int]]:
    return [batch["query"]["input_ids"].flatten().tolist() for batch in batches]


def test_seeded_collation_reproduces_instruction_sampling():
    collator = OrderedChunkCollator(_InstructionEchoProcessor(), workers=0)
    jobs = _variant_jobs()

    first = _query_ids(collator.iterate(jobs))
    second = _query_ids(collator.iterate(jobs))

    assert first == second
    assert len({tuple(ids) for ids in first}) > 1


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_pooled_collation_matches_inline_order_and_content(backend):
    jobs = _variant_jobs()
    inline = OrderedChunkCollator(_InstructionEchoProcessor(), workers=0)
    pooled = OrderedChunkCollator(
        _InstructionEchoProcessor(),
        workers=3,
        prefetch=2,
        backend=backend,
        start_method="fork",
    )
    try:
        pooled_ids = _query_ids(pooled.iterate(jobs + jobs))
    finally:
        pooled.close()

    inline_ids = _query_ids(inline.iterate(jobs))
    assert pooled_ids == inline_ids + inline_ids


def test_collator_rejects_invalid_worker_settings():
    with pytest.raises(ValueError, match="workers"):
        OrderedChunkCollator(_InstructionEchoProcessor(), workers=-1)
    with pytest.raises(ValueError, match="prefetch"):
        OrderedChunkCollator(_InstructionEchoProcessor(), workers=1, prefetch=0)
    with pytest.raises(ValueError, match="backend"):
        OrderedChunkCollator(_InstructionEchoProcessor(), backend="ray")


@pytest.mark.parametrize(
    ("ratio", "steps", "expected"),
    [(0.05, 8, 0), (0.05, 80, 4), (0.0, 8, 0), (0.1, 100, 10)],
)
def test_whole_warmup_steps_rounds_down(ratio, steps, expected):
    assert whole_warmup_steps(ratio, steps) == expected


def test_whole_warmup_steps_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="warmup_ratio"):
        whole_warmup_steps(1.0, 8)
    with pytest.raises(ValueError, match="max_steps"):
        whole_warmup_steps(0.05, 0)


def _toy_samples(count: int) -> list[dict]:
    return [
        {
            "loss_kind": "retrieval",
            "candidate_group": "text",
            "positive_key": f"p{index}",
            "negative_key": f"n{index}",
            "query": torch.tensor([1.0 + index, 0.5, -0.25]),
            "positive": torch.tensor([0.75, 1.0 + index, 0.25]),
            "negative": torch.tensor([-0.5, 0.25, 1.0 + index]),
        }
        for index in range(count)
    ]


def _toy_cached_trainer(micro_batch_size: int) -> CachedLFMVLMEmbeddingTrainer:
    trainer = object.__new__(CachedLFMVLMEmbeddingTrainer)
    trainer.micro_batch_size = micro_batch_size
    trainer.embedding_temperature = 0.2
    trainer.optimizer = SimpleNamespace(train=lambda: None)
    trainer.accelerator = SimpleNamespace(backward=lambda value: value.backward())
    trainer.compute_loss_context_manager = lambda: nullcontext()
    trainer._prepare_inputs = lambda value: value
    trainer._last_step_metrics = {}

    def collate(chunk):
        return {
            role: {"x": torch.stack([sample[role] for sample in chunk])}
            for role in ("query", "positive", "negative")
        }

    def encode(self, model, batch):
        return torch.nn.functional.normalize(model(batch["x"]), dim=-1)

    trainer._chunk_collator = SimpleNamespace(
        iterate=lambda jobs: (collate(chunk) for chunk, _seed in jobs)
    )
    trainer._encode = types.MethodType(encode, trainer)
    return trainer


def _distributed_step_worker(rank, world, init_file, bounds, state, queue):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        model = torch.nn.Linear(3, 2, bias=False)
        model.load_state_dict(state)
        start, end = bounds[rank]
        trainer = _toy_cached_trainer(micro_batch_size=2)
        loss = trainer.training_step(model, {"samples": _toy_samples(5)[start:end]})
        queue.put((rank, loss.item(), model.weight.grad.tolist()))
    finally:
        dist.destroy_process_group()


def test_distributed_cached_step_matches_single_process_full_batch(tmp_path):
    import multiprocessing

    reference = torch.nn.Linear(3, 2, bias=False)
    state = {key: value.clone() for key, value in reference.state_dict().items()}
    samples = _toy_samples(5)
    full_loss = _toy_cached_trainer(micro_batch_size=2).training_step(
        reference, {"samples": samples}
    )

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    bounds = [(0, 3), (3, 5)]
    workers = [
        context.Process(
            target=_distributed_step_worker,
            args=(rank, 2, str(tmp_path / "init"), bounds, state, queue),
        )
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    results = sorted(queue.get(timeout=120) for _ in workers)
    for worker in workers:
        worker.join(timeout=60)
        assert worker.exitcode == 0

    for _rank, loss, grad in results:
        torch.testing.assert_close(torch.tensor(loss), full_loss)
        torch.testing.assert_close(torch.tensor(grad), reference.weight.grad)


@pytest.mark.parametrize(
    ("global_size", "world", "expected"), [(512, 8, 64), (8192, 4, 2048), (7, 1, 7)]
)
def test_per_device_batch_size_splits_global_batch(global_size, world, expected):
    assert per_device_batch_size(global_size, world) == expected


def test_per_device_batch_size_rejects_uneven_split():
    with pytest.raises(ValueError, match="divisible"):
        per_device_batch_size(512, 3)
