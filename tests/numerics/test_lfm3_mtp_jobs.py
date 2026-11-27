"""MTP jobs: the ``mtp:`` job section, start-point reuse and ``liquid-eval-mtp``."""

import json

import pytest


def test_acceptance_simulation_counts_the_longest_correct_draft_prefix():
    from liquid_finetune.models.lfm3_mtp.evaluate import _accepted_per_step

    tokens = [9, 9, 1, 2, 3, 4, 5, 6]  # prompt = first 2 tokens
    # drafts[d-1][t] drafts x_{t+d+1}: module 1 is always right, module 2 only at t=1
    drafts = [tokens[2:] + [0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    drafts[1][1] = tokens[4]
    assert _accepted_per_step(tokens, drafts, 2, 1) == [1, 1, 1]
    # t=1: emits x2, accepts x3 (d1) and x4 (d2) -> t=4: x5 + x6 (d1) -> t=6: x7
    assert _accepted_per_step(tokens, drafts, 2, 2) == [2, 1, 0]


def test_job_config_mtp_section_is_validated():
    from pydantic import ValidationError

    from liquid_finetune.config.job_config import JobConfig

    base = {
        "model_name": "LFM3-1.7B",
        "training_type": "sft",
        "dataset": {"path": "HuggingFaceTB/smoltalk", "type": "sft"},
    }
    job = JobConfig(**base, mtp={"num_draft_tokens": 3, "objective": "soft"})
    assert job.mtp.num_draft_tokens == 3 and not job.mtp.train_backbone
    for bad in (
        {"num_draft_tokens": 0},
        {"num_draft_tokens": 6},
        {"num_draft_tokens": 2, "train_backbone": True},
    ):
        with pytest.raises(ValidationError):
            JobConfig(**base, mtp=bad)
    with pytest.raises(ValidationError, match="sft"):
        JobConfig(
            **{**base, "training_type": "dpo", "dataset": {"path": "x", "type": "dpo"}},
            mtp={"num_draft_tokens": 1},
        )


def test_a_start_point_is_reused_only_for_identical_settings(tmp_path):
    from liquid_finetune.models.lfm3_mtp.job_setup import _matches
    from liquid_finetune.models.lfm3_mtp.prepare import MANIFEST_NAME

    request = {
        "source": "LiquidAI/LFM3-1.7B",
        "revision": None,
        "mode": "heads",
        "objective": "soft",
        "num_draft_tokens": 2,
        "loss_weight": 0.3,
        "init_from_layer": None,
    }
    start = tmp_path / "mtp_start"
    start.mkdir()
    (start / "config.json").write_text("{}")
    (start / MANIFEST_NAME).write_text(json.dumps({"request": request}))
    assert _matches(start, request)
    for change in (
        {"loss_weight": 0.1},
        {"init_from_layer": 1},
        {"source": "somewhere/else"},
        {"num_draft_tokens": 3},
        {"objective": "hard"},
    ):
        assert not _matches(start, {**request, **change})
    assert not _matches(tmp_path / "missing", request)
    (start / "config.json").unlink()  # an unfinished start point is never reused
    assert not _matches(start, request)


def test_the_evaluator_uses_an_accelerator_only_when_there_is_one(monkeypatch):
    import torch

    from liquid_finetune.models.lfm3_mtp.evaluate import default_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert default_device() == "cpu"


def test_job_config_rejects_unsupported_mtp_combinations():
    from pydantic import ValidationError

    from liquid_finetune.config.job_config import JobConfig

    base = {
        "model_name": "LFM3-1.7B",
        "training_type": "sft",
        "dataset": {"path": "HuggingFaceTB/smoltalk", "type": "sft"},
        "mtp": {"num_draft_tokens": 2},
    }
    JobConfig(**base)
    with pytest.raises(ValidationError, match="adapter_path"):
        JobConfig(**base, training_config={"adapter_path": "./adapter"})
    with pytest.raises(ValidationError, match="quantization-aware"):
        JobConfig(**base, training_config={"qat": {"type": "vllm_fp8"}})
    with pytest.raises(ValidationError, match="output_pvc"):
        JobConfig(**base, kuberay={"head_gpu_count": 1})
    JobConfig(**base, kuberay={"head_gpu_count": 1, "output_pvc": "outputs"})


def test_evaluator_loads_images_one_prompt_at_a_time(tmp_path):
    from PIL import Image

    from liquid_finetune.models.lfm3_mtp.evaluate import _content_item, _open_images

    path = tmp_path / "red.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(path)
    item = _content_item({"type": "image", "image": str(path), "text": None}, True)
    assert item == {
        "type": "image",
        "image": str(path),
    }  # still a path, not a decoded image
    assert _content_item(item, False) is None  # text-only models drop images
    turns = [{"role": "user", "content": [item, {"type": "text", "text": "?"}]}]
    opened = _open_images(turns)
    assert isinstance(opened[0]["content"][0]["image"], Image.Image)
    assert turns[0]["content"][0]["image"] == str(
        path
    )  # the source prompt is unchanged
