import pytest

from liquid_finetune.distribution import local_trainer


pytestmark = pytest.mark.configs


def _job(training_type="grpo", *, vllm_mode="colocate", **overrides):
    job = {
        "training_type": training_type,
        "model_name": "LFM2-1.2B",
        "training_config": {"vllm_mode": vllm_mode},
    }
    job.update(overrides)
    return job


@pytest.fixture(autouse=True)
def one_visible_gpu(monkeypatch):
    monkeypatch.setattr(local_trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(local_trainer.torch.cuda, "device_count", lambda: 1)
    monkeypatch.delenv("LIQUID_LAUNCHER", raising=False)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("LIQUID_NUM_WORKERS", raising=False)


@pytest.mark.parametrize("training_type", ["grpo", "vlm_grpo"])
def test_grpo_uses_local_path_only_for_one_gpu_colocate(training_type):
    assert local_trainer.should_use_local(_job(training_type))
    assert not local_trainer.should_use_local(_job(training_type, vllm_mode="server"))
    assert not local_trainer.should_use_local(
        _job(training_type, grpo_rollout={"tensor_parallel_size": 2})
    )


def test_grpo_respects_explicit_ray_dispatch(monkeypatch):
    monkeypatch.setenv("LIQUID_LAUNCHER", "ray")
    assert not local_trainer.should_use_local(_job())


def test_grpo_respects_ray_worker_count(monkeypatch):
    monkeypatch.setenv("LIQUID_NUM_WORKERS", "2")
    assert not local_trainer.should_use_local(_job())
