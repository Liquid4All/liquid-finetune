import pytest
from ray.exceptions import RayTaskError
from ray.train.torch.config import _TorchBackend

from liquid_finetune.distribution.ray_trainer import _ColocatedVLLMTorchBackend


def _ray_error(message: str) -> RayTaskError:
    return RayTaskError(
        "shutdown", message, AssertionError(message), ip="127.0.0.1", pid=1
    )


def test_colocated_vllm_shutdown_tolerates_released_process_group(monkeypatch):
    def fail_shutdown(*args, **kwargs):
        raise _ray_error("Process group cannot be None")

    monkeypatch.setattr(_TorchBackend, "on_shutdown", fail_shutdown)
    _ColocatedVLLMTorchBackend().on_shutdown(None, None)


def test_colocated_vllm_shutdown_preserves_other_worker_failures(monkeypatch):
    error = _ray_error("worker failed")

    def fail_shutdown(*args, **kwargs):
        raise error

    monkeypatch.setattr(_TorchBackend, "on_shutdown", fail_shutdown)
    with pytest.raises(RayTaskError, match="worker failed"):
        _ColocatedVLLMTorchBackend().on_shutdown(None, None)
