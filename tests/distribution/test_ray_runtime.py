import os
from types import SimpleNamespace

import pytest

from leap_finetune.distribution import ray_runtime


def _usage(*, free: int, total: int | None = None):
    total = total or free
    return SimpleNamespace(total=total, used=total - free, free=free)


def test_select_ray_temp_dir_honors_explicit_override(tmp_path, monkeypatch):
    requested = tmp_path / "ray-runtime"
    monkeypatch.setenv("RAY_TMPDIR", str(requested))

    selected = ray_runtime.select_ray_temp_dir()

    assert selected == str(requested)
    assert requested.is_dir()


def test_select_ray_temp_dir_scopes_slurm_temp_to_tmpdir(tmp_path, monkeypatch):
    monkeypatch.delenv("RAY_TMPDIR", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "1234.test")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    selected = ray_runtime.select_ray_temp_dir()

    assert selected == str(tmp_path / "r1234test")
    assert (tmp_path / "r1234test").is_dir()


def test_select_object_spilling_dir_is_under_ray_temp_dir(tmp_path):
    spill_dir = ray_runtime.select_object_spilling_dir(str(tmp_path / "ray"))

    assert spill_dir == str(tmp_path / "ray" / "spill")
    assert (tmp_path / "ray" / "spill").is_dir()


def test_object_store_size_scales_with_available_memory_and_shm(monkeypatch):
    monkeypatch.delenv("LEAP_RAY_OBJECT_STORE_MEMORY", raising=False)
    monkeypatch.setattr(
        ray_runtime.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4 * 1024**3),
    )
    monkeypatch.setattr(
        ray_runtime.shutil,
        "disk_usage",
        lambda path: (
            _usage(free=4 * 1024**3) if path == "/dev/shm" else _usage(free=8 * 1024**3)
        ),
    )

    assert ray_runtime.resolve_local_object_store_memory() == int(4 * 1024**3 * 0.10)


def test_object_store_size_uses_tmp_when_shm_is_too_small(tmp_path, monkeypatch):
    monkeypatch.delenv("LEAP_RAY_OBJECT_STORE_MEMORY", raising=False)
    monkeypatch.delenv("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", raising=False)
    monkeypatch.setattr(
        ray_runtime.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=64 * 1024**3),
    )

    def disk_usage(path):
        if path == "/dev/shm":
            return _usage(free=64 * 1024**2)
        return _usage(free=2 * 1024**3)

    monkeypatch.setattr(ray_runtime.shutil, "disk_usage", disk_usage)

    size = ray_runtime.resolve_local_object_store_memory(str(tmp_path))

    assert size == 1 * 1024**3
    assert os.environ.get("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE") == "1"


def test_object_store_size_does_not_use_old_large_fixed_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("LEAP_RAY_OBJECT_STORE_MEMORY", raising=False)
    monkeypatch.setattr(
        ray_runtime.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=128 * 1024**3),
    )
    monkeypatch.setattr(
        ray_runtime.shutil,
        "disk_usage",
        lambda path: (
            _usage(free=64 * 1024**2)
            if path == "/dev/shm"
            else _usage(free=512 * 1024**2)
        ),
    )

    size = ray_runtime.resolve_local_object_store_memory(str(tmp_path))

    assert size == int(512 * 1024**2 * 0.8)
    assert size < 2 * 1024**3


def test_object_store_memory_can_be_explicitly_overridden(monkeypatch):
    configured = 256 * 1024**2
    monkeypatch.setenv("LEAP_RAY_OBJECT_STORE_MEMORY", str(configured))

    assert ray_runtime.resolve_local_object_store_memory() == configured


@pytest.mark.parametrize("value", ["not-a-size", "1"])
def test_object_store_memory_override_is_validated(monkeypatch, value):
    monkeypatch.setenv("LEAP_RAY_OBJECT_STORE_MEMORY", value)

    with pytest.raises(ValueError, match="LEAP_RAY_OBJECT_STORE_MEMORY"):
        ray_runtime.resolve_local_object_store_memory()
