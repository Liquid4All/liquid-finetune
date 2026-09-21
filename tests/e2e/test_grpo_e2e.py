"""End-to-end GRPO training smoke tests (GPU required).

These tests verify the whole pipeline: dataset loading, reward resolution,
vLLM colocate instantiation, multiple optimizer steps, and learning signals.
They deliberately use tiny settings (num_generations=2, 16 samples,
max_completion_length=16) so they finish in under ~5 minutes on 1 H100.

Run an individual case on a local GPU with:
    uv run pytest tests/e2e/test_grpo_e2e.py::TestDenseGRPO::test_text_grpo_optimizes -v

Submit the complete E2E matrix with:
    tests/e2e/slurm/submit_e2e_tests.sh
"""

import pathlib

import pytest

from conftest import (
    assert_grpo_optimization,
    requires_gpu,
    requires_single_gpu,
    run_e2e_training,
    run_local_e2e_training,
)

pytestmark = pytest.mark.dense

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class TestDenseGRPO:
    @requires_gpu
    def test_text_grpo_optimizes(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_grpo.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_grpo_optimization(result)

    @pytest.mark.single_gpu
    @requires_single_gpu
    def test_local_single_gpu_training(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_grpo.yaml")
        run_local_e2e_training(config_path, e2e_output_dir)
