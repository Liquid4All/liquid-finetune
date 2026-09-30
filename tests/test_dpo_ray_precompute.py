from types import SimpleNamespace

import torch
from datasets import Dataset

from leap_finetune.training.utils.trainer_mixins import RayPrecomputedRefLogpsMixin


class _LocalShardTrainer(RayPrecomputedRefLogpsMixin):
    def __init__(self):
        self.args = SimpleNamespace(
            dataloader_num_workers=0, dataloader_pin_memory=False, data_seed=42, seed=42
        )
        self.ref_model = object()
        self.model = object()
        self.is_deepspeed_enabled = False
        self.data_collator = self._collate

    @staticmethod
    def _collate(examples):
        return {
            "chosen": torch.tensor([example["chosen"] for example in examples]),
            "rejected": torch.tensor([example["rejected"] for example in examples]),
        }

    @staticmethod
    def _prepare_inputs(inputs):
        return inputs

    @staticmethod
    def compute_ref_log_probs(model, inputs):
        return inputs["chosen"], inputs["rejected"]


def test_precompute_ref_logps_stays_aligned_with_local_ray_shard(monkeypatch):
    monkeypatch.setenv("TQDM_DISABLE", "1")
    local_shard = Dataset.from_dict(
        {
            "sample_id": [11, 37, 52],
            "chosen": [-11.5, -37.5, -52.5],
            "rejected": [-12.5, -38.5, -53.5],
        }
    )

    torch.manual_seed(1234)
    expected_rng_values = torch.rand(4)
    torch.manual_seed(1234)

    result = _LocalShardTrainer()._precompute_ref_logps(
        local_shard, "train", batch_size=2
    )

    assert result["sample_id"] == [11, 37, 52]
    assert result["ref_chosen_logps"] == [-11.5, -37.5, -52.5]
    assert result["ref_rejected_logps"] == [-12.5, -38.5, -53.5]
    assert torch.equal(torch.rand(4), expected_rng_values)
