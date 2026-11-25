import os

import pytest

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from trl.experimental.distillation import DistillationConfig  # noqa: E402

from liquid_finetune.config import materialize_job_config, parse_job_config  # noqa: E402
from liquid_finetune.config.parser import TRAINING_TYPE_TO_CONFIG  # noqa: E402
from liquid_finetune.distribution.distributed_configs import (  # noqa: E402
    strip_distributed_training_config,
)
from liquid_finetune.training import (  # noqa: E402
    ROLLOUT_TRAINING_TYPES,
    TRAINING_DATASETS_TO_SPLIT,
    TRAINING_LOOPS,
)
from liquid_finetune.training.default_configs import TRAINING_DEFAULTS  # noqa: E402
from liquid_finetune.training.default_configs.opd_configs import (  # noqa: E402
    OPD_EXCLUDED_KEYS,
)
from liquid_finetune.training.utils.config_filter import (  # noqa: E402
    filter_runtime_config_kwargs,
)

from conftest import BASE_DPO_DATASET, BASE_SFT_DATASET, write_config  # noqa: E402

pytestmark = pytest.mark.configs

PROMPT_DATASET = {"path": "trl-lib/DeepMath-103K", "type": "grpo", "limit": 10}


@pytest.fixture(autouse=True)
def _isolated_output_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))


def _opd_config(dataset=PROMPT_DATASET, **training_config):
    return {
        "project_name": "opd_test",
        "model_name": "LFM2-1.2B",
        "training_type": "opd",
        "dataset": dataset,
        "training_config": {
            "teacher_model_name_or_path": "LFM2-2.6B",
            **training_config,
        },
    }


def _materialize(config, tmp_path):
    return materialize_job_config(parse_job_config(write_config(config, tmp_path)))


class TestOPDDefaults:
    def test_stage_is_registered_as_rollout_type(self):
        assert TRAINING_TYPE_TO_CONFIG["opd"] == "DEFAULT_OPD"
        assert "opd" in ROLLOUT_TRAINING_TYPES
        assert TRAINING_DATASETS_TO_SPLIT["opd"] == ()
        assert "opd" in TRAINING_LOOPS

    def test_default_config_is_reverse_kl_with_vllm(self):
        cfg = TRAINING_DEFAULTS["DEFAULT_OPD"]
        assert cfg["training_type"] == "opd"
        assert cfg["beta"] == 1.0
        assert (cfg["temperature"], cfg["top_p"], cfg["top_k"]) == (1.0, 1.0, 0)
        assert cfg["use_vllm"] is True
        assert cfg["vllm_mode"] == "colocate"
        assert cfg["eval_strategy"] == "no"
        assert cfg["loss_top_k"] is None

    def test_example_parses_and_materializes(self, job_configs_dir):
        parsed = parse_job_config(str(job_configs_dir / "opd_example.yaml"))
        materialized = materialize_job_config(parsed)
        resolved = materialized.training_config.value

        assert materialized.training_type == "opd"
        assert materialized.dataset.dataset_type == "sft"
        assert resolved["teacher_model_name_or_path"] == "LFM2-2.6B"
        assert resolved["beta"] == 1.0

    def test_prompt_only_dataset_and_rollout_server(self, tmp_path):
        config = _opd_config(vllm_mode="server", vllm_server_port=8000)
        config["grpo_rollout"] = {"server_gpus": 1, "tensor_parallel_size": 1}
        materialized = _materialize(config, tmp_path)

        assert materialized.dataset.dataset_type == "grpo"
        assert materialized.grpo_rollout == {
            "server_gpus": 1,
            "tensor_parallel_size": 1,
        }
        assert materialized.training_config.value["vllm_mode"] == "server"

    def test_trainer_config_receives_only_distillation_fields(self, tmp_path):
        resolved = _materialize(
            _opd_config(loss_top_k=64, teacher_model_revision="main"), tmp_path
        ).training_config.value
        filtered, dropped = filter_runtime_config_kwargs(
            strip_distributed_training_config(resolved, num_workers=1),
            excluded_keys=OPD_EXCLUDED_KEYS,
            config_cls=DistillationConfig,
        )

        assert dropped == []
        for key in ("teacher_model_name_or_path", "loss_top_k", "loss_add_tail"):
            assert key not in filtered
        filtered.update(use_cpu=True, bf16=False)
        args = DistillationConfig(**filtered)
        assert args.beta == 1.0
        assert args.max_prompt_length == 2048 - 512


class TestOPDValidation:
    def test_rewards_stay_grpo_only(self, tmp_path):
        config = _opd_config()
        config["rewards"] = ["./rewards/length.py::length_reward"]
        with pytest.raises(ValueError, match="only valid for training_type"):
            parse_job_config(write_config(config, tmp_path))

    @pytest.mark.parametrize(
        "training_config, message",
        [
            ({"teacher_model_name_or_path": None}, "teacher_model_name_or_path"),
            ({"beta": -0.5}, "beta"),
            ({"loss_top_k": 0}, "loss_top_k"),
            ({"lmbda": 0.5}, "always trains on student generations"),
            ({"seq_kd": True}, "always trains on student generations"),
            ({"num_generations": 3}, "divisible by num_generations"),
            ({"num_generations": 0}, "num_generations"),
            ({"generation_batch_size": 3}, "generation_batch_size"),
            ({"max_completion_length": 4096}, "max_completion_length"),
            ({"vllm_mode": "remote"}, "vllm_mode"),
            ({"qat": {"type": "gguf_q4_0"}}, "qat"),
        ],
    )
    def test_invalid_training_config_rejected(self, tmp_path, training_config, message):
        config = _opd_config()
        config["training_config"].update(training_config)
        config["training_config"] = {
            key: value
            for key, value in config["training_config"].items()
            if value is not None
        }
        with pytest.raises(ValueError, match=message):
            _materialize(config, tmp_path)

    def test_conversation_dataset_accepted(self, tmp_path):
        materialized = _materialize(_opd_config(dataset=BASE_SFT_DATASET), tmp_path)
        assert materialized.dataset.dataset_type == "sft"

    def test_preference_dataset_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="dataset.type"):
            _materialize(_opd_config(dataset=BASE_DPO_DATASET), tmp_path)
