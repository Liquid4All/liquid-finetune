import os

import pytest

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from trl.experimental.gkd import GKDConfig  # noqa: E402

from liquid_finetune.config import materialize_job_config, parse_job_config  # noqa: E402
from liquid_finetune.config.parser import TRAINING_TYPE_TO_CONFIG  # noqa: E402
from liquid_finetune.distribution.distributed_configs import (  # noqa: E402
    strip_distributed_training_config,
)
from liquid_finetune.training import (  # noqa: E402
    PRETOKENIZED_TRAINING_TYPES,
    TRAINING_LOOPS,
)
from liquid_finetune.training.default_configs import TRAINING_DEFAULTS  # noqa: E402
from liquid_finetune.training.default_configs.distill_configs import (  # noqa: E402
    DISTILL_EXCLUDED_KEYS,
)
from liquid_finetune.training.utils.config_filter import (  # noqa: E402
    filter_runtime_config_kwargs,
)

from conftest import BASE_DPO_DATASET, BASE_SFT_DATASET, write_config  # noqa: E402

pytestmark = pytest.mark.configs


@pytest.fixture(autouse=True)
def _isolated_output_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))


def _distill_config(**training_config):
    return {
        "project_name": "distill_test",
        "model_name": "LFM2-1.2B",
        "training_type": "distill",
        "dataset": BASE_SFT_DATASET,
        "training_config": {
            "teacher_model_name_or_path": "LFM2-2.6B",
            **training_config,
        },
    }


def _materialize(config, tmp_path):
    return materialize_job_config(parse_job_config(write_config(config, tmp_path)))


class TestDistillDefaults:
    def test_stage_is_registered(self):
        assert TRAINING_TYPE_TO_CONFIG["distill"] == "DEFAULT_DISTILL"
        assert "distill" in PRETOKENIZED_TRAINING_TYPES
        assert "distill" in TRAINING_LOOPS

    def test_default_config(self):
        cfg = TRAINING_DEFAULTS["DEFAULT_DISTILL"]
        assert cfg["training_type"] == "distill"
        assert cfg["beta"] == 0.0
        assert cfg["lmbda"] == 0.0
        assert cfg["loss_top_k"] is None
        assert cfg["loss_add_tail"] is True
        assert cfg["assistant_only_loss"] is True
        assert "deepspeed" in cfg

    def test_example_parses_and_materializes(self, distill_config_path):
        parsed = parse_job_config(distill_config_path)
        materialized = materialize_job_config(parsed)
        resolved = materialized.training_config.value

        assert parsed.training_type == "distill"
        assert materialized.training_type == "distill"
        assert materialized.dataset.dataset_type == "sft"
        assert resolved["teacher_model_name_or_path"] == "LFM2-2.6B"
        assert resolved["chat_template_path"].endswith(
            "job_configs/chat_templates/lfm2_chat_template.jinja"
        )

    def test_defaults_apply_without_extends(self, tmp_path):
        materialized = _materialize(_distill_config(beta=1.0), tmp_path)
        resolved = materialized.training_config.value
        assert resolved["beta"] == 1.0
        assert resolved["lmbda"] == 0.0
        assert resolved["learning_rate"] == 2e-5

    def test_trainer_config_receives_only_gkd_fields(self, tmp_path):
        resolved = _materialize(
            _distill_config(loss_top_k=64, teacher_model_revision="main"), tmp_path
        ).training_config.value
        filtered, dropped = filter_runtime_config_kwargs(
            strip_distributed_training_config(resolved, num_workers=1),
            excluded_keys=DISTILL_EXCLUDED_KEYS,
            config_cls=GKDConfig,
        )

        assert dropped == []
        for key in (
            "teacher_model_name_or_path",
            "teacher_model_revision",
            "loss_top_k",
            "loss_add_tail",
            "assistant_only_loss",
            "max_length",
        ):
            assert key not in filtered
        filtered.update(use_cpu=True, bf16=False)
        args = GKDConfig(**filtered)
        assert (args.lmbda, args.beta, args.seq_kd) == (0.0, 0.0, False)


class TestDistillValidation:
    @pytest.mark.parametrize("top_k", [1, 64])
    def test_valid_top_k(self, tmp_path, top_k):
        resolved = _materialize(
            _distill_config(loss_top_k=top_k, loss_add_tail=False), tmp_path
        ).training_config.value
        assert resolved["loss_top_k"] == top_k
        assert resolved["loss_add_tail"] is False

    def test_on_policy_mix_with_assistant_masks(self, tmp_path):
        resolved = _materialize(
            _distill_config(lmbda=0.5, beta=0.5), tmp_path
        ).training_config.value
        assert resolved["lmbda"] == 0.5

    @pytest.mark.parametrize(
        "training_config, message",
        [
            ({"teacher_model_name_or_path": None}, "teacher_model_name_or_path"),
            ({"teacher_model_name_or_path": " "}, "teacher_model_name_or_path"),
            ({"teacher_model_init_kwargs": {"dtype": "float32"}}, "init_kwargs"),
            ({"beta": 1.5}, "beta"),
            ({"lmbda": -0.1}, "lmbda"),
            ({"loss_top_k": 0}, "loss_top_k"),
            ({"loss_top_k": True}, "loss_top_k"),
            ({"loss_top_k": "64"}, "loss_top_k"),
            ({"loss_add_tail": "yes"}, "loss_add_tail"),
            ({"loss_top_k": 64, "use_liger_kernel": True}, "use_liger_kernel"),
            ({"packing": True}, "packing"),
            ({"padding_free": True}, "padding_free"),
            ({"qat": {"type": "gguf_q4_0"}}, "qat"),
            ({"lmbda": 0.5, "assistant_only_loss": False}, "prompt boundary"),
            ({"seq_kd": True, "assistant_only_loss": False}, "prompt boundary"),
        ],
    )
    def test_invalid_training_config_rejected(self, tmp_path, training_config, message):
        config = _distill_config()
        config["training_config"].update(training_config)
        config["training_config"] = {
            key: value
            for key, value in config["training_config"].items()
            if value is not None
        }
        with pytest.raises(ValueError, match=message):
            _materialize(config, tmp_path)

    def test_dataset_must_be_conversational_sft(self, tmp_path):
        config = _distill_config()
        config["dataset"] = BASE_DPO_DATASET
        with pytest.raises(ValueError, match="dataset.type: 'sft'"):
            _materialize(config, tmp_path)

    def test_moe_student_rejected(self, tmp_path):
        config = _distill_config()
        config["model_name"] = "LFM2-8B-A1B"
        with pytest.raises(ValueError, match="MoE"):
            _materialize(config, tmp_path)
