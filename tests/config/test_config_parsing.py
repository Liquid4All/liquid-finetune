import pytest

from liquid_finetune import run_config
from liquid_finetune.config import (
    DatasetConfig,
    EvalConfig,
    EvalRunConfig,
    EvalSuiteConfig,
    JobConfig,
    TrainingConfig,
    materialize_job_config,
    parse_job_config,
    resolve_config_path,
)
from liquid_finetune.distribution.distributed_configs import (
    strip_distributed_training_config,
)
from liquid_finetune.training.default_configs import TRAINING_DEFAULTS

from conftest import BASE_DPO_DATASET, BASE_SFT_DATASET, write_config

pytestmark = pytest.mark.configs


class TestResolveConfigPath:
    def test_absolute_path(self, sft_config_path):
        result = resolve_config_path(sft_config_path)
        assert result.exists()
        assert result.name == "sft_example.yaml"

    def test_repo_job_configs(self):
        result = resolve_config_path("sft_example.yaml")
        assert result.exists()

    def test_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            resolve_config_path("nonexistent_config.yaml")


class TestExampleSmoke:
    @pytest.mark.parametrize(
        "fixture_name, expected_type",
        [
            ("sft_config_path", "sft"),
            ("dpo_config_path", "dpo"),
            ("vlm_config_path", "vlm_sft"),
            ("moe_sft_config_path", "moe_sft"),
            ("moe_dpo_config_path", "moe_dpo"),
        ],
    )
    def test_example_parses_and_materializes(
        self, request, fixture_name, expected_type
    ):
        config_path = request.getfixturevalue(fixture_name)
        parsed = parse_job_config(config_path)
        materialized = materialize_job_config(parsed)

        assert parsed.training_type in {
            expected_type,
            expected_type.removeprefix("moe_"),
        }
        assert materialized.training_type == expected_type
        assert materialized.dataset is not None
        assert isinstance(materialized.to_dict()["training_config"], dict)

    def test_parse_kto_example(self, kto_config_path):
        job = parse_job_config(kto_config_path)
        materialized = materialize_job_config(job)

        assert job.training_type == "kto"
        assert materialized.job_name == "my_kto_project"
        assert job.dataset.type == "kto"
        assert "beta" in materialized.training_config.value
        assert "desirable_weight" in materialized.training_config.value
        assert "deepspeed" in materialized.training_config.value


class TestVLMDataLoaderDefaults:
    @pytest.mark.parametrize("profile", ["DEFAULT_VLM_SFT", "DEFAULT_VLM_DPO"])
    def test_vlm_profiles_enable_async_collation(self, profile):
        cfg = TRAINING_DEFAULTS[profile]

        assert cfg["dataloader_num_workers"] == 2
        assert cfg["dataloader_prefetch_factor"] == 2
        assert cfg["dataloader_persistent_workers"] is True
        assert cfg["dataloader_pin_memory"] is True


class TestDirectPythonConfig:
    def test_construct_job_config_directly(self):
        job = JobConfig(
            project_name="py_job",
            model_name="LFM2-1.2B",
            training_type="sft",
            dataset=DatasetConfig(**BASE_SFT_DATASET),
            training_config=TrainingConfig(
                num_train_epochs=4,
                learning_rate=1e-4,
            ),
            evals=EvalSuiteConfig(
                benchmarks=[
                    EvalConfig(
                        name="toy_eval",
                        path="/tmp/eval.jsonl",
                        metric="exact_match",
                    )
                ]
            ),
        )

        materialized = materialize_job_config(job)
        resolved = materialized.training_config.value
        assert materialized.job_name == "py_job"
        assert resolved["training_type"] == "sft"
        assert resolved["num_train_epochs"] == 4
        assert resolved["learning_rate"] == 1e-4
        assert "deepspeed" in resolved
        assert job.evals is not None

    def test_run_config_accepts_job_model(self, monkeypatch):
        calls = {}

        def fake_ray_trainer(job_dict):
            calls["job_dict"] = job_dict
            return {"ok": True}

        monkeypatch.setattr(
            "liquid_finetune.cli.main._assert_local_cuda_available",
            lambda: None,
        )
        monkeypatch.setattr(
            "liquid_finetune.cli.main.check_and_handle_slurm",
            lambda *args, **kwargs: False,
        )
        monkeypatch.setattr(
            "liquid_finetune.distribution.backends.kuberay.check_and_handle_kuberay",
            lambda *args, **kwargs: False,
        )
        monkeypatch.setattr(
            "liquid_finetune.distribution.backends.modal.check_and_handle_modal",
            lambda *args, **kwargs: False,
        )
        monkeypatch.setattr(
            "liquid_finetune.training.utils.logging.setup_training_environment",
            lambda: None,
        )
        monkeypatch.setattr(
            "liquid_finetune.distribution.ray_trainer.ray_trainer",
            fake_ray_trainer,
        )
        monkeypatch.setattr(
            "liquid_finetune.data_loading.dataset_loader.DatasetLoader.quick_validate",
            lambda self: None,
        )

        job = JobConfig(
            project_name="py_run",
            model_name="LFM2-1.2B",
            training_type="sft",
            dataset=DatasetConfig(**BASE_SFT_DATASET),
            training_config=TrainingConfig(num_train_epochs=1),
        )

        run_config(job)
        assert calls["job_dict"]["job_name"] == "py_run"
        assert calls["job_dict"]["training_type"] == "sft"

    def test_run_config_accepts_eval_model(self, monkeypatch):
        calls = {}

        def fake_run_eval_config(config, *, output_path=None):
            calls["config"] = config
            calls["output_path"] = output_path
            return {"benchmark/tiny_qa/score": 1.0}

        monkeypatch.setattr(
            "liquid_finetune.evaluation.runner.run_eval_config",
            fake_run_eval_config,
        )

        eval_config = EvalRunConfig(
            model_name="LFM2-1.2B",
            evals=EvalSuiteConfig(
                benchmarks=[
                    EvalConfig(
                        name="tiny_qa",
                        path="/tmp/tiny_qa.jsonl",
                        metric="short_answer",
                    )
                ]
            ),
        )

        result = run_config(eval_config, output_path="/tmp/results.json")

        assert result == {"benchmark/tiny_qa/score": 1.0}
        assert calls["config"] is eval_config
        assert calls["output_path"] == "/tmp/results.json"

    def test_run_config_dispatches_eval_yaml_without_training(
        self, tmp_path, monkeypatch
    ):
        calls = {}

        def fake_run_eval_config(config, *, output_path=None):
            calls["config"] = config
            calls["output_path"] = output_path
            return {"benchmark/tiny_qa/score": 1.0}

        monkeypatch.setattr(
            "liquid_finetune.cli.main.check_and_handle_slurm",
            lambda *args, **kwargs: False,
        )
        monkeypatch.setattr(
            "liquid_finetune.distribution.backends.kuberay.check_and_handle_kuberay",
            lambda *args, **kwargs: False,
        )
        monkeypatch.setattr(
            "liquid_finetune.distribution.backends.modal.check_and_handle_modal",
            lambda *args, **kwargs: False,
        )
        monkeypatch.setattr(
            "liquid_finetune.cli.main._assert_local_cuda_available",
            lambda: pytest.fail("eval-only run_config should not require CUDA"),
        )
        monkeypatch.setattr(
            "liquid_finetune.evaluation.runner.run_eval_config",
            fake_run_eval_config,
        )

        cfg_path = write_config(
            {
                "model_name": "LFM2-1.2B",
                "evals": {
                    "benchmarks": [
                        {
                            "name": "tiny_qa",
                            "path": "/tmp/tiny_qa.jsonl",
                            "metric": "short_answer",
                        }
                    ]
                },
            },
            tmp_path,
        )

        result = run_config(cfg_path)

        assert result == {"benchmark/tiny_qa/score": 1.0}
        assert isinstance(calls["config"], EvalRunConfig)
        assert calls["output_path"] is None


class TestFocusedValidation:
    def test_evals_and_benchmarks_both_parse(self, tmp_path):
        base = {
            "project_name": "eval_alias",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"eval_strategy": "epoch"},
        }
        eval_suite = {
            "max_new_tokens": 32,
            "benchmarks": [
                {
                    "name": "toy_eval",
                    "path": "/tmp/eval.jsonl",
                    "metric": "exact_match",
                }
            ],
        }

        parsed_new = parse_job_config(
            write_config({**base, "evals": eval_suite}, tmp_path)
        )
        parsed_old = parse_job_config(
            write_config({**base, "benchmarks": eval_suite}, tmp_path)
        )

        assert parsed_new.evals is not None
        assert parsed_old.evals is not None
        assert parsed_new.evals.benchmarks[0].name == "toy_eval"
        assert parsed_old.evals.benchmarks[0].name == "toy_eval"

    def test_materialize_resolves_repo_relative_eval_paths(self, tmp_path):
        config = {
            "project_name": "repo_relative_eval",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"eval_strategy": "epoch"},
            "evals": {
                "benchmarks": [
                    {
                        "name": "tiny_qa",
                        "path": "tests/e2e/fixtures/tiny_qa_bench.jsonl",
                        "metric": "short_answer",
                    }
                ]
            },
        }

        materialized = materialize_job_config(
            parse_job_config(write_config(config, tmp_path))
        )

        assert materialized.benchmark_configs is not None
        assert materialized.benchmark_configs["benchmarks"][0]["path"].endswith(
            "/tests/e2e/fixtures/tiny_qa_bench.jsonl"
        )

    def test_invalid_dataset_path_combination_rejected(self, tmp_path):
        config = {
            "project_name": "bad",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": {
                **BASE_SFT_DATASET,
                "train_path": "other-dataset",
            },
        }
        with pytest.raises(ValueError, match="dataset.path or dataset.train_path"):
            parse_job_config(write_config(config, tmp_path))

    def test_unknown_extends_rejected(self, tmp_path):
        config = {
            "project_name": "bad",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "NOT_A_REAL_PROFILE"},
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        with pytest.raises(ValueError, match="Unknown base config"):
            materialize_job_config(parsed)

    def test_eval_strategy_uses_default_split_when_test_size_omitted(self, tmp_path):
        # Offline training types get the default 0.2 eval split when no explicit
        # validation source is given, so eval_strategy works without test_size.
        config = {
            "project_name": "default_eval",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": {
                "path": BASE_SFT_DATASET["path"],
                "type": "sft",
                "limit": 10,
            },
            "training_config": {
                "eval_strategy": "steps",
                "eval_steps": 10,
            },
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        materialized = materialize_job_config(parsed)
        assert materialized.training_config.value["eval_strategy"] == "steps"

    def test_training_type_defaults_apply_without_extends(self, tmp_path):
        config = {
            "project_name": "dpo_defaults",
            "model_name": "LFM2-1.2B",
            "training_type": "dpo",
            "dataset": BASE_DPO_DATASET,
            "training_config": {"num_train_epochs": 5, "learning_rate": 2e-6},
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        materialized = materialize_job_config(parsed)
        resolved = materialized.training_config.value
        assert resolved["training_type"] == "dpo"
        assert resolved["beta"] == 0.1
        assert resolved["num_train_epochs"] == 5
        assert resolved["learning_rate"] == 2e-6

    def test_single_worker_strips_distributed_training_config(self):
        train_config = {
            "learning_rate": 1e-5,
            "deepspeed": {"zero_optimization": {"stage": 2}},
            "fsdp": ["full_shard"],
            "fsdp_config": {"use_orig_params": True},
        }

        stripped = strip_distributed_training_config(train_config, num_workers=1)

        assert stripped["learning_rate"] == 1e-5
        assert "deepspeed" not in stripped
        assert "fsdp" not in stripped
        assert "fsdp_config" not in stripped
        assert "deepspeed" in train_config

    def test_peft_extends_still_works(self, tmp_path):
        config = {
            "project_name": "peft",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"extends": "DEFAULT_LORA", "use_peft": True, "r": 32},
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        materialized = materialize_job_config(parsed)
        assert materialized.peft_config is not None
        assert materialized.peft_config.value.r == 32
