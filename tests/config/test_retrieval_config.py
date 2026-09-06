import pytest
import yaml

from liquid_finetune.config.parser import materialize_job_config, parse_job_config

pytestmark = pytest.mark.configs


@pytest.mark.parametrize("dataset_type", ["embedding", "colbert"])
def test_retrieval_config_materializes_defaults(tmp_path, dataset_type):
    data_path = tmp_path / "retrieval.jsonl"
    data_path.write_text(
        '{"query":"q","positive":"p","negative":"n"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
project_name: retrieval-test
model_name: LiquidAI/LFM2.5-Embedding-350M
training_type: {dataset_type}
dataset:
  path: {data_path}
  type: {dataset_type}
  test_size: 0.5
training_config: {{}}
peft_config:
  use_peft: false
""",
        encoding="utf-8",
    )
    job = materialize_job_config(parse_job_config(config_path))
    assert job.training_type == dataset_type
    assert job.dataset.dataset_type == dataset_type
    assert job.training_config.value["gather_across_devices"] is True
    assert job.training_config.value["gradient_accumulation_steps"] == 1


def test_retrieval_config_rejects_type_mismatch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project_name: mismatch
model_name: LiquidAI/LFM2.5-Embedding-350M
training_type: embedding
dataset:
  path: dataset.jsonl
  type: colbert
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="matching training_type"):
        parse_job_config(config_path)


def test_retrieval_config_rejects_peft(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project_name: peft
model_name: LiquidAI/LFM2.5-Embedding-350M
training_type: embedding
dataset:
  path: dataset.jsonl
  type: embedding
peft_config:
  use_peft: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="PEFT is not yet supported"):
        parse_job_config(config_path)


def _write_retrieval_config(tmp_path, training_type, training_config):
    data_path = tmp_path / "retrieval.jsonl"
    data_path.write_text(
        '{"query":"q","positive":"p","negative":"n"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "typed-config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_name": "typed-retrieval",
                "model_name": "LiquidAI/LFM2.5-Embedding-350M",
                "training_type": training_type,
                "dataset": {"path": str(data_path), "type": training_type},
                "training_config": training_config,
                "peft_config": {"use_peft": False},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_retrieval_values_are_typed_by_pydantic(tmp_path):
    config_path = _write_retrieval_config(
        tmp_path,
        "embedding",
        {
            "loss": "cached_multiple_negatives_ranking",
            "mini_batch_size": "8",
            "gather_across_devices": "false",
            "prompts": {"query": "search: "},
        },
    )

    parsed = parse_job_config(config_path)
    assert parsed.training_config.mini_batch_size == 8
    assert parsed.training_config.gather_across_devices is False
    assert parsed.training_config.prompts.query == "search: "
    assert parsed.training_config.prompts.positive == "document: "

    resolved = materialize_job_config(parsed).training_config.value
    assert resolved["mini_batch_size"] == 8
    assert resolved["gather_across_devices"] is False
    assert resolved["prompts"]["positive"] == "document: "


@pytest.mark.parametrize(
    "training_type,training_config,error",
    [
        ("embedding", {"loss": "contrastive"}, "Unsupported embedding loss"),
        (
            "colbert",
            {"loss": "multiple_negatives_ranking"},
            "Unsupported colbert loss",
        ),
        (
            "embedding",
            {"gradient_accumulation_steps": 2},
            "gradient_accumulation_steps=1",
        ),
        (
            "embedding",
            {"mini_batch_size": 8},
            "mini_batch_size requires a cached retrieval loss",
        ),
        (
            "embedding",
            {"extends": "DEFAULT_COLBERT"},
            "requires training_config.extends",
        ),
        (
            "embedding",
            {"temperature": 0.02},
            "temperature is only valid for ColBERT",
        ),
        (
            "colbert",
            {"prompts": {"query": "search: "}},
            "prompts is only valid for embedding",
        ),
        ("colbert", {"temperature": 0}, "greater than 0"),
    ],
)
def test_invalid_retrieval_config_is_rejected_by_pydantic(
    tmp_path, training_type, training_config, error
):
    config_path = _write_retrieval_config(tmp_path, training_type, training_config)
    with pytest.raises(ValueError, match=error):
        parse_job_config(config_path)
