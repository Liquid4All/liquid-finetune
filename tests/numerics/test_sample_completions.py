import json
import sys
import types
from types import SimpleNamespace

import pytest

from liquid_finetune.checkpointing import model_loading
from liquid_finetune.cli import sample_completions


def _completion(text, finish_reason="stop"):
    return SimpleNamespace(text=text, finish_reason=finish_reason)


def _write_prompts(path):
    rows = [
        {"prompt": "What is 2 + 2?"},
        {
            "messages": [
                {"role": "user", "content": "Name a color."},
                {"role": "assistant", "content": "Blue."},
            ]
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return {"input_ids": [1] + [10 + len(m["content"]) for m in messages]}


def test_load_prompts_reads_prompt_and_message_rows(tmp_path):
    path = tmp_path / "prompts.jsonl"
    _write_prompts(path)

    assert sample_completions.load_prompts(str(path)) == [
        [{"role": "user", "content": "What is 2 + 2?"}],
        [{"role": "user", "content": "Name a color."}],
    ]
    assert len(sample_completions.load_prompts(str(path), limit=1)) == 1


def test_messages_rows_drop_truncated_completions_by_default():
    prompts = [[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]]
    outputs = [
        SimpleNamespace(outputs=[_completion("a"), _completion("b", "length")]),
        SimpleNamespace(outputs=[_completion("c")]),
    ]

    rows = sample_completions.messages_rows(prompts, outputs)
    assert rows == [
        {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "c"},
            ]
        },
    ]
    kept = sample_completions.messages_rows(prompts, outputs, keep_truncated=True)
    assert [row["messages"][-1]["content"] for row in kept] == ["a", "b", "c"]


def test_main_samples_with_vllm_and_writes_sft_rows(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "round" / "samples.jsonl"
    _write_prompts(prompts_path)
    calls = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["llm"] = kwargs

        def generate(self, inputs, sampling_params):
            calls["inputs"] = inputs
            return [
                SimpleNamespace(
                    outputs=[
                        _completion(f"answer {index}.{sample}")
                        for sample in range(sampling_params.n)
                    ]
                )
                for index in range(len(inputs))
            ]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(
        model_loading, "load_tokenizer", lambda *args, **kwargs: _FakeTokenizer()
    )

    sample_completions.main(
        [
            "--model",
            "example-org/student",
            "--dataset",
            str(prompts_path),
            "--output",
            str(output_path),
            "--num-samples",
            "2",
            "--max-model-len",
            "64",
        ]
    )

    assert calls["llm"]["model"] == "example-org/student"
    assert calls["llm"]["max_model_len"] == 64
    assert calls["inputs"] == [
        {"prompt_token_ids": [1, 24]},
        {"prompt_token_ids": [1, 23]},
    ]
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 4
    assert rows[0]["messages"] == [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "answer 0.0"},
    ]


def test_main_explains_missing_vllm(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(SystemExit, match="requires vLLM"):
        sample_completions.main(
            ["--model", "m", "--dataset", "d.jsonl", "--output", str(tmp_path / "o")]
        )
