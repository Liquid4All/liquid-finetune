import pytest
from datasets import Dataset


class TestKTOFormat:
    def test_missing_columns_rejected(self):
        from liquid_finetune.data_loading.validate_dataset_format import (
            validate_kto_format,
        )

        ds = Dataset.from_list([{"prompt": "Hi", "completion": "Hello"}])
        with pytest.raises(ValueError, match="label"):
            validate_kto_format(ds)

    def test_empty_completion_rejected(self):
        from liquid_finetune.data_loading.validate_dataset_format import (
            validate_kto_format,
        )

        ds = Dataset.from_list([{"prompt": "Hi", "completion": "", "label": True}])
        with pytest.raises(ValueError, match="invalid prompt/completion"):
            validate_kto_format(ds)

    def test_valid_rows_pass(self):
        from liquid_finetune.data_loading.validate_dataset_format import (
            validate_kto_format,
        )

        ds = Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": "Capital of France?"}],
                    "completion": [{"role": "assistant", "content": "Paris."}],
                    "label": True,
                },
                {
                    "prompt": [{"role": "user", "content": "Capital of France?"}],
                    "completion": [{"role": "assistant", "content": "London."}],
                    "label": False,
                },
            ]
        )
        validate_kto_format(ds)

    def test_row_filter_accepts_valid_and_rejects_invalid(self):
        from liquid_finetune.data_loading.validate_dataset_format import get_row_filter

        f = get_row_filter("kto")
        valid = {
            "prompt": [{"role": "user", "content": "Hi"}],
            "completion": [{"role": "assistant", "content": "Hello"}],
            "label": True,
        }
        assert f(valid) is True
        assert f({**valid, "label": None}) is False
        assert f({**valid, "label": "yes"}) is False
        assert f({**valid, "completion": []}) is False
        assert f({"prompt": "Hi", "completion": "Hello", "label": False}) is True

    def test_row_filter_rejects_foreign_tool_markers(self):
        from liquid_finetune.data_loading.validate_dataset_format import get_row_filter

        f = get_row_filter("kto")
        assert (
            f(
                {
                    "prompt": "Get weather",
                    "completion": "<tool_call>x</tool_call>",
                    "label": True,
                }
            )
            is False
        )

    def test_invalid_message_shape_rejected(self):
        from liquid_finetune.data_loading.validate_dataset_format import (
            validate_kto_format,
        )

        ds = Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user"}],
                    "completion": [{"role": "assistant", "content": "Hello"}],
                    "label": True,
                }
            ]
        )
        with pytest.raises(ValueError, match="invalid prompt/completion"):
            validate_kto_format(ds)


class TestKTODataLoader:
    def test_preserves_order_and_drops_incomplete_batches(self):
        from types import SimpleNamespace

        from liquid_finetune.training.kto import LFMKTOTrainer

        trainer = LFMKTOTrainer.__new__(LFMKTOTrainer)
        trainer.train_dataset = list(range(5))
        trainer.eval_dataset = list(range(5))
        trainer.args = SimpleNamespace(
            per_device_train_batch_size=2, per_device_eval_batch_size=1
        )
        trainer.data_collator = lambda rows: rows

        assert list(trainer.get_train_dataloader()) == [[0, 1], [2, 3]]
        assert list(trainer.get_eval_dataloader()) == [[0, 1], [2, 3]]
