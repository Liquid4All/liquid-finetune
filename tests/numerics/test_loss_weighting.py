from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from leap_finetune.loss_weighting.alignment import (
    build_token_loss_weights,
)
from leap_finetune.loss_weighting.config import LossWeightingConfig
from leap_finetune.loss_weighting.fingerprint import loss_weighting_fingerprint_data
from leap_finetune.loss_weighting.loss import weighted_causal_lm_loss
from leap_finetune.loss_weighting.selectors import select_spans
from leap_finetune.training.utils.trainer_mixins import TokenWeightedLossMixin


def _config(default_weight=0.2, rules=None):
    return LossWeightingConfig.model_validate(
        {
            "default_weight": default_weight,
            "rules": rules or [],
        }
    )


def test_markdown_selector_handles_repeated_and_missing_fields():
    text = "product_category: Shoes\nother: x\nproduct_category: Boots\n"
    selector = {
        "type": "key_value_line",
        "key": "product_category",
        "include_key": True,
        "include_value": True,
    }

    spans = select_spans(text, selector)

    assert [text[start:end] for start, end in spans] == [
        "product_category: Shoes",
        "product_category: Boots",
    ]
    assert select_spans(text, {**selector, "key": "missing"}) == []


def test_json_pointer_and_regex_capture_selectors():
    text = '{"catalog":{"category":"Shoes"},"score":0.9}'

    json_spans = select_spans(
        text,
        {
            "type": "json_pointer",
            "pointer": "/catalog/category",
            "include_key": False,
            "include_delimiter": False,
            "include_value": True,
        },
    )
    regex_spans = select_spans(
        text,
        {
            "type": "regex",
            "pattern": r'"score":(?P<value>[0-9.]+)',
            "group": "value",
        },
    )

    assert [text[start:end] for start, end in json_spans] == ['"Shoes"']
    assert [text[start:end] for start, end in regex_spans] == ["0.9"]


def test_actual_tokenizer_merged_boundary_uses_overlap_rule():
    text = "product_category: Shoes"
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "product_category:": 1,
                "Shoes": 2,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    encoding = tokenizer.encode(text)
    assert encoding.tokens[0] == "product_category:"

    weights = build_token_loss_weights(
        messages=[{"role": "assistant", "content": text}],
        rendered_text=text,
        offset_mapping=encoding.offsets,
        assistant_mask=[1, 1],
        config=_config(
            rules=[
                {
                    "name": "value",
                    "selector": {
                        "type": "key_value_line",
                        "key": "product_category",
                        "include_key": False,
                        "include_delimiter": True,
                        "include_value": True,
                    },
                    "weight": 1.0,
                }
            ]
        ),
    )

    # The first token contains both key and delimiter. Character overlap makes
    # its boundary behavior explicit instead of assuming ':' is its own token.
    assert weights == [1.0, 1.0]


def test_rules_are_last_wins_and_prompt_tokens_remain_zero():
    text = "prompt product_category: Shoes tail"
    weights = build_token_loss_weights(
        messages=[
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "product_category: Shoes"},
        ],
        rendered_text=text,
        offset_mapping=[(0, 6), (7, 24), (25, 30), (31, 35)],
        assistant_mask=[0, 1, 1, 0],
        config=_config(
            rules=[
                {
                    "name": "line",
                    "selector": {
                        "type": "key_value_line",
                        "key": "product_category",
                        "include_key": True,
                    },
                    "weight": 2.0,
                },
                {
                    "name": "shoes",
                    "selector": {
                        "type": "regex",
                        "pattern": "(Shoes)",
                        "group": 1,
                    },
                    "weight": 3.0,
                },
            ]
        ),
    )

    assert weights == [0.0, 2.0, 3.0, 0.0]


def test_repeated_prompt_text_aligns_to_assistant_region():
    text = "product_category: Shoes product_category: Shoes"
    weights = build_token_loss_weights(
        messages=[
            {"role": "user", "content": "product_category: Shoes"},
            {"role": "assistant", "content": "product_category: Shoes"},
        ],
        rendered_text=text,
        offset_mapping=[(0, 23), (24, 47)],
        assistant_mask=[0, 1],
        config=_config(
            default_weight=0.2,
            rules=[
                {
                    "name": "category",
                    "selector": {
                        "type": "key_value_line",
                        "key": "product_category",
                    },
                    "weight": 1.0,
                }
            ],
        ),
    )

    assert weights == [0.0, 1.0]


def test_truncated_selected_field_reports_match_but_no_weighted_token():
    text = "intro\nproduct_category: Shoes"
    diagnostics = {}
    weights = build_token_loss_weights(
        messages=[{"role": "assistant", "content": text}],
        rendered_text=text,
        offset_mapping=[(0, 5)],
        assistant_mask=[1],
        config=LossWeightingConfig.model_validate(
            {
                "default_weight": 0.0,
                "zero_weight_action": "allow",
                "rules": [
                    {
                        "name": "category",
                        "selector": {
                            "type": "key_value_line",
                            "key": "product_category",
                        },
                        "weight": 1.0,
                    }
                ],
            }
        ),
        diagnostics=diagnostics,
    )

    assert weights == [0.0]
    assert diagnostics == {
        "weighted_token_count": 0,
        "effective_weight_sum": 0.0,
        "rule_matches": {"category": True},
    }


def test_all_one_weights_match_normal_causal_ce():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))
    labels[0, :2] = -100
    weights = labels.ne(-100).float()

    actual = weighted_causal_lm_loss(logits, labels, weights)
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
    expected = F.cross_entropy(
        logits.view(-1, 11), shifted.reshape(-1), ignore_index=-100
    )

    torch.testing.assert_close(actual, expected)


def test_zero_weight_matches_true_shifted_label_masking():
    torch.manual_seed(1)
    logits = torch.randn(1, 5, 7, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    weights = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]])

    actual = weighted_causal_lm_loss(logits, labels, weights)
    masked_labels = labels.clone()
    masked_labels[weights == 0] = -100
    shifted = F.pad(masked_labels, (0, 1), value=-100)[..., 1:]
    expected = F.cross_entropy(
        logits.view(-1, 7), shifted.reshape(-1), ignore_index=-100
    )

    torch.testing.assert_close(actual, expected)


def test_weighted_denominator_preserves_optimization_scale():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    labels = torch.tensor([[0, 1, 0]])
    weights = torch.tensor([[0.0, 2.0, 1.0]])

    local = weighted_causal_lm_loss(logits, labels, weights)
    doubled_world = weighted_causal_lm_loss(
        logits,
        labels,
        weights,
        num_items_in_batch=2 * weights[..., 1:].sum(),
    )

    torch.testing.assert_close(doubled_world * 2, local)


class _WeightedMixinHarness(TokenWeightedLossMixin):
    def __init__(self):
        self._leap_token_weighting = None
        self._leap_weight_diagnostics = {}
        self.args = SimpleNamespace(average_tokens_across_devices=True, n_gpu=1)
        self.accelerator = SimpleNamespace(
            gather=lambda value: torch.stack([value, value * 2]),
            num_processes=2,
        )


def test_weighted_mixin_uses_accumulated_global_denominator_and_ddp_scale():
    harness = _WeightedMixinHarness()
    samples = [
        {
            "labels": torch.tensor([[1, 2, 3]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 2.0]]),
        },
        {
            "labels": torch.tensor([[4, -100, 5]]),
            "loss_weights": torch.tensor([[0.0, 9.0, 3.0]]),
        },
    ]

    denominator = harness._get_num_items_in_batch(samples, torch.device("cpu"))
    assert denominator.item() == 18.0

    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    labels = torch.tensor([[0, 1, 0]])
    weights = torch.tensor([[0.0, 2.0, 1.0]])

    def model(**_kwargs):
        return {"logits": logits}

    scaled = harness.compute_loss(
        model,
        {"input_ids": labels, "labels": labels, "loss_weights": weights},
        num_items_in_batch=denominator,
    )
    unscaled = weighted_causal_lm_loss(
        logits, labels, weights, num_items_in_batch=denominator
    )

    torch.testing.assert_close(scaled, unscaled * 2)
    assert harness._leap_weight_diagnostics["train"] == {
        "rows": 1.0,
        "tokens": 2.0,
        "weight": 3.0,
        "unweighted_numerator": 0.0,
        "unweighted_denominator": 0.0,
        "rule_rows": 0.0,
        "rule_matches": {},
    }


def test_callable_selector_fingerprint_tracks_source(tmp_path):
    selector_file = tmp_path / "selector.py"
    selector_file.write_text("def spans(text): return [(0, len(text))]\n")
    config = {
        "default_weight": 0.2,
        "rules": [
            {
                "name": "custom",
                "selector": {
                    "type": "callable",
                    "callable": f"{selector_file}:spans",
                },
                "weight": 1.0,
            }
        ],
    }

    first = loss_weighting_fingerprint_data(config)
    selector_file.write_text("def spans(text): return []\n")
    second = loss_weighting_fingerprint_data(config)

    first_hash = first["rules"][0]["selector"]["callable_source_sha256"]
    second_hash = second["rules"][0]["selector"]["callable_source_sha256"]
    assert first_hash != second_hash
    assert "callable_source_sha256" not in config["rules"][0]["selector"]
