from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from liquid_finetune.rl.environments.adapter import build_openenv_rollout_func


class _Processor:
    @staticmethod
    def batch_decode(completion_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return [f"completion-{ids[0]}" for ids in completion_ids]


def test_openenv_rollout_uses_trainer_generation_and_returns_rewards():
    env = Mock()
    env.step.side_effect = [
        SimpleNamespace(reward=1.5),
        SimpleNamespace(reward=None),
    ]
    trainer = SimpleNamespace(
        _tokenize_prompts=Mock(
            return_value=([[10], [20]], ["image-a", "image-b"], {"pixels": [1, 2]})
        ),
        _generate_single_turn=Mock(
            return_value=([[30, 31], [40]], [[-0.1, -0.2], [-0.3]])
        ),
        processing_class=_Processor(),
    )

    rollout = build_openenv_rollout_func(
        env, reset_kwargs={"seed": 7}, action_key="answer"
    )
    output = rollout(["first", "second"], trainer)

    trainer._tokenize_prompts.assert_called_once_with(["first", "second"])
    trainer._generate_single_turn.assert_called_once_with(
        [[10], [20]], ["image-a", "image-b"], {"pixels": [1, 2]}
    )
    assert output == {
        "prompt_ids": [[10], [20]],
        "completion_ids": [[30, 31], [40]],
        "logprobs": [[-0.1, -0.2], [-0.3]],
        "env_reward": [1.5, 0.0],
    }
    assert env.reset.call_args_list == [call(seed=7), call(seed=7)]
    assert env.step.call_args_list == [
        call({"answer": "completion-30"}),
        call({"answer": "completion-40"}),
    ]


def test_openenv_rollout_rejects_multi_turn():
    with pytest.raises(NotImplementedError, match="max_turns > 1"):
        build_openenv_rollout_func(Mock(), max_turns=2)
