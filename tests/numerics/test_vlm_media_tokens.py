from types import SimpleNamespace

from liquid_finetune.data_loading.media_tokens import resolve_media_tokens
from liquid_finetune.training.vlm_grpo import (
    _collapse_media_token_runs,
    _configure_media_generation_kwargs,
    _patch_lfm2_vllm_weight_sync,
)


class _Tokenizer:
    unk_token_id = 0

    def get_vocab(self):
        return {"<|image|>": 10, "<|video|>": 11, "existing": 12}

    def convert_ids_to_tokens(self, token_id):
        return {10: "<|image|>", 11: "<|video|>"}.get(token_id)


def _processor():
    return SimpleNamespace(tokenizer=_Tokenizer(), image_token_id=10, video_token_id=11)


def test_resolves_lfm3_image_and_video_tokens():
    tokens = resolve_media_tokens(_processor())
    assert tokens["image"] == {"ids": {10}, "tokens": {"<|image|>"}}
    assert tokens["video"] == {"ids": {11}, "tokens": {"<|video|>"}}


def test_hf_generation_suppresses_media_token_ids():
    args = SimpleNamespace(use_vllm=False, generation_kwargs={"suppress_tokens": [12]})
    _configure_media_generation_kwargs(args, _processor())
    assert args.generation_kwargs == {"suppress_tokens": [10, 11, 12]}


def test_vllm_generation_rejects_media_token_strings():
    args = SimpleNamespace(use_vllm=True, generation_kwargs={"bad_words": ["existing"]})
    _configure_media_generation_kwargs(args, _processor())
    assert args.generation_kwargs == {
        "bad_words": ["<|image|>", "<|video|>", "existing"]
    }


def test_vllm_prompt_collapses_expanded_media_runs():
    assert _collapse_media_token_runs(
        [[1, 10, 10, 10, 2, 11, 11, 3, 10, 10, 4], [5, 6]], {10, 11}
    ) == [[1, 10, 2, 11, 3, 10, 4], [5, 6]]


def test_lfm2_vllm_sync_restores_checkpoint_vision_prefix():
    pushed = []
    generation = SimpleNamespace(
        _push_param_to_vllm=lambda name, param: pushed.append((name, param))
    )
    _patch_lfm2_vllm_weight_sync(generation, "lfm2_vl")

    generation._push_param_to_vllm(
        "model.vision_tower.embeddings.patch_embedding.weight", "vision"
    )
    generation._push_param_to_vllm("model.language_model.embed_tokens.weight", "text")

    assert pushed == [
        (
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision",
        ),
        ("model.language_model.embed_tokens.weight", "text"),
    ]
