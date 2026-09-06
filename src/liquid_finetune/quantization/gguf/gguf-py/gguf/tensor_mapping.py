from __future__ import annotations

from typing import Sequence

from .constants import MODEL_ARCH, MODEL_TENSOR, MODEL_TENSORS, TENSOR_NAMES


class TensorNameMap:
    mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Token embeddings
        MODEL_TENSOR.TOKEN_EMBD: (
            "gpt_neox.embed_in",  # gptneox
            "model.embed_tokens",  # llama-hf nemotron olmoe olmo2 rwkv6qwen2 glm4-0414 plamo2 granite-hybrid
        ),
        # Token type embeddings
        MODEL_TENSOR.TOKEN_TYPES: (
            "embeddings.token_type_embeddings",  # bert nomic-bert
        ),
        # Normalization of token embeddings
        MODEL_TENSOR.TOKEN_EMBD_NORM: (
            "word_embeddings_layernorm",  # bloom
            "model.embedding_norm",  # lfm2
        ),
        # Position embeddings
        MODEL_TENSOR.POS_EMBD: (
            "transformer.wpe",  # gpt2
        ),
        # Output
        MODEL_TENSOR.OUTPUT: (
            "embed_out",  # gptneox
            "lm_head",  # gpt2 mpt falcon llama-hf baichuan qwen mamba dbrx jais nemotron exaone olmoe olmo2 phimoe plamo2
        ),
        MODEL_TENSOR.DENSE_2_OUT: (
            "dense_2_out",  # embeddinggemma
        ),
        MODEL_TENSOR.DENSE_3_OUT: (
            "dense_3_out",  # embeddinggemma
        ),
        # Output norm
        MODEL_TENSOR.OUTPUT_NORM: (
            "gpt_neox.final_layer_norm",  # gptneox
            "model.norm",  # llama-hf baichuan internlm2 olmoe olmo2 phimoe plamo2
        ),
        # Rope frequencies
        MODEL_TENSOR.ROPE_FREQS: (
            "rope.freqs",  # llama-pth
        ),
        MODEL_TENSOR.ROPE_FACTORS_LONG: (),
        MODEL_TENSOR.ROPE_FACTORS_SHORT: (),
    }

    block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Attention norm
        MODEL_TENSOR.ATTN_NORM: (
            "gpt_neox.layers.{bid}.input_layernorm",  # gptneox
            "model.layers.{bid}.input_layernorm",  # llama-hf nemotron olmoe phimoe granite-hybrid
            "model.layers.{bid}.operator_norm",  # lfm2
        ),
        # Attention norm 2
        MODEL_TENSOR.ATTN_NORM_2: (
            "transformer.h.{bid}.ln_attn",  # falcon40b
        ),
        # Attention query-key-value
        MODEL_TENSOR.ATTN_QKV: (
            "gpt_neox.layers.{bid}.attention.query_key_value",  # gptneox
        ),
        # Attention query
        MODEL_TENSOR.ATTN_Q: (
            "model.layers.{bid}.self_attn.q_proj",  # llama-hf nemotron olmoe olmo2 phimoe
        ),
        # Attention key
        MODEL_TENSOR.ATTN_K: (
            "model.layers.{bid}.self_attn.k_proj",  # llama-hf nemotron olmoe olmo2 phimoe
        ),
        # Attention value
        MODEL_TENSOR.ATTN_V: (
            "model.layers.{bid}.self_attn.v_proj",  # llama-hf nemotron olmoe olmo2 phimoe
        ),
        # Attention output
        MODEL_TENSOR.ATTN_OUT: (
            "gpt_neox.layers.{bid}.attention.dense",  # gptneox
            "model.layers.{bid}.self_attn.o_proj",  # llama-hf nemotron olmoe olmo2 phimoe
            "model.layers.{bid}.self_attn.out_proj",  # lfm2
        ),
        # Attention output norm
        MODEL_TENSOR.ATTN_OUT_NORM: (
            "encoder.layer.{bid}.attention.output.LayerNorm",  # bert
        ),
        MODEL_TENSOR.ATTN_POST_NORM: (
            "model.layers.{bid}.post_attention_layernorm",  # gemma2 olmo2
        ),
        # Rotary embeddings
        MODEL_TENSOR.ATTN_ROT_EMBD: (
            "model.layers.{bid}.self_attn.rotary_emb.inv_freq",  # llama-hf
        ),
        # Feed-forward norm
        MODEL_TENSOR.FFN_NORM: (
            "gpt_neox.layers.{bid}.post_attention_layernorm",  # gptneox
            "model.layers.{bid}.post_attention_layernorm",  # llama-hf nemotron olmoe phimoe
        ),
        MODEL_TENSOR.FFN_GATE_INP: (
            "layers.{bid}.feed_forward.gate",  # mixtral
            "model.layers.{bid}.block_sparse_moe.gate",  # mixtral phimoe
            "model.layers.{bid}.feed_forward.gate",  # lfm2moe
        ),
        MODEL_TENSOR.FFN_EXP_PROBS_B: (
            "model.layers.{bid}.mlp.gate.e_score_correction",  # deepseek-v3 dots1
            "model.layers.{bid}.feed_forward.expert_bias",  # lfm2moe
        ),
        # Feed-forward up
        MODEL_TENSOR.FFN_UP: (
            "gpt_neox.layers.{bid}.mlp.dense_h_to_4h",  # gptneox
            "model.layers.{bid}.mlp.up_proj",  # llama-hf refact nemotron olmo2
        ),
        MODEL_TENSOR.FFN_UP_EXP: (
            "layers.{bid}.feed_forward.experts.w3",  # mixtral (merged)
            "model.layers.{bid}.feed_forward.experts.up_proj",  # llama4
        ),
        # Feed-forward gate
        MODEL_TENSOR.FFN_GATE: (
            "model.layers.{bid}.mlp.gate_proj",  # llama-hf refact olmo2
            "model.layers.{bid}.feed_forward.gate_proj",  # llama4 jamba granite-hybrid
        ),
        MODEL_TENSOR.FFN_GATE_EXP: (
            "layers.{bid}.feed_forward.experts.w1",  # mixtral (merged)
            "model.layers.{bid}.feed_forward.experts.gate_proj",  # llama4
        ),
        # Feed-forward down
        MODEL_TENSOR.FFN_DOWN: (
            "gpt_neox.layers.{bid}.mlp.dense_4h_to_h",  # gptneox
            "model.layers.{bid}.mlp.down_proj",  # llama-hf nemotron olmo2
        ),
        MODEL_TENSOR.FFN_DOWN_EXP: (
            "layers.{bid}.feed_forward.experts.w2",  # mixtral (merged)
            "model.layers.{bid}.feed_forward.experts.down_proj",  # llama4
        ),
        MODEL_TENSOR.ATTN_Q_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.q_layernorm",
            "model.layers.{bid}.self_attn.q_norm",  # cohere olmoe chameleon olmo2
        ),
        MODEL_TENSOR.ATTN_K_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.k_layernorm",
            "model.layers.{bid}.self_attn.k_norm",  # cohere olmoe chameleon olmo2
        ),
        MODEL_TENSOR.ROPE_FREQS: (
            "language_model.encoder.layers.{bid}.self_attention.rotary_emb.inv_freq",  # persimmon
        ),
        MODEL_TENSOR.SHORTCONV_CONV: ("model.layers.{bid}.conv.conv",),
        MODEL_TENSOR.SHORTCONV_INPROJ: ("model.layers.{bid}.conv.in_proj",),
        MODEL_TENSOR.SHORTCONV_OUTPROJ: ("model.layers.{bid}.conv.out_proj",),
        # audio (mtmd)
        MODEL_TENSOR.A_ENC_EMBD_POS: (
            "audio_tower.embed_positions",  # ultravox
            "audio_embedding.embedding",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_EMBD_NORM: (
            "audio_embedding.embedding_norm",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_EMBD_TO_LOGITS: (
            "audio_embedding.to_logits",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_CONV1D: (
            "audio_tower.conv{bid}",  # ultravox
            "conformer.pre_encode.conv.{bid}",  # lfm2
        ),
        MODEL_TENSOR.A_PRE_NORM: (),
        MODEL_TENSOR.A_POST_NORM: (
            "audio_tower.layer_norm",  # ultravox
        ),
        MODEL_TENSOR.A_ENC_ATTN_Q: (
            "audio_tower.layers.{bid}.self_attn.q_proj",  # ultravox
            "conformer.layers.{bid}.self_attn.linear_q",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_ATTN_K: (
            "audio_tower.layers.{bid}.self_attn.k_proj",  # ultravox
            "conformer.layers.{bid}.self_attn.linear_k",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_ATTN_V: (
            "audio_tower.layers.{bid}.self_attn.v_proj",  # ultravox
            "conformer.layers.{bid}.self_attn.linear_v",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_INPUT_NORM: (
            "audio_tower.layers.{bid}.self_attn_layer_norm",  # ultravox
            "conformer.layers.{bid}.norm_self_att",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_OUTPUT: (
            "audio_tower.layers.{bid}.self_attn.out_proj",  # ultravox
            "conformer.layers.{bid}.self_attn.linear_out",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_OUTPUT_NORM: (
            "audio_tower.layers.{bid}.final_layer_norm",  # ultravox
            "conformer.layers.{bid}.norm_out",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_FFN_NORM: (
            "conformer.layers.{bid}.norm_feed_forward1",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_FFN_UP: (
            "audio_tower.layers.{bid}.fc1",  # ultravox
            "conformer.layers.{bid}.feed_forward1.linear1",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_FFN_GATE: (),
        MODEL_TENSOR.A_ENC_FFN_DOWN: (
            "audio_tower.layers.{bid}.fc2",  # ultravox
            "conformer.layers.{bid}.feed_forward1.linear2",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_FFN_UP_1: (
            "conformer.layers.{bid}.feed_forward2.linear1",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_FFN_DOWN_1: (
            "conformer.layers.{bid}.feed_forward2.linear2",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_FFN_NORM_1: (
            "conformer.layers.{bid}.norm_feed_forward2",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_LINEAR_POS: (
            "conformer.layers.{bid}.self_attn.linear_pos",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_POS_BIAS_U: (
            "conformer.layers.{bid}.self_attn.pos_bias_u",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_POS_BIAS_V: (
            "conformer.layers.{bid}.self_attn.pos_bias_v",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_OUT: (
            "conformer.pre_encode.out",  # lfm2
        ),
        # note: some tensors below has "audio." pseudo-prefix, to prevent conflicts with vision tensors
        # this prefix is added in the conversion code in modify_tensors()
        MODEL_TENSOR.A_MMPROJ: (
            "audio.multi_modal_projector.linear_{bid}",  # ultravox
            "audio_adapter.model.{bid}",  # lfm2
        ),
        MODEL_TENSOR.A_MMPROJ_FC: (
            "audio.multi_modal_projector.linear",  # qwen2audio
        ),
        MODEL_TENSOR.A_MM_NORM_PRE: (
            "audio.multi_modal_projector.ln_pre",  # ultravox
        ),
        MODEL_TENSOR.A_MM_NORM_MID: (
            "audio.multi_modal_projector.ln_mid",  # ultravox
        ),
        MODEL_TENSOR.A_ENC_CONV_DW: (
            "conformer.layers.{bid}.conv.depthwise_conv",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_CONV_NORM: (
            "conformer.layers.{bid}.conv.batch_norm",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_CONV_PW1: (
            "conformer.layers.{bid}.conv.pointwise_conv1",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_CONV_PW2: (
            "conformer.layers.{bid}.conv.pointwise_conv2",  # lfm2
        ),
        MODEL_TENSOR.A_ENC_NORM_CONV: (
            "conformer.layers.{bid}.norm_conv",  # lfm2
        ),
    }

    # architecture-specific block mappings
    arch_block_mappings_cfg: dict[MODEL_ARCH, dict[MODEL_TENSOR, tuple[str, ...]]] = {}

    mapping: dict[str, tuple[MODEL_TENSOR, str]]

    def __init__(self, arch: MODEL_ARCH, n_blocks: int):
        self.mapping = {}
        for tensor, keys in self.mappings_cfg.items():
            if tensor not in MODEL_TENSORS[arch]:
                continue
            tensor_name = TENSOR_NAMES[tensor]
            self.mapping[tensor_name] = (tensor, tensor_name)
            for key in keys:
                self.mapping[key] = (tensor, tensor_name)
        if arch in self.arch_block_mappings_cfg:
            self.block_mappings_cfg.update(self.arch_block_mappings_cfg[arch])
        for bid in range(n_blocks):
            for tensor, keys in self.block_mappings_cfg.items():
                if tensor not in MODEL_TENSORS[arch]:
                    continue

                tensor_name = TENSOR_NAMES[tensor].format(bid=bid)
                self.mapping[tensor_name] = (tensor, tensor_name)
                for key in keys:
                    key = key.format(bid=bid)
                    self.mapping[key] = (tensor, tensor_name)

    def get_type_and_name(
        self, key: str, try_suffixes: Sequence[str] = ()
    ) -> tuple[MODEL_TENSOR, str] | None:
        result = self.mapping.get(key)
        if result is not None:
            return result
        for suffix in try_suffixes:
            if key.endswith(suffix):
                result = self.mapping.get(key[: -len(suffix)])
                if result is not None:
                    return result[0], result[1] + suffix
        return None

    def get_name(self, key: str, try_suffixes: Sequence[str] = ()) -> str | None:
        result = self.get_type_and_name(key, try_suffixes=try_suffixes)
        if result is None:
            return None
        return result[1]

    def get_type(
        self, key: str, try_suffixes: Sequence[str] = ()
    ) -> MODEL_TENSOR | None:
        result = self.get_type_and_name(key, try_suffixes=try_suffixes)
        if result is None:
            return None
        return result[0]

    def __getitem__(self, key: str) -> str:
        try:
            return self.mapping[key][1]
        except KeyError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return repr(self.mapping)


def get_tensor_name_map(arch: MODEL_ARCH, n_blocks: int) -> TensorNameMap:
    return TensorNameMap(arch, n_blocks)
