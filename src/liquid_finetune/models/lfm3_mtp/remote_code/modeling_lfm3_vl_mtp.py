# Remote-code module shipped into LFM3-VL checkpoints by `liquid-prepare-mtp`; it is copied next to
# modeling_lfm3_vl.py / modeling_lfm3.py and imported from there (relative imports), never from the liquid_finetune package.
#
# LFM3-VL + MTP draft module(s). The MTP math is `Lfm3MTPMixin` (modeling_lfm3_mtp.py), shared with the text model: the
# modules read the language model's final hidden states and its INPUT embeddings, i.e. the token embeddings with the
# image / video features already scattered into their placeholder positions, and score through the shared lm_head.
# Weights are top-level `mtp.*` / `mtp2.*` / `mtp3.*`, exactly as for the text model. Inference / generation (no labels)
# runs the unchanged LFM3-VL forward.
from .modeling_lfm3_mtp import Lfm3MTPMixin
from .modeling_lfm3_vl import Lfm3VlForConditionalGeneration


class Lfm3VlForConditionalGenerationWithMTP(
    Lfm3MTPMixin, Lfm3VlForConditionalGeneration
):
    def __init__(self, config):
        super().__init__(config)
        self._init_mtp_modules()
        self.post_init()

    def _mtp_text_config(self):
        return self.config.text_config

    def _mtp_text_model(self):
        return self.model.language_model

    def _mtp_backbone(self, input_ids, attention_mask, position_ids, kwargs):
        captured = {}

        def grab_inputs_embeds(_module, _args, module_kwargs):
            captured["embeds"] = module_kwargs["inputs_embeds"]

        handle = self.model.language_model.register_forward_pre_hook(
            grab_inputs_embeds, with_kwargs=True
        )
        try:
            h = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **kwargs,
            ).last_hidden_state
        finally:
            handle.remove()
        return h, captured["embeds"]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        mm_encoder_outputs=None,
        **kwargs,
    ):
        # The vision inputs are named in the signature: `generate` only forwards model kwargs that it finds there.
        vision = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "mm_encoder_outputs": mm_encoder_outputs,
        }
        if labels is None or self.mtp is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **vision,
                **kwargs,
            )
        vision = {k: v for k, v in vision.items() if v is not None}
        return self._mtp_forward(
            input_ids, attention_mask, position_ids, labels, {**vision, **kwargs}
        )


__all__ = ["Lfm3VlForConditionalGenerationWithMTP"]
