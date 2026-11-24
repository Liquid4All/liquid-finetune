# Remote-code module shipped into LFM3 checkpoints by `liquid-prepare-mtp`; it is copied next to
# modeling_lfm3.py and imported from there (relative imports), never from the liquid_finetune package.
#
# LFM3 + one Multi-Token-Prediction (MTP) module using Qwen3.5-style component names:
# mtp.pre_fc_norm_embedding, mtp.pre_fc_norm_hidden, mtp.fc (2H -> H), mtp.layers.0 (one LFM3 decoder layer of the type of
# `mtp_init_from_layer`, full attention by default), mtp.norm; logits through the shared tied lm_head. An inference engine
# still needs an LFM3-specific implementation; the shared names do not make an existing Qwen MTP path architecture-compatible.
#
# Draft rule (DeepSeek-V3 / Qwen3-Next style): at position t the module takes the main model's final hidden state h_t
# (after the final `embedding_norm`) and the embedding of token x_{t+1}, and predicts x_{t+2}:
#   z_t = fc([pre_fc_norm_embedding(E[x_{t+1}]) ; pre_fc_norm_hidden(h_t)]);  z = layer(z, positions of x_{t+1});
#   logits_{t+2} = lm_head(norm(z_t)).
#
# Config keys: mtp_num_hidden_layers (1), mtp_use_dedicated_embeddings (False), mtp_init_from_layer, mtp_loss_weight
# (lambda), mtp_train_only (True = frozen backbone, MTP loss only; False = main loss + lambda * MTP loss),
# mtp_joint_after_loss (joint mode only, default 3.0): while the running MTP loss (EMA over training forwards, per rank) is above
# it, the MTP branch trains on DETACHED inputs (h_t, E[x_{t+1}], lm_head weight), so a freshly initialised head cannot push
# its large early gradients into the backbone / tied embeddings; once the head is competent, full joint gradients. Resume-safe
# (a resumed, trained head is below the threshold at its first step).
# mtp_soft (default False): Soft-MTP (Teoh 2026, "Multi-Token Prediction Done Right", github.com/JaydenTeoh/Soft-MTP). The main
# model's hidden state at the position that predicts the SAME token as the head (h_{t+1} for x_{t+2}) gets
# s * (z_t.detach() - z_t) added: forward value unchanged, but in backward the NTP cross-entropy gradient there, negated and
# scaled to the MTP term's weight, flows into the head, so the head's gradient is W^T (q - p) = the gradient of
# KL(p_ntp.detach() || q_head): the head distills the model's own next-token distribution; the one-hot label terms cancel.
# Frozen backbone (mtp_train_only): the coupled NTP CE is added as (c - c.detach()), so the reported loss stays the hard MTP CE.
# mtp_num_modules (default 1) = 2 adds a second, sequential module `mtp2` (same layout as `mtp`; DeepSeek-V3 style) that drafts
# the token after next: at position t it takes module 1's post-norm output z1_t (the state that predicts x_{t+2}) and the
# embedding of x_{t+2}, runs at RoPE position t+2 with causal self-attention over its own states, and predicts x_{t+3}:
#   u_t = mtp2.fc([mtp2.pre_fc_norm_embedding(E[x_{t+2}]) ; mtp2.pre_fc_norm_hidden(z1_t)]);  logits_{t+3} = lm_head(mtp2.norm(layer(u)_t)).
# mtp_num_modules = 3 adds a third sequential module `mtp3` (same layout) that drafts x_{t+4}: at position t it takes module 2's
# post-norm output z2_t (the state that predicts x_{t+3}) and the embedding of x_{t+3}, runs at RoPE position t+3 with causal
# self-attention over its own states, and predicts x_{t+4}.
# mtp_train_modules (frozen backbone only): which modules train, e.g. [2] keeps module 1 frozen; default = every module present.
# With 2 or 3 modules: mtp_train_only=True only; the hard objective (CE of module d on x_{t+d+1}) or mtp_soft (Soft-MTP for every
# trained module d: h_{t+d}, the main state that predicts the same token, gets (z_d.detach() - z_d) added inside a coupled
# value-neutral CE, so module d's gradient is W^T (q_d - p_main) = the gradient of KL(p_main.detach() || q_d); the reported loss
# stays the hard CE).
# Inference / generation (no labels) runs the unchanged LFM3 forward. Training computes both cross-entropies over
# supervised positions only, in checkpointed chunks, so no full-vocabulary logits are materialised.
import contextlib

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_lfm3 import (
    Lfm3DecoderLayer,
    Lfm3ForCausalLM,
    Lfm3RMSNorm,
    create_masks_by_layer_type,
)


def _ce_sum(hidden, targets, weight):
    return F.cross_entropy(F.linear(hidden, weight).float(), targets, reduction="sum")


def chunked_ce(hidden, targets, weight, chunk=4096):
    """Sum of token cross-entropies over positions with targets != -100, and their count."""
    keep = targets.ne(-100)
    hs, ys = hidden[keep], targets[keep]
    # Keep an autograd connection for batches with no MTP-supervised tokens
    # (for example, an assistant-only sample truncated before its response).
    total = hidden.float().sum() * 0.0
    for s in range(0, hs.shape[0], chunk):
        h, y = hs[s : s + chunk], ys[s : s + chunk]
        total = total + (
            checkpoint(_ce_sum, h, y, weight, use_reentrant=False)
            if torch.is_grad_enabled()
            else _ce_sum(h, y, weight)
        )
    return total, keep.sum()


def _mtp_mask_and_positions(attention_mask, position_ids, offset):
    """Align masks/positions and isolate states that cross document boundaries."""
    positions = position_ids[:, offset:]
    continuous = positions.eq(position_ids[:, :-offset] + offset)
    if attention_mask is None or (
        torch.is_tensor(attention_mask)
        and attention_mask.ndim == 2
        and bool(attention_mask.all())
    ):
        if not bool(continuous.all()):
            # The generic LFM3 mask recognizes a restart from position IDs. A
            # cross-document draft state must be its own empty document, and
            # the next valid state must restart again rather than attend it.
            previous = torch.cat([positions[:, :1] - 1, positions[:, :-1]], dim=1)
            positions = torch.where(continuous, positions, previous)
        return None, positions
    if not torch.is_tensor(attention_mask):
        raise TypeError("MTP training expects attention_mask to be a tensor")
    if attention_mask.ndim == 2:
        # Draft state t combines the backbone state at t with the embedding at
        # t + offset, so it is valid only when that complete input window is.
        mask = attention_mask.unfold(1, offset + 1, 1).bool().all(-1)
        return mask & continuous, positions
    if attention_mask.ndim == 4:
        return attention_mask[..., offset:, offset:], positions
    raise ValueError(
        f"MTP training expects a 2D or 4D attention_mask, got {attention_mask.ndim}D"
    )


def _shifted_targets(labels, position_ids, attention_mask, shift):
    """Targets whose complete MTP context stays inside one non-padding sequence."""
    targets = labels[:, shift:].clone()
    valid = position_ids[:, shift:].eq(position_ids[:, :-shift] + shift)
    if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
        valid = valid & attention_mask.unfold(1, shift + 1, 1).bool().all(-1)
    return targets.masked_fill(~valid, -100)


def _loss_denominator(num_items, local_items, device):
    if torch.is_tensor(num_items):
        return num_items.to(device).float().clamp_min(1)
    if num_items is not None:
        return max(float(num_items), 1.0)
    return local_items.clamp_min(1)


class Lfm3MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        src = int(
            getattr(
                config,
                "mtp_init_from_layer",
                max(
                    i for i, t in enumerate(config.layer_types) if t == "full_attention"
                ),
            )
        )
        self.layer_type = config.layer_types[src]
        if self.layer_type == "conv":
            raise ValueError("the MTP layer must be an attention layer type")
        self.pre_fc_norm_embedding = Lfm3RMSNorm(H, eps=config.norm_eps)
        self.pre_fc_norm_hidden = Lfm3RMSNorm(H, eps=config.norm_eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = nn.ModuleList([Lfm3DecoderLayer(config, src)])
        self.norm = Lfm3RMSNorm(H, eps=config.norm_eps)


class Lfm3ForCausalLMWithMTP(Lfm3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        on = int(getattr(config, "mtp_num_hidden_layers", 0) or 0) > 0
        self.mtp = Lfm3MTP(config) if on else None
        self.mtp2 = (
            Lfm3MTP(config)
            if on and int(getattr(config, "mtp_num_modules", 1) or 1) >= 2
            else None
        )
        if on and int(getattr(config, "mtp_num_modules", 1) or 1) >= 3:
            self.mtp3 = Lfm3MTP(config)
        else:
            self.mtp3 = None
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        out = super().from_pretrained(*args, **kwargs)
        model = (
            out[0] if isinstance(out, tuple) else out
        )  # output_loading_info=True returns (model, info)
        if model.mtp is not None and bool(
            getattr(model.config, "mtp_train_only", False)
        ):
            # frozen backbone: only the MTP module(s) in mtp_train_modules train (default: every module present);
            # lm_head / embeddings stay the main model's
            mods = (
                getattr(model.config, "mtp_train_modules", None)
                or model._present_modules()
            )
            prefixes = tuple({1: "mtp.", 2: "mtp2.", 3: "mtp3."}[int(m)] for m in mods)
            for name, p in model.named_parameters():
                p.requires_grad_(name.startswith(prefixes))
        return out

    def _present_modules(self):
        return [
            d
            for d, m in ((1, self.mtp), (2, self.mtp2), (3, self.mtp3))
            if m is not None
        ]

    def mtp_hidden(
        self,
        last_hidden,
        input_ids,
        position_ids,
        attention_mask=None,
        detach_embed=False,
    ):
        """MTP features for positions t = 0..T-2 (predicting x_{t+2}); shape (B, T-1, H)."""
        e = self.model.embed_tokens(input_ids[:, 1:])
        if detach_embed:
            e = e.detach()
        z = self.mtp.fc(
            torch.cat(
                [
                    self.mtp.pre_fc_norm_embedding(e),
                    self.mtp.pre_fc_norm_hidden(last_hidden[:, :-1]),
                ],
                dim=-1,
            )
        )
        mask, pos = _mtp_mask_and_positions(attention_mask, position_ids, 1)
        masks = create_masks_by_layer_type(self.config, z, mask, None, pos)
        pe = self.model.rotary_emb(z, position_ids=pos, layer_type=self.mtp.layer_type)
        for layer in self.mtp.layers:
            z = layer(
                z,
                attention_mask=masks[self.mtp.layer_type],
                position_embeddings=pe,
                position_ids=pos,
                past_key_values=None,
            )
        return self.mtp.norm(z)

    def mtp2_hidden(
        self,
        z1,
        input_ids,
        position_ids,
        attention_mask=None,
        detach_embed=False,
    ):
        """Second-module features for positions t = 0..T-3 (predicting x_{t+3}); shape (B, T-2, H). z1 = mtp_hidden(...)
        (B, T-1, H), index t = module 1's state that predicts x_{t+2}; input at t = [E[x_{t+2}] ; z1_t] at RoPE position t+2."""
        m = self.mtp2
        e = self.model.embed_tokens(input_ids[:, 2:])
        if detach_embed:
            e = e.detach()
        u = m.fc(
            torch.cat(
                [m.pre_fc_norm_embedding(e), m.pre_fc_norm_hidden(z1[:, :-1])], dim=-1
            )
        )
        mask, pos = _mtp_mask_and_positions(attention_mask, position_ids, 2)
        masks = create_masks_by_layer_type(self.config, u, mask, None, pos)
        pe = self.model.rotary_emb(u, position_ids=pos, layer_type=m.layer_type)
        for layer in m.layers:
            u = layer(
                u,
                attention_mask=masks[m.layer_type],
                position_embeddings=pe,
                position_ids=pos,
                past_key_values=None,
            )
        return m.norm(u)

    def mtp3_hidden(
        self,
        z2,
        input_ids,
        position_ids,
        attention_mask=None,
        detach_embed=False,
    ):
        """Third-module features for positions t = 0..T-4 (predicting x_{t+4}); shape (B, T-3, H). z2 = mtp2_hidden(...)
        (B, T-2, H), index t = module 2's state that predicts x_{t+3}; input at t = [E[x_{t+3}] ; z2_t] at RoPE position t+3."""
        m = self.mtp3
        e = self.model.embed_tokens(input_ids[:, 3:])
        if detach_embed:
            e = e.detach()
        u = m.fc(
            torch.cat(
                [m.pre_fc_norm_embedding(e), m.pre_fc_norm_hidden(z2[:, :-1])], dim=-1
            )
        )
        mask, pos = _mtp_mask_and_positions(attention_mask, position_ids, 3)
        masks = create_masks_by_layer_type(self.config, u, mask, None, pos)
        pe = self.model.rotary_emb(u, position_ids=pos, layer_type=m.layer_type)
        for layer in m.layers:
            u = layer(
                u,
                attention_mask=masks[m.layer_type],
                position_embeddings=pe,
                position_ids=pos,
                past_key_values=None,
            )
        return m.norm(u)

    def _multi_module_loss(
        self, h, input_ids, position_ids, attention_mask, labels, num_items
    ):
        """Frozen backbone, 2 or 3 sequential modules: for every trained module d, CE on x_{t+d+1} (hard), or Soft-MTP (mtp_soft):
        the same CE plus a value-neutral coupled CE that routes -W^T (p_main - onehot) into z_d, so module d's gradient is the
        gradient of KL(p_main.detach() || q_d) at the main state h_{t+d} that predicts the same token. Modules run only up to the
        deepest trained one; a module runs with grad only if it or an earlier module trains."""
        if not bool(getattr(self.config, "mtp_train_only", False)):
            raise NotImplementedError(
                "2 or 3 MTP modules support mtp_train_only=True (frozen backbone) only"
            )
        soft = bool(getattr(self.config, "mtp_soft", False)) and torch.is_grad_enabled()
        mods = sorted(
            int(m)
            for m in (
                getattr(self.config, "mtp_train_modules", None)
                or self._present_modules()
            )
        )
        assert all(m in self._present_modules() for m in mods), (
            f"mtp_train_modules {mods} vs modules present {self._present_modules()}"
        )
        W = self.lm_head.weight.detach()
        hd = h.detach()
        zs, prev = {}, None
        for dpt in range(1, max(mods) + 1):
            with (
                contextlib.nullcontext()
                if any(m <= dpt for m in mods)
                else torch.no_grad()
            ):
                if dpt == 1:
                    prev = self.mtp_hidden(
                        hd,
                        input_ids,
                        position_ids,
                        attention_mask,
                        detach_embed=True,
                    )
                elif dpt == 2:
                    prev = self.mtp2_hidden(
                        prev,
                        input_ids,
                        position_ids,
                        attention_mask,
                        detach_embed=True,
                    )
                else:
                    prev = self.mtp3_hidden(
                        prev,
                        input_ids,
                        position_ids,
                        attention_mask,
                        detach_embed=True,
                    )
            zs[dpt] = prev
        total, n = None, None
        for dpt in sorted(mods, reverse=True):
            z = zs[dpt][:, :-1]
            y = _shifted_targets(labels, position_ids, attention_mask, dpt + 1)
            s, k = chunked_ce(z, y, W)
            if soft:
                c, _ = chunked_ce(hd[:, dpt:-1] + (z.detach() - z), y, W)
                s = s + c - c.detach()
            total, n = (s, k) if total is None else (total + s, n + k)
        denom = _loss_denominator(num_items, n, h.device)
        return CausalLMOutputWithPast(
            loss=total / denom,
            logits=h.new_empty((0,)),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
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
                **kwargs,
            )
        if input_ids is None:
            raise ValueError(
                "MTP training needs input_ids (the draft input is the embedding of the next token)"
            )
        num_items = kwargs.pop("num_items_in_batch", None)
        train_only = bool(getattr(self.config, "mtp_train_only", False))
        lam = float(getattr(self.config, "mtp_loss_weight", 0.3))
        if position_ids is None:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )
        with torch.no_grad() if train_only else contextlib.nullcontext():
            h = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                **kwargs,
            ).last_hidden_state
        if self.mtp2 is not None:
            return self._multi_module_loss(
                h, input_ids, position_ids, attention_mask, labels, num_items
            )
        W = self.lm_head.weight
        thr = float(getattr(self.config, "mtp_joint_after_loss", 3.0))
        ema = getattr(self, "_mtp_loss_ema", None)
        warm = (not train_only) and self.training and (ema is None or ema > thr)
        if not train_only and warm != getattr(self, "_mtp_warm_state", None):
            print(
                f"[mtp] joint gradients {'OFF (head warm-up, detached inputs)' if warm else 'ON'} (running MTP loss {ema}, threshold {thr})",
                flush=True,
            )
            self._mtp_warm_state = warm
        detach = train_only or warm
        z = self.mtp_hidden(
            h.detach() if detach else h,
            input_ids,
            position_ids,
            attention_mask,
            detach_embed=detach,
        )
        mtp_targets = _shifted_targets(labels, position_ids, attention_mask, 2)
        mtp_sum, mtp_n = chunked_ce(z[:, :-1], mtp_targets, W.detach() if detach else W)
        if self.training and not train_only:
            cur = float(mtp_sum.detach()) / max(1, int(mtp_n))
            self._mtp_loss_ema = cur if ema is None else 0.9 * ema + 0.1 * cur
        # HF semantics: with num_items_in_batch (global supervised-token count) divide by it (the Trainer then rescales
        # for the data-parallel average); without it, a per-batch token mean.
        denom = _loss_denominator(num_items, mtp_n, h.device)
        soft = bool(getattr(self.config, "mtp_soft", False)) and torch.is_grad_enabled()
        if train_only:
            main_sum, main_n = None, labels[:, 1:].ne(-100).sum()
            if soft:  # Soft-MTP, frozen backbone: same weight as the MTP term (s = 1), value-neutral
                c_sum, _ = chunked_ce(
                    h[:, 1:-1].detach() + (z[:, :-1].detach() - z[:, :-1]),
                    mtp_targets,
                    W.detach(),
                )
                mtp_sum = mtp_sum + c_sum - c_sum.detach()
        else:
            main_n = labels[:, 1:].ne(-100).sum()
            hn = h
            if soft:  # Soft-MTP, joint: s = MTP-term weight / main-term weight (= lambda when both divide by num_items)
                s = (
                    lam
                    if num_items is not None
                    else lam * float(main_n) / max(1, int(mtp_n))
                )
                hn = torch.cat(
                    [
                        h[:, :1],
                        h[:, 1:-1] + s * (z[:, :-1].detach() - z[:, :-1]),
                        h[:, -1:],
                    ],
                    dim=1,
                )
            main_sum, _ = chunked_ce(hn[:, :-1], labels[:, 1:], W)
        mtp_loss = mtp_sum / denom
        loss = (
            mtp_loss
            if train_only
            else main_sum / _loss_denominator(num_items, main_n, h.device)
            + lam * mtp_loss
        )
        return CausalLMOutputWithPast(loss=loss, logits=h.new_empty((0,)))


__all__ = ["Lfm3ForCausalLMWithMTP", "Lfm3MTP"]
