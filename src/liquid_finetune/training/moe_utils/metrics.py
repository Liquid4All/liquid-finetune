import logging

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MoEMetricsCallback(TrainerCallback):
    """Collects MoE health metrics from patched MoE blocks and logs to wandb/trainer.

    Reads module attributes set by the MoE loss patching path:
      _moe_aux_loss, _moe_z_loss, _moe_tokens_per_expert,
      _moe_router_logits_mean, _moe_router_logits_std, _moe_num_experts
    """

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            return

        # model may be passed directly or via kwargs
        if model is None:
            model = kwargs.get("model")
        if model is None:
            return

        metrics = self._collect_metrics(model)
        if metrics:
            logs.update(metrics)

    def _collect_metrics(self, model) -> dict[str, float]:
        aux_losses = []
        z_losses = []
        all_tokens_per_expert = []
        logit_means = []
        logit_stds = []
        num_experts = 0

        for module in model.modules():
            if type(module).__name__ != "Lfm2MoeSparseMoeBlock":
                continue

            if hasattr(module, "_moe_aux_loss"):
                aux_losses.append(module._moe_aux_loss.item())
            if hasattr(module, "_moe_z_loss"):
                z_losses.append(module._moe_z_loss.item())
            if hasattr(module, "_moe_tokens_per_expert"):
                all_tokens_per_expert.append(module._moe_tokens_per_expert)
                num_experts = module._moe_num_experts
            if hasattr(module, "_moe_router_logits_mean"):
                logit_means.append(module._moe_router_logits_mean.item())
            if hasattr(module, "_moe_router_logits_std"):
                logit_stds.append(module._moe_router_logits_std.item())

        if not all_tokens_per_expert:
            return {}

        metrics = {}

        if aux_losses:
            metrics["moe/aux_loss"] = sum(aux_losses) / len(aux_losses)
        if z_losses:
            metrics["moe/z_loss"] = sum(z_losses) / len(z_losses)

        # Aggregate token distribution across all layers
        # [n_layers, n_experts] → [n_experts] mean across layers
        stacked = torch.stack(all_tokens_per_expert)
        mean_tpe = stacked.mean(dim=0)  # [n_experts]
        total_tokens = mean_tpe.sum()

        if total_tokens > 0:
            # Coefficient of variation — measures load imbalance
            cv = mean_tpe.std() / (mean_tpe.mean() + 1e-8)
            metrics["moe/tokens_per_expert_cv"] = cv.item()

            # Expert utilization — fraction of experts receiving at least 1 token
            active_experts = (mean_tpe > 0).sum().item()
            metrics["moe/expert_utilization"] = (
                active_experts / num_experts if num_experts > 0 else 0.0
            )

            # Max expert load ratio — max tokens / mean tokens
            mean_load = total_tokens / num_experts if num_experts > 0 else 1.0
            metrics["moe/max_expert_load_ratio"] = (mean_tpe.max() / mean_load).item()

        if logit_means:
            metrics["moe/router_logit_mean"] = sum(logit_means) / len(logit_means)
        if logit_stds:
            metrics["moe/router_logit_std"] = sum(logit_stds) / len(logit_stds)

        # Router entropy — measures diversity of routing decisions
        if all_tokens_per_expert and total_tokens > 0:
            probs = mean_tpe / total_tokens
            probs = probs.clamp(min=1e-10)
            entropy = -(probs * probs.log()).sum().item()
            max_entropy = torch.tensor(float(num_experts)).log().item()
            metrics["moe/router_entropy"] = entropy
            metrics["moe/router_entropy_normalized"] = (
                entropy / max_entropy if max_entropy > 0 else 0.0
            )

        return metrics
