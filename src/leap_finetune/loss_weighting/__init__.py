"""Semantic per-token weighting for supervised causal language modeling."""

from .alignment import build_token_loss_weights
from .config import LossWeightingConfig
from .loss import weighted_causal_lm_loss
from .selectors import select_spans

__all__ = [
    "LossWeightingConfig",
    "build_token_loss_weights",
    "select_spans",
    "weighted_causal_lm_loss",
]
