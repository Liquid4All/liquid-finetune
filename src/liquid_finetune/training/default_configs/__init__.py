from liquid_finetune.training.default_configs.dpo_configs import (
    DEFAULT_DPO,
    DEFAULT_VLM_DPO,
    MOE_DPO,
)
from liquid_finetune.training.default_configs.grpo_configs import (
    DEFAULT_GRPO,
    DEFAULT_VLM_GRPO,
    MOE_GRPO,
)
from liquid_finetune.training.default_configs.retrieval_configs import (
    DEFAULT_COLBERT,
    DEFAULT_EMBEDDING,
)
from liquid_finetune.training.default_configs.kto_configs import DEFAULT_KTO
from liquid_finetune.training.default_configs.sft_configs import DEFAULT_SFT, MOE_SFT
from liquid_finetune.training.default_configs.vlm_sft_configs import DEFAULT_VLM_SFT
from liquid_finetune.training.peft.peft_configs import (
    DEFAULT_LORA,
    DEFAULT_VLM_LORA,
    HIGH_R_LORA,
    MINIMAL_VLM_LORA,
    MOE_LORA,
    MOE_LORA_HIGH_R,
)

TRAINING_DEFAULTS = {
    "DEFAULT_SFT": DEFAULT_SFT,
    "DEFAULT_DPO": DEFAULT_DPO,
    "DEFAULT_KTO": DEFAULT_KTO,
    "DEFAULT_EMBEDDING": DEFAULT_EMBEDDING,
    "DEFAULT_COLBERT": DEFAULT_COLBERT,
    "DEFAULT_VLM_SFT": DEFAULT_VLM_SFT,
    "DEFAULT_VLM_DPO": DEFAULT_VLM_DPO,
    "MOE_SFT": MOE_SFT,
    "MOE_DPO": MOE_DPO,
    "DEFAULT_GRPO": DEFAULT_GRPO,
    "DEFAULT_VLM_GRPO": DEFAULT_VLM_GRPO,
    "MOE_GRPO": MOE_GRPO,
}

PEFT_DEFAULTS = {
    "DEFAULT_LORA": DEFAULT_LORA,
    "HIGH_R_LORA": HIGH_R_LORA,
    "DEFAULT_VLM_LORA": DEFAULT_VLM_LORA,
    "MINIMAL_VLM_LORA": MINIMAL_VLM_LORA,
    "MOE_LORA": MOE_LORA,
    "MOE_LORA_HIGH_R": MOE_LORA_HIGH_R,
}

__all__ = [
    "DEFAULT_DPO",
    "DEFAULT_EMBEDDING",
    "DEFAULT_COLBERT",
    "DEFAULT_GRPO",
    "DEFAULT_KTO",
    "DEFAULT_LORA",
    "DEFAULT_SFT",
    "DEFAULT_VLM_DPO",
    "DEFAULT_VLM_GRPO",
    "DEFAULT_VLM_LORA",
    "DEFAULT_VLM_SFT",
    "HIGH_R_LORA",
    "MINIMAL_VLM_LORA",
    "MOE_DPO",
    "MOE_GRPO",
    "MOE_LORA",
    "MOE_LORA_HIGH_R",
    "MOE_SFT",
    "PEFT_DEFAULTS",
    "TRAINING_DEFAULTS",
]
