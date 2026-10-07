from peft import LoraConfig, TaskType

GLU_MODULES = ["feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]
MHA_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
]
CONV_MODULES = ["conv.in_proj", "conv.out_proj"]


DEFAULT_LORA = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=GLU_MODULES + MHA_MODULES + CONV_MODULES,
)

HIGH_R_LORA = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=GLU_MODULES + MHA_MODULES + CONV_MODULES,
)


LFM_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "conv.in_proj",
]
VISION_TOWER_MODULES = ["fc1", "fc2"]
MULTI_MODAL_PROJECTOR_MODULES = ["linear_1", "linear_2"]

DEFAULT_VLM_LORA = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=LFM_MODULES + VISION_TOWER_MODULES + MULTI_MODAL_PROJECTOR_MODULES,
)

MINIMAL_VLM_LORA = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    target_modules=LFM_MODULES + VISION_TOWER_MODULES + MULTI_MODAL_PROJECTOR_MODULES,
)


########################
#     MOE CONFIGS      #
########################

MOE_LORA = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules="all-linear",  # Target all linear layers in MoE architecture
)

MOE_LORA_HIGH_R = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules="all-linear",  # Target all linear layers in MoE architecture
)
