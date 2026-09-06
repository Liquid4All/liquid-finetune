import os
from pathlib import Path

from liquid_finetune.cli.main import main, run_config

HOME = Path.home()

_current_file = Path(__file__).resolve()
LIQUID_FINETUNE_DIR = _current_file.parent.parent.parent
LIQUID_FINETUNE_DIR = Path(os.getenv("LIQUID_FINETUNE_DIR", LIQUID_FINETUNE_DIR))

RUNTIME_DIR = LIQUID_FINETUNE_DIR / "src" / "liquid_finetune"

BASE_OUTPUT_PATH = LIQUID_FINETUNE_DIR / "outputs"
SFT_OUTPUT_PATH = BASE_OUTPUT_PATH / "sft"
DPO_OUTPUT_PATH = BASE_OUTPUT_PATH / "dpo"
KTO_OUTPUT_PATH = BASE_OUTPUT_PATH / "kto"
GRPO_OUTPUT_PATH = BASE_OUTPUT_PATH / "grpo"

TOKENIZATION_CACHE_DIR = LIQUID_FINETUNE_DIR / ".cache" / "tokenized"


__all__ = [
    "BASE_OUTPUT_PATH",
    "DPO_OUTPUT_PATH",
    "GRPO_OUTPUT_PATH",
    "HOME",
    "KTO_OUTPUT_PATH",
    "LIQUID_FINETUNE_DIR",
    "RUNTIME_DIR",
    "SFT_OUTPUT_PATH",
    "TOKENIZATION_CACHE_DIR",
    "main",
    "run_config",
]
