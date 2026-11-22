import logging
import os
import pathlib
import warnings

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LFM_TIED_WORD_EMBEDDING_MODEL_TYPES = {"lfm2", "lfm2_moe", "lfm3", "lfm3_vl"}
LFM2_ROPE_OVERRIDE_MODEL_TYPES = {"lfm2", "lfm2_moe"}
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_LFM2_5_DEFAULT_CHAT_TEMPLATE_PATH = (
    _REPO_ROOT / "job_configs" / "chat_templates" / "lfm2_5_chat_template.jinja"
)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_ATTN_FALLBACK_WARNING_EMITTED = False


def _get_attn_implementation() -> str:
    usable, reason = _flash_attn_2_status()
    if usable:
        return "flash_attention_2"

    if _requires_flash_attn_2():
        raise RuntimeError(
            f"LIQUID_REQUIRE_FLASH_ATTN_2 is set, but flash-attn is not usable: {reason}"
        )

    _warn_flash_attn_2_fallback(reason)
    return "sdpa"


def _requires_flash_attn_2() -> bool:
    return os.getenv("LIQUID_REQUIRE_FLASH_ATTN_2", "").lower() in _TRUE_VALUES


def _warn_flash_attn_2_fallback(reason: str) -> None:
    global _ATTN_FALLBACK_WARNING_EMITTED
    if _ATTN_FALLBACK_WARNING_EMITTED:
        return
    _ATTN_FALLBACK_WARNING_EMITTED = True
    message = f"FlashAttention 2 not available ({reason}); falling back to SDPA."
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _flash_attn_2_status() -> tuple[bool, str]:
    if not is_flash_attn_2_available():
        return False, "Transformers does not report flash-attn 2 as available"

    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401
    except Exception as exc:
        return False, f"flash-attn import failed: {exc}"

    return True, "flash-attn 2 is usable"


def _resolve_model_id(model_name: str) -> str:
    """Resolve model_name to a local path or HuggingFace model ID."""
    model_path = pathlib.Path(model_name)
    if model_path.exists() and model_path.is_dir():
        return model_name
    if "/" in model_name or "://" in model_name:
        return model_name
    return f"LiquidAI/{model_name}"


def _is_lfm25_chat_template_model(model_name: str) -> bool:
    model_lower = model_name.lower()
    return any(
        marker in model_lower
        for marker in (
            "lfm2.5",
            "lfm2-24b",
            "lfm2_24b",
        )
    )


def _default_chat_template_path(model_name: str) -> pathlib.Path | None:
    model_path = pathlib.Path(model_name)
    if model_path.exists() and model_path.is_dir():
        return None
    if _is_lfm25_chat_template_model(model_name):
        return _LFM2_5_DEFAULT_CHAT_TEMPLATE_PATH
    return None


def _resolve_chat_template(
    model_name: str = "",
    chat_template: str | None = None,
    chat_template_path: str | None = None,
) -> str | None:
    if chat_template and chat_template_path:
        raise ValueError("Specify either chat_template or chat_template_path, not both")

    if chat_template_path:
        return pathlib.Path(chat_template_path).expanduser().read_text()

    default_template_path = _default_chat_template_path(model_name)
    if default_template_path is not None and default_template_path.exists():
        logger.info(
            "Using default tokenizer chat template for %s from %s",
            model_name,
            default_template_path,
        )
        return default_template_path.read_text()

    return chat_template


def _apply_chat_template_override(
    tokenizer: AutoTokenizer,
    *,
    model_name: str,
    chat_template: str | None = None,
    chat_template_path: str | None = None,
) -> AutoTokenizer:
    resolved = _resolve_chat_template(model_name, chat_template, chat_template_path)
    if resolved:
        tokenizer.chat_template = resolved
        logger.info("Applied tokenizer chat template override")
    return tokenizer


def normalize_model_config_overrides(
    config: AutoConfig,
    model_config: dict | None,
) -> dict:
    """Normalize user overrides to the config schema expected by the model."""
    if not model_config:
        return {}

    normalized = dict(model_config)

    # LFM2 / LFM2-MoE consumers do not all read the same RoPE field. Keep the
    # native rope_parameters field populated even when the user only overrides
    # top-level rope_theta.
    model_type = getattr(config, "model_type", "")
    if model_type in LFM_TIED_WORD_EMBEDDING_MODEL_TYPES:
        if normalized.get("tie_word_embeddings") is False:
            raise ValueError("LFM models require tied word embeddings")
        if normalized.get("tie_embedding") is False:
            raise ValueError("LFM models require tied word embeddings")

    if model_type in LFM2_ROPE_OVERRIDE_MODEL_TYPES:
        rope_parameters = None
        if "rope_parameters" in normalized:
            rope_parameters = dict(normalized["rope_parameters"])
        elif "rope_scaling" in normalized:
            rope_parameters = dict(normalized["rope_scaling"])
        elif "rope_theta" in normalized:
            base_rope_parameters = getattr(config, "rope_parameters", None)
            if isinstance(base_rope_parameters, dict):
                rope_parameters = dict(base_rope_parameters)
            else:
                rope_parameters = {}

        if rope_parameters is not None:
            if normalized.get("rope_theta") is not None:
                rope_parameters["rope_theta"] = normalized["rope_theta"]
            elif (
                "rope_theta" not in rope_parameters
                or rope_parameters["rope_theta"] is None
            ):
                rope_parameters["rope_theta"] = getattr(config, "default_theta", None)
            if "rope_type" not in rope_parameters and "type" in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters["type"]
            normalized["rope_parameters"] = rope_parameters

    return normalized


def _is_moe_model(model: AutoModelForCausalLM) -> bool:
    model_type = getattr(model.config, "model_type", "")
    architectures = getattr(model.config, "architectures", []) or []
    return "moe" in model_type.lower() or any("Moe" in arch for arch in architectures)


def _maybe_enable_grouped_mm(model: AutoModelForCausalLM) -> None:
    """Enable grouped_mm only for MoE models that expose the hook."""
    if os.getenv("LIQUID_ENABLE_GROUPED_MM", "1") == "0":
        logger.info("Skipping grouped_mm expert implementation override")
        return
    if not hasattr(model, "set_experts_implementation") or not _is_moe_model(model):
        return
    model.set_experts_implementation("grouped_mm")


def _enforce_lfm_tied_word_embeddings(model: AutoModelForCausalLM) -> None:
    if (
        getattr(model.config, "model_type", "")
        not in LFM_TIED_WORD_EMBEDDING_MODEL_TYPES
    ):
        return
    if not hasattr(model, "tie_weights"):
        return
    model.config.tie_word_embeddings = True
    model.tie_weights()
    logger.info("Enforced tied LFM input/output word embeddings")


def load_tokenizer(
    model_name: str,
    *,
    chat_template: str | None = None,
    chat_template_path: str | None = None,
) -> AutoTokenizer:
    """Load only the tokenizer (lightweight, no model weights)."""
    model_id = _resolve_model_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _apply_chat_template_override(
        tokenizer,
        model_name=model_name,
        chat_template=chat_template,
        chat_template_path=chat_template_path,
    )


def load_model(
    model_name: str,
    model_config: dict | None = None,
    *,
    chat_template: str | None = None,
    chat_template_path: str | None = None,
    install_memory_efficient_loss: bool = True,
    enable_grouped_mm: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model from the Hugging Face Hub or from a local path.

    Args:
        model_name: HuggingFace model ID or local path
        model_config: optional overrides applied to the model config before loading
            (e.g. rope_scaling, max_position_embeddings for YaRN)
    """
    attn_impl = _get_attn_implementation()
    model_id = _resolve_model_id(model_name)
    logger.info(f"Loading model: {model_id}")

    # LFM3 ships its modeling code with the checkpoint (``auto_map``), like the
    # VLM and retrieval paths below; it cannot be loaded without remote code.
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    normalized_model_config = normalize_model_config_overrides(config, model_config)
    if normalized_model_config:
        for key, value in normalized_model_config.items():
            setattr(config, key, value)
        logger.info(
            f"Applied model config overrides: {list(normalized_model_config.keys())}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    _enforce_lfm_tied_word_embeddings(model)
    model.config.use_cache = False
    if (
        install_memory_efficient_loss
        and os.getenv("LIQUID_INSTALL_MEMORY_EFFICIENT_LOSS", "1") == "1"
    ):
        from liquid_finetune.training.utils.causal_lm_loss import (
            install_memory_efficient_causal_lm_loss,
        )

        install_memory_efficient_causal_lm_loss(model)
    else:
        logger.info("Skipping memory-efficient causal LM loss install")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = _apply_chat_template_override(
        tokenizer,
        model_name=model_name,
        chat_template=chat_template,
        chat_template_path=chat_template_path,
    )

    if enable_grouped_mm:
        _maybe_enable_grouped_mm(model)
    else:
        logger.info("Skipping grouped_mm expert implementation override")

    logger.info(f"Architecture: {model.config.architectures[0]}")
    logger.info(f"Model type: {model.config.model_type}")
    logger.info(
        f"Layers: {model.config.num_hidden_layers}, Dim: {model.config.hidden_size}"
    )
    logger.info(f"Vocab size: {model.config.vocab_size}")

    return model, tokenizer


def load_vlm_model(
    model_name: str,
    min_image_tokens: int | None = None,
    max_image_tokens: int | None = None,
    do_image_splitting: bool = True,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load a VLM model from the Hugging Face Hub or from a local path."""

    model_id = _resolve_model_id(model_name)
    logger.info(f"Loading VLM: {model_id}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    processor_kwargs = {"trust_remote_code": True}
    if model_type == "lfm3_vl":
        if not do_image_splitting:
            logger.warning(
                "LFM3-VL uses dynamic-resolution images; do_image_splitting=False "
                "has no effect"
            )
        if min_image_tokens is not None or max_image_tokens is not None:
            if min_image_tokens is not None and min_image_tokens <= 0:
                raise ValueError("min_image_tokens must be positive")
            if max_image_tokens is not None and max_image_tokens <= 0:
                raise ValueError("max_image_tokens must be positive")
            vision_config = config.vision_config
            patch_size = int(vision_config.patch_size)
            merge_size = int(vision_config.spatial_merge_size)
            pixels_per_token = (patch_size * merge_size) ** 2
            min_pixels = (
                65_536
                if min_image_tokens is None
                else int(min_image_tokens) * pixels_per_token
            )
            max_pixels = (
                16_777_216
                if max_image_tokens is None
                else int(max_image_tokens) * pixels_per_token
            )
            if min_pixels > max_pixels:
                raise ValueError("min_image_tokens cannot exceed max_image_tokens")
            processor_kwargs["size"] = {
                "shortest_edge": min_pixels,
                "longest_edge": max_pixels,
            }
            logger.info(
                "Limiting LFM3-VL images to %d-%d pixels",
                min_pixels,
                max_pixels,
            )
    else:
        processor_kwargs["do_image_splitting"] = do_image_splitting
        if min_image_tokens is not None:
            processor_kwargs["min_image_tokens"] = min_image_tokens
        if max_image_tokens is not None:
            processor_kwargs["max_image_tokens"] = max_image_tokens

    # SDPA is the common compatible implementation for the supported VLM towers.
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    # Disable KV cache for training (required for gradient checkpointing)
    model.config.use_cache = False

    # Ensure padding is configured correctly
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor
