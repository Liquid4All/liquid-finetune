def is_moe_model_from_name(model_name: str) -> bool:
    moe_indicators = ["8B-A1B", "8BA1B", "24B-A2B", "24BA2B", "moe", "MoE"]
    return any(indicator.lower() in model_name.lower() for indicator in moe_indicators)


def get_model_family(model_name: str) -> str:
    """Return model family for format-specific behavior."""
    model_lower = model_name.lower()
    if "2.5" in model_lower:
        return "lfm25"
    if "24b" in model_lower and "2.5" not in model_lower:
        return "lfm25"
    return "lfm2"
