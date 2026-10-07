import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# === Marker constants ===

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_LIST_START = "<|tool_list_start|>"
TOOL_LIST_END = "<|tool_list_end|>"
TOOL_RESPONSE_START = "<|tool_response_start|>"
TOOL_RESPONSE_END = "<|tool_response_end|>"

_FOREIGN_TOOL_MARKERS = [
    ("<tool_call>", "Qwen"),
    ("</tool_call>", "Qwen"),
    ("[TOOL_CALLS]", "Mistral"),
    ("[/TOOL_CALLS]", "Mistral"),
    ("<function_call>", "unknown"),
    ("</function_call>", "unknown"),
    ("<|plugin|>", "unknown"),
    ("<start_function_call>", "unknown"),
    ("<end_function_call>", "unknown"),
]

FOREIGN_MARKERS = [
    "<tool_call>",
    "</tool_call>",
    "[TOOL_CALLS]",
    "[/TOOL_CALLS]",
    "<function_call>",
    "</function_call>",
    "<|plugin|>",
    "<start_function_call>",
    "<end_function_call>",
]


def has_foreign_tool_markers(text: str) -> str | None:
    """Return the foreign tool-call format name if known markers are present."""
    for marker, format_name in _FOREIGN_TOOL_MARKERS:
        if marker in text:
            return format_name
    return None


# === Detection ===


@dataclass
class ToolFormatInfo:
    has_tool_calls: bool = False
    has_tool_call_markers: bool = False
    has_tool_list_markers: bool = False
    has_tool_response_markers: bool = False
    has_tool_role: bool = False
    has_structured_tool_calls: bool = False
    has_foreign_markers: bool = False
    foreign_markers_found: list[str] = field(default_factory=list)
    detected_format: str = "none"


def detect_tool_format(samples: list[dict]) -> ToolFormatInfo:
    """Scan a list of samples for tool call markers and return a format descriptor.

    Each sample should have a 'messages' or 'conversations' key containing a
    list of message dicts with 'role' and 'content'.
    """
    info = ToolFormatInfo()
    foreign_set = set()

    for sample in samples:
        messages = sample.get("messages") or sample.get("conversations", [])
        if not messages:
            continue

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""

            if role == "tool":
                info.has_tool_role = True

            if TOOL_CALL_START in content:
                info.has_tool_call_markers = True
            if TOOL_LIST_START in content:
                info.has_tool_list_markers = True
            if TOOL_RESPONSE_START in content:
                info.has_tool_response_markers = True

            if role == "assistant" and "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    info.has_structured_tool_calls = True

            for marker in FOREIGN_MARKERS:
                if marker in content:
                    foreign_set.add(marker)

    info.foreign_markers_found = sorted(foreign_set)
    info.has_foreign_markers = len(foreign_set) > 0
    info.has_tool_calls = (
        info.has_tool_call_markers
        or info.has_tool_role
        or info.has_structured_tool_calls
        or info.has_foreign_markers
    )

    # Determine format
    if info.has_foreign_markers:
        info.detected_format = "foreign"
    elif info.has_structured_tool_calls:
        info.detected_format = "structured"
    elif info.has_tool_list_markers and info.has_tool_call_markers:
        info.detected_format = "lfm2"
    elif info.has_tool_call_markers:
        info.detected_format = "lfm25"
    else:
        info.detected_format = "none"

    return info


# === Validation ===


@dataclass
class ToolFormatIssue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    auto_fixable: bool = False
    fix_hint: str = ""


def validate_tool_format(
    format_info: ToolFormatInfo,
    model_family: str,
) -> list[ToolFormatIssue]:
    """Check format compatibility with target model. Return issues sorted by severity."""
    issues = []

    if not format_info.has_tool_calls:
        return issues

    if format_info.has_foreign_markers:
        markers = ", ".join(format_info.foreign_markers_found)
        issues.append(
            ToolFormatIssue(
                "error",
                "FOREIGN_MARKERS",
                f"Found foreign tool call markers: {markers}. Not LFM format.",
                fix_hint=f'Convert to LFM bracket notation: {TOOL_CALL_START}[func(arg="value")]{TOOL_CALL_END}',
            )
        )

    if format_info.has_structured_tool_calls:
        issues.append(
            ToolFormatIssue(
                "warning",
                "STRUCTURED_TOOL_CALLS",
                "Found 'tool_calls' field on assistant messages. LFM templates drop this. Will auto-convert to bracket notation.",
                auto_fixable=True,
            )
        )

    if format_info.has_tool_response_markers:
        issues.append(
            ToolFormatIssue(
                "info",
                "PREBAKED_TOOL_RESPONSE",
                f"Stripping pre-baked {TOOL_RESPONSE_START} from role=tool to prevent double wrapping.",
                auto_fixable=True,
            )
        )

    if format_info.has_tool_list_markers and model_family == "lfm25":
        issues.append(
            ToolFormatIssue(
                "warning",
                "TOOL_LIST_MARKERS_ON_LFM25",
                f"Found {TOOL_LIST_START} markers not used by LFM2.5. Will strip.",
                auto_fixable=True,
            )
        )

    if (
        format_info.has_tool_call_markers
        and not format_info.has_tool_list_markers
        and model_family == "lfm2"
    ):
        issues.append(
            ToolFormatIssue(
                "warning",
                "MISSING_TOOL_LIST_MARKERS",
                f"Missing {TOOL_LIST_START} markers expected by LFM2. Training will work but won't fully match pretraining.",
                fix_hint=f"Wrap tool defs with {TOOL_LIST_START}[...]{TOOL_LIST_END} or use tools= parameter.",
            )
        )

    if format_info.detected_format == model_family:
        issues.append(
            ToolFormatIssue(
                "info",
                "FORMAT_MATCHED",
                f"Tool call format matches {model_family.upper()} model.",
            )
        )

    return sorted(
        issues, key=lambda i: {"error": 0, "warning": 1, "info": 2}[i.severity]
    )


# === Normalization (per-row, for Ray .map()) ===


def _format_tool_call_value(v) -> str:
    # json.dumps gives double-quoted + proper escaping of
    # ", \n, \. Everything else: repr()
    # gives valid Python literal syntax
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    return repr(v)


def tool_calls_to_pythonic(tool_calls: list[dict]) -> str:
    """Convert structured tool_calls list to LFM pythonic bracket notation."""
    calls = []
    for tc in tool_calls:
        func = tc.get("function", tc)
        name = func["name"]
        args = func.get("arguments", {})
        if args is None:
            args = {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                rendered_args = args.strip()
                calls.append(f"{name}({rendered_args})")
                continue
        # Skip malformed tool calls instead of failing the whole row.
        if not isinstance(args, dict):
            if isinstance(args, str):
                rendered_args = args.strip()
                calls.append(f"{name}({rendered_args})")
                continue
            logger.warning(
                "Skipping tool_call with non-dict arguments (got %s): name=%s",
                type(args).__name__,
                name,
            )
            continue
        parts = [f"{k}={_format_tool_call_value(v)}" for k, v in args.items()]
        calls.append(f"{name}({', '.join(parts)})")

    return f"{TOOL_CALL_START}[{', '.join(calls)}]{TOOL_CALL_END}"


def _normalize_argument_payload_for_template(args):
    if args is None:
        return {}
    if not isinstance(args, str):
        return args

    try:
        parsed_args = json.loads(args)
    except json.JSONDecodeError:
        raise ValueError(
            "Structured tool_call arguments must be a dict, JSON object string, "
            "or None before chat-template rendering"
        ) from None

    if parsed_args is None:
        return {}
    if isinstance(parsed_args, dict):
        return parsed_args
    raise ValueError(
        "Structured tool_call arguments JSON must decode to an object before "
        "chat-template rendering"
    )


def _normalize_tool_call_arguments_for_template(tool_calls: list[dict]) -> list[dict]:
    """Canonicalize structured tool-call arguments before chat-template rendering."""
    normalized_tool_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized_tool_calls.append(tool_call)
            continue

        new_tool_call = dict(tool_call)
        func = tool_call.get("function")
        if isinstance(func, dict):
            new_func = dict(func)
            new_func["arguments"] = _normalize_argument_payload_for_template(
                new_func.get("arguments", {})
            )
            new_tool_call["function"] = new_func
        else:
            new_tool_call["arguments"] = _normalize_argument_payload_for_template(
                new_tool_call.get("arguments", {})
            )

        normalized_tool_calls.append(new_tool_call)

    return normalized_tool_calls


def _is_message_list(value) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict) and "role" in item for item in value
    )


def normalize_messages_for_chat_template(messages: list[dict]) -> list[dict]:
    """Normalize structured tool-call arguments before applying a chat template."""
    normalized_messages = []
    modified = False

    for message in messages:
        if not isinstance(message, dict):
            normalized_messages.append(message)
            continue

        new_message = dict(message)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            new_tool_calls = _normalize_tool_call_arguments_for_template(tool_calls)
            if new_tool_calls != tool_calls:
                new_message["tool_calls"] = new_tool_calls
                modified = True

        normalized_messages.append(new_message)

    return normalized_messages if modified else messages


def normalize_row_for_chat_template(row: dict) -> dict:
    """Normalize any conversational fields before TRL/HF chat templating."""
    normalized_row = None
    for key in ("messages", "prompt", "chosen", "rejected"):
        value = row.get(key)
        if not _is_message_list(value):
            continue

        normalized_value = normalize_messages_for_chat_template(value)
        if normalized_value is not value:
            if normalized_row is None:
                normalized_row = dict(row)
            normalized_row[key] = normalized_value

    return row if normalized_row is None else normalized_row


def normalize_tool_format(row: dict, model_family: str) -> dict:
    """Auto-fix tool call format mismatches. Stateless, used as Ray .map() fn."""
    messages = row.get("messages") or row.get("conversations")
    if not messages:
        return row

    modified = False
    new_messages = []

    for msg in messages:
        new_msg = dict(msg)
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        # 1. Strip tool_list markers for LFM2.5
        if role == "system" and model_family == "lfm25":
            if TOOL_LIST_START in content or TOOL_LIST_END in content:
                content = content.replace(TOOL_LIST_START, "")
                content = content.replace(TOOL_LIST_END, "")
                new_msg["content"] = content
                modified = True

        # 2. Strip pre-baked tool_response markers (prevents double wrapping)
        if role == "tool":
            if TOOL_RESPONSE_START in content or TOOL_RESPONSE_END in content:
                content = content.replace(TOOL_RESPONSE_START, "")
                content = content.replace(TOOL_RESPONSE_END, "")
                new_msg["content"] = content
                modified = True

        # 3. Convert structured tool_calls to pre-baked content
        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                pythonic = tool_calls_to_pythonic(tool_calls)
                existing_content = content.strip()
                if existing_content and model_family == "lfm25":
                    new_msg["content"] = f"{existing_content}\n{pythonic}"
                elif existing_content:
                    new_msg["content"] = f"{pythonic}\n{existing_content}"
                else:
                    new_msg["content"] = pythonic
                modified = True
            new_msg.pop("tool_calls", None)
            modified = True

        new_messages.append(new_msg)

    if not modified:
        return row

    # Preserve the original column name
    col_name = "messages" if "messages" in row else "conversations"
    new_row = dict(row)
    new_row[col_name] = new_messages
    return new_row


def get_tool_normalizer(model_family: str):
    """Return a Ray-compatible map function for tool call normalization."""

    def _normalize(row: dict) -> dict:
        return normalize_tool_format(row, model_family)

    return _normalize


def _tool_call_must_be_first(model_family: str) -> bool:
    """Legacy LFM2 expects tool-call-first text; LFM2.5 permits prose first."""
    return model_family != "lfm25"


def validate_tool_calls_in_messages(
    messages: list,
    sample_idx: int,
    model_family: str = "lfm2",
) -> None:
    """Validate LFM tool-call formatting in a message list."""
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        if role != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue

        # === Foreign Markers ===
        format_name = has_foreign_tool_markers(content)
        if format_name:
            raise ValueError(
                f"Sample {sample_idx}, message {msg_idx}: Found {format_name} tool call markers "
                f"in assistant content. LFM requires '{TOOL_CALL_START}'/'{TOOL_CALL_END}' "
                f"markers in the content field."
            )

        # === Tool-First Ordering ===
        tool_call_pos = content.find(TOOL_CALL_START)
        if tool_call_pos == -1:
            continue

        text_before = content[:tool_call_pos].strip()
        if text_before and _tool_call_must_be_first(model_family):
            raise ValueError(
                f"Sample {sample_idx}, message {msg_idx}: Text appears before tool call "
                f"in assistant content. Legacy LFM2 expects tool call first, then text. "
                f"Found: '{text_before[:50]}...' before {TOOL_CALL_START}"
            )

    # === Tool Response Pairing ===
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")

        structured_tool_calls = msg.get("tool_calls")
        has_tool_call = (
            isinstance(content, str)
            and TOOL_CALL_START in content
            or (
                role == "assistant"
                and isinstance(structured_tool_calls, list)
                and len(structured_tool_calls) > 0
            )
        )
        if not has_tool_call or role != "assistant":
            continue

        found_tool_response = False
        for next_msg in messages[msg_idx + 1 :]:
            if not isinstance(next_msg, dict):
                continue
            next_role = next_msg.get("role", "")
            if next_role == "tool":
                found_tool_response = True
                break
            if next_role == "user":
                break

        if not found_tool_response:
            raise ValueError(
                f"Sample {sample_idx}, message {msg_idx}: Assistant has tool call but "
                f"no tool response (role='tool') found before next user message."
            )


def validate_tool_calls_in_text(
    text: str,
    sample_idx: int,
    field: str,
    model_family: str = "lfm2",
) -> None:
    """Validate LFM tool-call formatting in a pre-formatted DPO string."""
    format_name = has_foreign_tool_markers(text)
    if format_name:
        raise ValueError(
            f"Sample {sample_idx}, {field}: Found {format_name} tool call markers. "
            f"LFM requires '{TOOL_CALL_START}'/'{TOOL_CALL_END}' markers."
        )

    tool_call_pos = text.find(TOOL_CALL_START)
    if tool_call_pos == -1:
        return

    text_before = text[:tool_call_pos].strip()
    if text_before and _tool_call_must_be_first(model_family):
        raise ValueError(
            f"Sample {sample_idx}, {field}: Text appears before tool call. "
            f"Legacy LFM2 expects tool call first, then text. "
            f"Found: '{text_before[:50]}...' before {TOOL_CALL_START}"
        )


def validate_tool_calls_dpo(
    chosen: Any,
    rejected: Any,
    sample_idx: int,
    model_family: str = "lfm2",
) -> None:
    """Validate LFM tool-call formatting in DPO chosen/rejected data."""
    for field_name, data in [("chosen", chosen), ("rejected", rejected)]:
        if isinstance(data, str):
            validate_tool_calls_in_text(data, sample_idx, field_name, model_family)
        elif isinstance(data, list):
            validate_tool_calls_in_messages(data, sample_idx, model_family)
