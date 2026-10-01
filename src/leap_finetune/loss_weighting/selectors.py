from __future__ import annotations

import json
import re
from dataclasses import dataclass

from leap_finetune.data_processing.importing import load_callable

from .config import SelectorConfig

CharSpan = tuple[int, int]


def _validate_spans(text: str, spans) -> list[CharSpan]:
    validated = []
    for raw_span in spans:
        if isinstance(raw_span, dict):
            start, end = raw_span.get("start"), raw_span.get("end")
        else:
            try:
                start, end = raw_span
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "selector callables must return (start, end) pairs or span dicts"
                ) from exc
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("selector span offsets must be integers")
        if start < 0 or end <= start or end > len(text):
            raise ValueError(
                f"invalid selector span ({start}, {end}) for text length {len(text)}"
            )
        validated.append((start, end))
    return validated


def _component_spans(
    *,
    key_span: CharSpan | None,
    delimiter_span: CharSpan | None,
    value_span: CharSpan,
    include_key: bool,
    include_delimiter: bool,
    include_value: bool,
) -> list[CharSpan]:
    components = []
    if include_key and key_span is not None:
        components.append(key_span)
    if include_delimiter and delimiter_span is not None:
        components.append(delimiter_span)
    if include_value:
        components.append(value_span)
    if not components:
        return []

    components.sort()
    merged = [components[0]]
    for start, end in components[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _key_value_line_spans(text: str, selector: SelectorConfig) -> list[CharSpan]:
    flags = re.MULTILINE
    if selector.ignore_case:
        flags |= re.IGNORECASE
    pattern = re.compile(
        rf"^[ \t]*(?:[-*][ \t]+)?({re.escape(selector.key)})([ \t]*:[ \t]*)(.*?)[ \t]*(?:\r?$)",
        flags,
    )
    result = []
    for match in pattern.finditer(text):
        value_start, value_end = match.span(3)
        result.extend(
            _component_spans(
                key_span=match.span(1),
                delimiter_span=match.span(2),
                value_span=(value_start, value_end),
                include_key=selector.include_key,
                include_delimiter=selector.include_delimiter,
                include_value=selector.include_value,
            )
        )
    return result


@dataclass(frozen=True)
class _JsonFieldSpan:
    key: CharSpan | None
    delimiter: CharSpan | None
    value: CharSpan


class _JsonSpanParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.length = len(text)
        self.position = 0
        self.fields: dict[str, _JsonFieldSpan] = {}

    def parse(self) -> dict[str, _JsonFieldSpan]:
        self._skip_space()
        self._parse_value("")
        self._skip_space()
        if self.position != self.length:
            raise ValueError(f"unexpected JSON content at offset {self.position}")
        return self.fields

    def _skip_space(self) -> None:
        while self.position < self.length and self.text[self.position].isspace():
            self.position += 1

    def _parse_string(self) -> tuple[str, CharSpan]:
        start = self.position
        if self.text[start] != '"':
            raise ValueError(f"expected JSON string at offset {start}")
        self.position += 1
        escaped = False
        while self.position < self.length:
            character = self.text[self.position]
            self.position += 1
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                raw = self.text[start : self.position]
                return json.loads(raw), (start, self.position)
        raise ValueError(f"unterminated JSON string at offset {start}")

    @staticmethod
    def _escape_pointer(value: str) -> str:
        return value.replace("~", "~0").replace("/", "~1")

    def _parse_value(self, path: str) -> CharSpan:
        self._skip_space()
        start = self.position
        if self.position >= self.length:
            raise ValueError("unexpected end of JSON input")
        character = self.text[self.position]
        if character == "{":
            self._parse_object(path)
        elif character == "[":
            self._parse_array(path)
        elif character == '"':
            self._parse_string()
        else:
            while (
                self.position < self.length
                and self.text[self.position] not in ",]} \t\r\n"
            ):
                self.position += 1
            json.loads(self.text[start : self.position])
        return start, self.position

    def _parse_object(self, path: str) -> None:
        self.position += 1
        self._skip_space()
        if self.position < self.length and self.text[self.position] == "}":
            self.position += 1
            return
        while True:
            key, key_span = self._parse_string()
            self._skip_space()
            if self.position >= self.length or self.text[self.position] != ":":
                raise ValueError(f"expected ':' at offset {self.position}")
            delimiter_start = self.position
            self.position += 1
            self._skip_space()
            value_start = self.position
            child_path = f"{path}/{self._escape_pointer(key)}"
            value_span = self._parse_value(child_path)
            self.fields[child_path] = _JsonFieldSpan(
                key=key_span,
                delimiter=(delimiter_start, value_start),
                value=value_span,
            )
            self._skip_space()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                self._skip_space()
                continue
            if self.position < self.length and self.text[self.position] == "}":
                self.position += 1
                return
            raise ValueError(f"expected ',' or '}}' at offset {self.position}")

    def _parse_array(self, path: str) -> None:
        self.position += 1
        self._skip_space()
        if self.position < self.length and self.text[self.position] == "]":
            self.position += 1
            return
        index = 0
        while True:
            child_path = f"{path}/{index}"
            value_span = self._parse_value(child_path)
            self.fields[child_path] = _JsonFieldSpan(None, None, value_span)
            index += 1
            self._skip_space()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                self._skip_space()
                continue
            if self.position < self.length and self.text[self.position] == "]":
                self.position += 1
                return
            raise ValueError(f"expected ',' or ']' at offset {self.position}")


def _json_pointer_spans(text: str, selector: SelectorConfig) -> list[CharSpan]:
    field = _JsonSpanParser(text).parse().get(selector.pointer)
    if field is None:
        return []
    return _component_spans(
        key_span=field.key,
        delimiter_span=field.delimiter,
        value_span=field.value,
        include_key=selector.include_key,
        include_delimiter=selector.include_delimiter,
        include_value=selector.include_value,
    )


def _regex_spans(text: str, selector: SelectorConfig) -> list[CharSpan]:
    flags = 0
    if selector.ignore_case:
        flags |= re.IGNORECASE
    if selector.multiline:
        flags |= re.MULTILINE
    if selector.dotall:
        flags |= re.DOTALL
    pattern = re.compile(selector.pattern, flags)
    spans = []
    for match in pattern.finditer(text):
        try:
            span = match.span(selector.group)
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"regex selector group {selector.group!r} does not exist"
            ) from exc
        if span != (-1, -1) and span[0] != span[1]:
            spans.append(span)
    return spans


def select_spans(text: str, selector: SelectorConfig | dict) -> list[CharSpan]:
    selector = (
        selector
        if isinstance(selector, SelectorConfig)
        else SelectorConfig.model_validate(selector)
    )
    if selector.type == "key_value_line":
        return _key_value_line_spans(text, selector)
    if selector.type == "json_pointer":
        return _json_pointer_spans(text, selector)
    if selector.type == "regex":
        return _regex_spans(text, selector)
    function = load_callable(selector.callable)
    return _validate_spans(text, function(text, **selector.kwargs))
