import json
import re
from json import JSONDecodeError
from typing import Any


FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def _parse_json_if_tool_call(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate)
    except JSONDecodeError:
        return None
    if isinstance(parsed, dict) and "tool_call" in parsed and isinstance(parsed["tool_call"], dict):
        return parsed
    return None


def _extract_balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    stack = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    objects.append(text[start : i + 1])
                    start = None
    return objects


def extract_tool_call(text: str) -> dict | None:
    if not text:
        return None
    stripped = text.strip()

    parsed = _parse_json_if_tool_call(stripped)
    if parsed:
        return parsed

    for match in FENCED_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        parsed = _parse_json_if_tool_call(candidate)
        if parsed:
            return parsed

    for candidate in _extract_balanced_json_objects(text):
        parsed = _parse_json_if_tool_call(candidate)
        if parsed:
            return parsed

    return None


def clean_text_response(text: str) -> str:
    if not text:
        return ""

    tool_call = extract_tool_call(text)
    if not tool_call:
        return text.strip()

    serialized = json.dumps(tool_call, ensure_ascii=False)

    cleaned = text.replace(serialized, "")

    for match in FENCED_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        if _parse_json_if_tool_call(candidate):
            cleaned = cleaned.replace(match.group(0), "")

    return cleaned.strip()
