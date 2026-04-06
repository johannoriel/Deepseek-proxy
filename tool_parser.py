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


def _normalize_to_tool_calls(parsed: Any) -> list[dict[str, Any]]:
    """Normalize JSON payloads into a list of {tool_call: {name, arguments}} entries."""
    results: list[dict[str, Any]] = []

    def _coerce_one(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        if "tool_call" in item and isinstance(item["tool_call"], dict):
            tc = item["tool_call"]
            return {"tool_call": {"name": tc.get("name", ""), "arguments": tc.get("arguments", {})}}

        if "tool_calls" in item and isinstance(item["tool_calls"], list):
            coerced: list[dict[str, Any]] = []
            for tc in item["tool_calls"]:
                normalized = _coerce_one(tc)
                if normalized:
                    coerced.append(normalized)
            if coerced:
                return {"tool_calls": coerced}
            return None

        if "name" in item and ("arguments" in item or "function" in item):
            if "function" in item and isinstance(item["function"], dict):
                fn = item["function"]
                return {
                    "tool_call": {
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", {}),
                    }
                }
            return {"tool_call": {"name": item.get("name", ""), "arguments": item.get("arguments", {})}}

        return None

    if isinstance(parsed, list):
        for entry in parsed:
            coerced = _coerce_one(entry)
            if not coerced:
                continue
            if "tool_calls" in coerced:
                results.extend(coerced["tool_calls"])
            else:
                results.append(coerced)
        return results

    coerced = _coerce_one(parsed)
    if not coerced:
        return []
    if "tool_calls" in coerced:
        return coerced["tool_calls"]
    return [coerced]


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
    calls = extract_tool_calls(text)
    return calls[0] if calls else None


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    stripped = text.strip()
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append_normalized(parsed: Any) -> None:
        for entry in _normalize_to_tool_calls(parsed):
            fingerprint = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            found.append(entry)

    try:
        parsed = json.loads(stripped)
        _append_normalized(parsed)
    except JSONDecodeError:
        pass

    for match in FENCED_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            _append_normalized(parsed)
        except JSONDecodeError:
            continue

    for candidate in _extract_balanced_json_objects(text):
        try:
            parsed = json.loads(candidate)
            _append_normalized(parsed)
        except JSONDecodeError:
            continue

    return found


def clean_text_response(text: str) -> str:
    if not text:
        return ""

    tool_calls = extract_tool_calls(text)
    if not tool_calls:
        return text.strip()

    cleaned = text

    for match in FENCED_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
        except JSONDecodeError:
            continue
        if _normalize_to_tool_calls(parsed):
            cleaned = cleaned.replace(match.group(0), "")

    for candidate in _extract_balanced_json_objects(text):
        try:
            parsed = json.loads(candidate)
        except JSONDecodeError:
            continue
        if _normalize_to_tool_calls(parsed):
            cleaned = cleaned.replace(candidate, "")

    return cleaned.strip()
