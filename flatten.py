import json
from copy import deepcopy
from typing import Any


SEPARATOR = "\n\n---\n\n"


def collapse_consecutive_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive messages with the same role into one message."""
    collapsed: list[dict[str, Any]] = []
    for msg in messages:
        current = deepcopy(msg)
        should_merge = bool(collapsed and collapsed[-1].get("role") == current.get("role"))
        if should_merge and current.get("role") == "tool":
            # Never merge tool responses: each tool_call_id must remain distinct.
            should_merge = False

        if should_merge:
            prev = collapsed[-1]
            prev_content = prev.get("content") or ""
            new_content = current.get("content") or ""
            merged = "\n\n".join([p for p in [prev_content, new_content] if p]).strip()
            prev["content"] = merged
            if current.get("tool_calls"):
                prev.setdefault("tool_calls", [])
                prev["tool_calls"].extend(current["tool_calls"])
        else:
            collapsed.append(current)
    return collapsed


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    items: list[dict[str, Any]] = []
    for tc in tool_calls:
        function = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = function.get("name", "")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass
        items.append({"name": name, "arguments": arguments})
    return json.dumps({"tool_calls": items}, ensure_ascii=False)


def _format_available_tools(tools: list[dict[str, Any]]) -> str:
    sections = [
        "[AVAILABLE_TOOLS]",
        "The following tools are available. Use them only when needed:",
    ]
    for i, tool in enumerate(tools, 1):
        func = tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        params = func.get("parameters", {})
        sections.append(f"{i}. name: {name}")
        sections.append(f"   description: {description}")
        sections.append(
            "   parameters_schema: " + json.dumps(params, ensure_ascii=False, sort_keys=True)
        )

    sections.append(
        'To call tools, respond with JSON: {"tool_call": {"name": "...", "arguments": {...}}} or {"tool_calls": [{"name": "...", "arguments": {...}}]}.'
    )
    return "\n".join(sections)


def flatten_messages_to_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    """Flatten OpenAI-style messages into a single prompt string for text-only backends."""
    collapsed = collapse_consecutive_roles(messages)

    sections: list[str] = []
    tools_block = _format_available_tools(tools) if tools else None
    tools_inserted = False

    for msg in collapsed:
        role = msg.get("role", "")
        content = _normalize_content(msg.get("content"))

        if tools_block and not tools_inserted and role == "user":
            sections.append(tools_block)
            tools_inserted = True

        if role == "system":
            sections.append(f"[SYSTEM]\n{content}".rstrip())
        elif role == "user":
            # Keep user turns clean for backend-visible transcript readability.
            sections.append(content.rstrip())
        elif role == "assistant":
            body_parts: list[str] = []
            if content:
                body_parts.append(content.rstrip())
            if msg.get("tool_calls"):
                body_parts.append(_format_tool_calls(msg["tool_calls"]))
                if not content:
                    # For assistant tool-call handoff messages, keep raw JSON only (no assistant tag)
                    # so downstream text backends don't reinterpret this as a fresh user-style instruction.
                    sections.append(body_parts[-1])
                    continue
            sections.append(f"[ASSISTANT]\n" + "\n".join(body_parts).rstrip())
        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            sections.append(f"[TOOL_RESULT id={tc_id}]\n{content}".rstrip())
        else:
            sections.append(f"[{role.upper() or 'UNKNOWN'}]\n{content}".rstrip())

    if tools_block and not tools_inserted:
        sections.append(tools_block)

    return SEPARATOR.join(section for section in sections if section)
