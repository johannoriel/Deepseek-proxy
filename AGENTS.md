# AGENTS.md — Deepseek-proxy quick architecture guide

This repository provides an **OpenAI-compatible chat/completions proxy** in front of DeepSeek (via `ron.api`).

## Project intent

`proxy_server_v2.py` is the primary compatibility layer with goals:

1. Accept OpenAI-style `/v1/chat/completions` payloads (`messages`, `tools`, `stream`, `session_id`).
2. Flatten chat history into a text prompt DeepSeek can consume.
3. Preserve conversation continuity across requests using backend session/message IDs.
4. Support tool-calling workflows (including multiple tool calls in one assistant turn).
5. Return OpenAI-like JSON responses and SSE streaming chunks.

## Main components

- `proxy_server_v2.py`
  - Flask app + endpoints (`/health`, `/v1/models`, `/v1/chat/completions`).
  - Session orchestration with `SessionManager`.
  - Calls backend through `DeepSeekAPI`.
  - Parses backend text output for tool calls and emits OpenAI-compatible `tool_calls`.
  - Supports streaming both text deltas and tool-call deltas.
  - Includes configurable typing slowdown (`--slowdown`, seconds per 1000 prompt chars).

- `flatten.py`
  - Converts OpenAI message arrays into a single prompt string.
  - Handles system/user/assistant/tool roles.
  - Tool-calls are serialized as compact JSON for backend consistency.
  - Tool results are preserved with `[TOOL_RESULT id=...]` markers.

- `tool_parser.py`
  - Extracts tool calls from backend text (single and multi-call forms).
  - Normalizes tool-call shapes and strips tool JSON from display text.

- `session_manager.py`
  - Maps client `session_id` to backend session/message pointers.
  - Tracks last seen client message count to send only unseen increments.
  - Handles TTL cleanup.

- `test_openai_compat.py`
  - End-to-end compatibility scenarios (phased tests), including multi-tool flows.

## Message-flow summary

1. Client sends OpenAI-style request.
2. Proxy computes incremental unseen messages for this session.
3. `flatten_messages_to_prompt(...)` builds backend prompt text.
4. Backend returns plain text (possibly containing tool-call JSON).
5. Parser extracts tool calls, strips leaked tool JSON from text, and proxy returns:
   - regular assistant text (`finish_reason=stop`), or
   - assistant `tool_calls` (`finish_reason=tool_calls`), optionally with residual natural-language content.
6. Client executes tools and sends `role="tool"` messages.
7. Proxy forwards unseen turns and continues same backend conversation.

## Practical notes for future agents

- Prefer changing `proxy_server_v2.py` over legacy `proxy-server.py` unless explicitly requested.
- Keep OpenAI response shape stable (`choices[0].message`, `finish_reason`, `usage`, `session_id`).
- Be careful when modifying flattening/parsing: regressions usually appear as duplicated tool JSON, lost tool IDs, or broken follow-up memory.
- If adjusting session logic, ensure multi-message increments (assistant tool_calls + multiple tool results) remain in order.
