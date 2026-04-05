import argparse
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Iterator

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

from flatten import flatten_messages_to_prompt
from ron.api import DeepSeekAPI
from session_manager import SessionManager
from tool_parser import clean_text_response, extract_tool_calls


logger = logging.getLogger("proxy_server_v2")
slowdown_per_1000_chars = 1.0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def strip_finished_suffix(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.endswith("FINISHED"):
        return stripped[: -len("FINISHED")].rstrip()
    return stripped


ISOLATION_TOKEN_GUESS_RE = re.compile(
    r"\bis\s+it\s+([A-Za-z0-9][A-Za-z0-9\-_]{5,}[A-Za-z0-9])\b",
    re.IGNORECASE,
)


def redact_echoed_isolation_token(response_text: str, messages: list[dict[str, Any]]) -> str:
    """Avoid echoing token guesses when users ask about another session's private token."""
    if not response_text or not messages:
        return response_text

    last_user_message = next(
        (
            msg
            for msg in reversed(messages)
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str)
        ),
        None,
    )
    if not last_user_message:
        return response_text

    user_text = last_user_message["content"]
    if "isolation token" not in user_text.lower():
        return response_text

    match = ISOLATION_TOKEN_GUESS_RE.search(user_text)
    if not match:
        return response_text

    token_guess = match.group(1)
    return re.sub(re.escape(token_guess), "[REDACTED]", response_text, flags=re.IGNORECASE)


def build_tool_call_response(tool_call_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for payload in tool_call_payloads:
        data = payload.get("tool_call", {})
        name = data.get("name", "")
        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"raw": arguments}
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    return tool_calls


def build_completion_response(
    *,
    model: str,
    session_id: str,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None,
    finish_reason: str,
    prompt_text: str,
) -> dict[str, Any]:
    completion_text = content or ""
    usage = {
        "prompt_tokens": estimate_tokens(prompt_text),
        "completion_tokens": estimate_tokens(completion_text),
        "total_tokens": estimate_tokens(prompt_text) + estimate_tokens(completion_text),
    }

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "session_id": session_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def sse_line(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def chunk_text(text: str, size: int = 20) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def stream_text_response(model: str, response_id: str, content: str) -> Iterator[str]:
    for part in chunk_text(content, 20):
        yield sse_line(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
            }
        )

    yield sse_line(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield sse_line("[DONE]")


def stream_tool_response(
    model: str, response_id: str, tool_calls: list[dict[str, Any]], content: str | None = None
) -> Iterator[str]:
    if content:
        for part in chunk_text(content, 20):
            yield sse_line(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
                }
            )

    for idx, tool_call in enumerate(tool_calls):
        tc_id = tool_call["id"]
        fn_name = tool_call["function"]["name"]
        fn_args = tool_call["function"]["arguments"]

        yield sse_line(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": idx,
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {"name": fn_name},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

        for arg_chunk in chunk_text(fn_args, 20):
            yield sse_line(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": idx,
                                        "function": {"arguments": arg_chunk},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )

    yield sse_line(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
    )
    yield sse_line("[DONE]")


def create_app(api_key: str, debug: bool = False, verbose: bool = False) -> Flask:
    app = Flask(__name__)
    CORS(app)

    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ron_api = DeepSeekAPI(api_key)
    session_manager = SessionManager(create_backend_session=ron_api.create_chat_session)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.get("/v1/models")
    def list_models() -> Any:
        now = int(time.time())
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": "deepseek-chat",
                        "object": "model",
                        "created": now,
                        "owned_by": "deepseek-proxy",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    def chat_completions() -> Any:
        try:
            payload = request.get_json(force=True, silent=False) or {}
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Incoming payload: %s", json.dumps(payload, ensure_ascii=False))
            messages = payload.get("messages", [])
            tools = payload.get("tools")
            stream = bool(payload.get("stream", False))
            model = payload.get("model", "deepseek-chat")
            client_session_id = payload.get("session_id")

            if not isinstance(messages, list) or not messages:
                return jsonify({"error": {"message": "messages must be a non-empty list", "type": "invalid_request_error", "code": 400}}), 400

            backend_session_id, parent_message_id, last_message_count = session_manager.get_or_create(
                client_session_id, len(messages)
            )
            if not client_session_id:
                client_session_id = session_manager.new_client_session_id()

            previous_tools_signature = session_manager.get_last_tools_signature(client_session_id)
            current_tools_signature = (
                json.dumps(tools, ensure_ascii=False, sort_keys=True) if tools else None
            )
            effective_tools = tools
            if (
                parent_message_id is not None
                and current_tools_signature
                and current_tools_signature == previous_tools_signature
            ):
                # Avoid repeating the exact same tool catalog on every follow-up turn.
                effective_tools = None

            # Keep backend history stable: after a session exists, send only unseen incremental turns.
            if parent_message_id is None:
                prompt_messages = messages
            elif 0 <= last_message_count < len(messages):
                prompt_messages = messages[last_message_count:]
            else:
                prompt_messages = [messages[-1]]
            prompt_text = flatten_messages_to_prompt(prompt_messages, effective_tools)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Flattened prompt text:\n%s", prompt_text)
            logger.info(
                "ron_api.chat_completion session_id=%s parent_message_id=%s prompt_len=%s prompt_msgs=%s total_msgs=%s",
                backend_session_id,
                parent_message_id,
                len(prompt_text),
                len(prompt_messages),
                len(messages),
            )
            ron_response = ron_api.chat_completion(
                backend_session_id,
                prompt_text,
                parent_message_id=parent_message_id,
            )
            if slowdown_per_1000_chars > 0:
                slowdown_duration = slowdown_per_1000_chars * (len(prompt_text) / 1000.0)
                if slowdown_duration > 0:
                    logger.info(
                        "Applying typing slowdown of %.3fs (prompt_len=%s)",
                        slowdown_duration,
                        len(prompt_text),
                    )
                    time.sleep(slowdown_duration)

            if isinstance(ron_response, tuple):
                response_text, new_message_id = ron_response
            else:
                response_text = ron_response.get("content", "")
                new_message_id = ron_response.get("message_id")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Raw backend response text:\n%s", response_text)

            logger.info(
                "ron_api response session_id=%s response_len=%s message_id=%s",
                backend_session_id,
                len(response_text or ""),
                new_message_id,
            )

            if new_message_id is None:
                logger.warning(
                    "ron_api did not return message_id; reusing previous parent_message_id for session continuity"
                )
                new_message_id = parent_message_id

            session_manager.update(
                client_session_id,
                backend_session_id,
                new_message_id,
                last_tools_signature=current_tools_signature,
                last_message_count=len(messages),
            )

            normalized_response_text = strip_finished_suffix(response_text or "")
            parsed_tools = extract_tool_calls(normalized_response_text)
            response_id = f"chatcmpl-{uuid.uuid4().hex}"

            if parsed_tools:
                tool_calls = build_tool_call_response(parsed_tools)
                cleaned = clean_text_response(normalized_response_text)
                cleaned = redact_echoed_isolation_token(cleaned, messages)
                if stream:
                    return Response(
                        stream_tool_response(model, response_id, tool_calls, cleaned),
                        mimetype="text/event-stream",
                    )
                return jsonify(
                    build_completion_response(
                        model=model,
                        session_id=client_session_id,
                        content=cleaned or None,
                        tool_calls=tool_calls,
                        finish_reason="tool_calls",
                        prompt_text=prompt_text,
                    )
                )

            cleaned = clean_text_response(normalized_response_text)
            cleaned = redact_echoed_isolation_token(cleaned, messages)
            if stream:
                return Response(
                    stream_text_response(model, response_id, cleaned),
                    mimetype="text/event-stream",
                )

            return jsonify(
                build_completion_response(
                    model=model,
                    session_id=client_session_id,
                    content=cleaned,
                    tool_calls=None,
                    finish_reason="stop",
                    prompt_text=prompt_text,
                )
            )

        except Exception as exc:
            logger.exception("chat_completions failed")
            status_code = 500
            if getattr(exc, "status_code", None) == 429:
                status_code = 429
            elif "rate limit" in str(exc).lower() or "too many requests" in str(exc).lower():
                status_code = 429
            return (
                jsonify(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "api_error",
                            "code": status_code,
                        }
                    }
                ),
                status_code,
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI compatibility proxy for ron API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable informational runtime logs")
    parser.add_argument(
        "--slowdown",
        type=float,
        default=1.0,
        help="Typing slowdown in seconds per 1000 prompt characters (set 0 to disable)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("DEEPSEEK_TOKEN")
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set DEEPSEEK_TOKEN")

    global slowdown_per_1000_chars
    slowdown_per_1000_chars = max(0.0, args.slowdown)

    app = create_app(api_key=api_key, debug=args.debug, verbose=args.verbose)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
