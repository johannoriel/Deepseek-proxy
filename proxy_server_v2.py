import argparse
import json
import logging
import os
import time
import uuid
from typing import Any, Iterator

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

from flatten import flatten_messages_to_prompt
from ron.api import DeepSeekAPI
from session_manager import SessionManager
from tool_parser import clean_text_response, extract_tool_call


logger = logging.getLogger("proxy_server_v2")
slowdown_seconds = 0.0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def build_tool_call_response(tool_call_payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = tool_call_payload.get("tool_call", {})
    name = data.get("name", "")
    arguments = data.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {"raw": arguments}
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
    ]


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
        message["content"] = None
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


def stream_tool_response(model: str, response_id: str, tool_call: dict[str, Any]) -> Iterator[str]:
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
                                "index": 0,
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
                                    "index": 0,
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


def create_app(api_key: str, debug: bool = False) -> Flask:
    app = Flask(__name__)
    CORS(app)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
            messages = payload.get("messages", [])
            tools = payload.get("tools")
            stream = bool(payload.get("stream", False))
            model = payload.get("model", "deepseek-chat")
            client_session_id = payload.get("session_id")

            if not isinstance(messages, list) or not messages:
                return jsonify({"error": {"message": "messages must be a non-empty list", "type": "invalid_request_error", "code": 400}}), 400

            backend_session_id, parent_message_id = session_manager.get_or_create(
                client_session_id, len(messages)
            )
            if not client_session_id:
                client_session_id = session_manager.new_client_session_id()

            prompt_text = flatten_messages_to_prompt(messages, tools)
            logger.info(
                "ron_api.chat_completion session_id=%s parent_message_id=%s prompt_len=%s",
                backend_session_id,
                parent_message_id,
                len(prompt_text),
            )
            ron_response = ron_api.chat_completion(
                backend_session_id,
                prompt_text,
                parent_message_id=parent_message_id,
            )
            if slowdown_seconds > 0:
                logger.info("Applying slowdown of %ss after API call", slowdown_seconds)
                time.sleep(slowdown_seconds)

            if isinstance(ron_response, tuple):
                response_text, new_message_id = ron_response
            else:
                response_text = ron_response.get("content", "")
                new_message_id = ron_response.get("message_id")

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

            session_manager.update(client_session_id, backend_session_id, new_message_id)

            parsed_tool = extract_tool_call(response_text or "")
            response_id = f"chatcmpl-{uuid.uuid4().hex}"

            if parsed_tool:
                tool_calls = build_tool_call_response(parsed_tool)
                if stream:
                    return Response(
                        stream_tool_response(model, response_id, tool_calls[0]),
                        mimetype="text/event-stream",
                    )
                return jsonify(
                    build_completion_response(
                        model=model,
                        session_id=client_session_id,
                        content=None,
                        tool_calls=tool_calls,
                        finish_reason="tool_calls",
                        prompt_text=prompt_text,
                    )
                )

            cleaned = clean_text_response(response_text or "")
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
    parser.add_argument(
        "--slowdown",
        type=float,
        default=0.0,
        help="Wait this many seconds after each API call to reduce rate-limit pressure",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("DEEPSEEK_TOKEN")
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set DEEPSEEK_TOKEN")

    global slowdown_seconds
    slowdown_seconds = args.slowdown

    app = create_app(api_key=api_key, debug=args.debug)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
