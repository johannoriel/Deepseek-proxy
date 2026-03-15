from flask import Flask, request, jsonify
from flask_cors import CORS
from ron.api import DeepSeekAPI
import time
import argparse
import uuid
import sys
import json
import hashlib
from threading import Lock

app = Flask(__name__)
CORS(app)

deepseek_api = None

# Cache keyed by history_hash (hash of all turns EXCEPT the last user message).
# Each entry: { "session_id": str, "last_message_id": int|None, "turn_count": int }
session_cache = {}
cache_lock = Lock()


def dbg(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


def history_hash(conversation_without_last):
    """Stable hash of the conversation history (all turns before the final user message)."""
    key = json.dumps(conversation_without_last, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(key.encode()).hexdigest()


@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek",
                "permission": []
            }
        ]
    })


def get_session_for_messages(messages):
    """
    Given the full messages array from the client:
    - Extract the conversation (user/assistant turns only)
    - Hash everything EXCEPT the final user message
    - If that hash is cached → reuse the session (no replay needed)
    - Otherwise → create a new session and replay history

    Returns (session_id, deepseek_session_id, last_message_id, final_user_message)
    """
    conversation = [
        {"role": msg["role"], "content": msg.get("content", "")}
        for msg in messages
        if msg.get("role") in ("user", "assistant")
    ]

    dbg(f"get_session: {len(conversation)} turns in messages[]")
    for i, m in enumerate(conversation):
        dbg(f"  [{i}] {m['role']}: {repr(m['content'][:80])}")

    if not conversation or conversation[-1]["role"] != "user":
        dbg("get_session: ERROR - last turn is not user")
        return None, None, None, None

    final_user_message = conversation[-1]["content"]
    history = conversation[:-1]  # everything before the final user message

    h = history_hash(history)
    dbg(f"get_session: history_hash={h[:16]}... (history_len={len(history)})")

    with cache_lock:
        if h in session_cache:
            cached = session_cache[h]
            dbg(f"get_session: CACHE HIT → reusing session {cached['session_id']}, "
                f"last_message_id={cached['last_message_id']}, turns={cached['turn_count']}")
            return cached["session_id"], cached["deepseek_session_id"], cached["last_message_id"], final_user_message

    # Cache miss — create new session and replay history
    dbg(f"get_session: CACHE MISS → creating new DeepSeek session and replaying {len(history)} turns")
    deepseek_session_id = deepseek_api.create_chat_session()
    dbg(f"get_session: DeepSeek session created: {deepseek_session_id}")

    last_message_id = None
    i = 0
    while i < len(history):
        role = history[i]["role"]
        content = history[i]["content"]
        if role == "user":
            dbg(f"get_session: replaying turn [{i}] user, parent_message_id={last_message_id}")
            try:
                text, last_message_id = deepseek_api.chat_completion(
                    deepseek_session_id,
                    content,
                    parent_message_id=last_message_id
                )
                dbg(f"get_session: replay [{i}] done, message_id={last_message_id}, reply_len={len(text)}")
            except Exception as e:
                dbg(f"get_session: EXCEPTION replaying turn [{i}]: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                return None, None, None, None

            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                dbg(f"get_session: skipping assistant turn [{i+1}]")
                i += 2
                continue
        i += 1

    session_id = str(uuid.uuid4())
    entry = {
        "session_id": session_id,
        "deepseek_session_id": deepseek_session_id,
        "last_message_id": last_message_id,
        "turn_count": len([m for m in history if m["role"] == "user"]),
    }
    with cache_lock:
        session_cache[h] = entry

    dbg(f"get_session: ready. session={session_id}, last_message_id={last_message_id}")
    return session_id, deepseek_session_id, last_message_id, final_user_message


def update_cache_after_reply(history_before_final, final_user_message, assistant_reply, new_message_id):
    """
    After a successful reply, cache the NEW history state so the next turn is a cache hit.
    New history = old history + final user message + assistant reply.
    """
    new_history = history_before_final + [
        {"role": "user", "content": final_user_message},
        {"role": "assistant", "content": assistant_reply},
    ]
    new_hash = history_hash(new_history)

    # Find the current session entry by old hash to copy session IDs
    old_hash = history_hash(history_before_final)
    with cache_lock:
        old_entry = session_cache.get(old_hash)
        if old_entry:
            new_entry = {
                "session_id": old_entry["session_id"],
                "deepseek_session_id": old_entry["deepseek_session_id"],
                "last_message_id": new_message_id,
                "turn_count": old_entry["turn_count"] + 1,
            }
            session_cache[new_hash] = new_entry
            dbg(f"update_cache: stored new hash {new_hash[:16]}... with message_id={new_message_id}")


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        messages = data.get('messages', [])
        stream = data.get('stream', False)

        dbg(f"--- NEW REQUEST: {len(messages)} messages, stream={stream} ---")

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        session_id, deepseek_session_id, last_message_id, user_message = get_session_for_messages(messages)

        if not deepseek_session_id or not user_message:
            return jsonify({"error": "Invalid message structure"}), 400

        # Compute history_before_final for cache update after reply
        conversation = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        history_before_final = conversation[:-1]

        dbg(f"chat_completions: sending final message, parent_message_id={last_message_id}, stream={stream}")

        if stream:
            return handle_streaming_response(deepseek_session_id, user_message, last_message_id, history_before_final)
        else:
            return handle_normal_response(deepseek_session_id, user_message, last_message_id, history_before_final)

    except Exception as e:
        dbg(f"chat_completions: TOP-LEVEL EXCEPTION: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500


def handle_normal_response(deepseek_session_id, user_message, parent_message_id, history_before_final):
    try:
        response, message_id = deepseek_api.chat_completion(
            deepseek_session_id, user_message, parent_message_id=parent_message_id
        )
        dbg(f"handle_normal: response len={len(response)}, message_id={message_id}")
        dbg(f"handle_normal: FULL RESPONSE:\n{response}\n--- END ---")

        update_cache_after_reply(history_before_final, user_message, response, message_id)

        return jsonify({
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message) // 4,
                "completion_tokens": len(response) // 4,
                "total_tokens": (len(user_message) + len(response)) // 4
            }
        })
    except Exception as e:
        dbg(f"handle_normal: EXCEPTION: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": f"DeepSeek API error: {str(e)}"}), 500


def handle_streaming_response(deepseek_session_id, user_message, parent_message_id, history_before_final):
    from flask import Response, stream_with_context

    def generate():
        try:
            response, message_id = deepseek_api.chat_completion(
                deepseek_session_id, user_message, parent_message_id=parent_message_id
            )
            dbg(f"handle_streaming: response len={len(response)}, message_id={message_id}")
            dbg(f"handle_streaming: FULL RESPONSE:\n{response}\n--- END ---")

            update_cache_after_reply(history_before_final, user_message, response, message_id)

            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            chunks_sent = 0

            for i in range(0, len(response), 20):
                chunk = response[i:i + 20]
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'deepseek-chat', 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                chunks_sent += 1
                time.sleep(0.05)

            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'deepseek-chat', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            dbg(f"handle_streaming: done, {chunks_sent} chunks sent")

        except Exception as e:
            dbg(f"handle_streaming: EXCEPTION: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data.get('prompt', '')
    session_id = deepseek_api.create_chat_session()
    response, _ = deepseek_api.chat_completion(session_id, prompt)
    return jsonify({
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [{"text": response, "index": 0, "logprobs": None, "finish_reason": "stop"}]
    })


@app.route('/health', methods=['GET'])
def health_check():
    with cache_lock:
        count = len(session_cache)
    return jsonify({"status": "healthy", "cached_sessions": count})


def main():
    parser = argparse.ArgumentParser(description='Run DeepSeek OpenAI-compatible server')
    parser.add_argument('--api-key', type=str, required=True, help='DeepSeek auth token')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    global deepseek_api
    deepseek_api = DeepSeekAPI(args.api_key)

    print(f"Starting DeepSeek OpenAI-compatible server on http://{args.host}:{args.port}")
    print("Endpoints:")
    print(f"  GET  /v1/models")
    print(f"  POST /v1/chat/completions")
    print(f"  POST /v1/completions")
    print(f"  GET  /health")
    print("\nPress Ctrl+C to stop")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
