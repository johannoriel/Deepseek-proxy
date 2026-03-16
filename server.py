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
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


def extract_tools_from_messages(messages):
    """Extract tool definitions from messages if present."""
    tools = None
    for msg in messages:
        if msg.get("role") == "system" and "tools" in msg:
            tools = msg.get("tools")
            break
    return tools


def get_session_for_messages(messages):
    """
    Given the full messages array from the client:
    - Extract the conversation (user/assistant turns only)
    - Hash everything EXCEPT the final user message
    - If that hash is cached → reuse the session (no replay needed)
    - Otherwise → create a new session and replay history

    Returns (session_id, deepseek_session_id, last_message_id, final_user_message, tools)
    """
    conversation = [
        {"role": msg["role"], "content": msg.get("content", "")}
        for msg in messages
        if msg.get("role") in ("user", "assistant", "system", "tool")
    ]

    # Extract tools if present
    tools = None
    for msg in messages:
        if msg.get("role") == "system" and "tools" in msg:
            tools = msg.get("tools")
            break

    dbg(f"get_session: {len(conversation)} turns in messages[]")
    for i, m in enumerate(conversation):
        dbg(f"  [{i}] {m['role']}: {repr(m['content'][:80] if m.get('content') else '')}")

    # Handle tool message responses - the last message could be from tool
    # We need to find the last user message and include everything before it in history
    last_user_index = None
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i]["role"] == "user":
            last_user_index = i
            break

    if last_user_index is None:
        dbg("get_session: ERROR - no user message found")
        return None, None, None, None, None

    final_user_message = conversation[last_user_index]["content"]
    history = conversation[:last_user_index]  # everything before the last user message

    h = history_hash(history)
    dbg(f"get_session: history_hash={h[:16]}... (history_len={len(history)})")

    # Only consult the cache when there is prior history.
    if history:
        with cache_lock:
            if h in session_cache:
                cached = session_cache[h]
                dbg(f"get_session: CACHE HIT → reusing session {cached['session_id']}, "
                    f"last_message_id={cached['last_message_id']}, turns={cached['turn_count']}")
                return cached["session_id"], cached["deepseek_session_id"], cached["last_message_id"], final_user_message, tools
    else:
        dbg("get_session: empty history → forcing new DeepSeek session (no cache lookup)")

    # Cache miss — create new session and replay history
    dbg(f"get_session: CACHE MISS → creating new DeepSeek session and replaying {len(history)} turns")
    deepseek_session_id = deepseek_api.create_chat_session()
    dbg(f"get_session: DeepSeek session created: {deepseek_session_id}")

    last_message_id = None
    i = 0
    while i < len(history):
        role = history[i]["role"]
        content = history[i].get("content", "")

        if role == "user":
            dbg(f"get_session: replaying turn [{i}] user, parent_message_id={last_message_id}")
            try:
                # Check if this user message should include tool definitions
                user_tools = None
                if i == 0 and tools:  # First message might have tools
                    user_tools = tools

                response_data = deepseek_api.chat_completion(
                    deepseek_session_id,
                    content,
                    parent_message_id=last_message_id,
                    tools=user_tools
                )

                if isinstance(response_data, tuple):
                    text, last_message_id = response_data
                else:
                    text = response_data.get('content', '')
                    last_message_id = response_data.get('message_id')

                dbg(f"get_session: replay [{i}] done, message_id={last_message_id}, reply_len={len(text)}")
            except Exception as e:
                dbg(f"get_session: EXCEPTION replaying turn [{i}]: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                return None, None, None, None, None

            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                dbg(f"get_session: skipping assistant turn [{i+1}]")
                i += 2
                continue
            elif i + 1 < len(history) and history[i + 1]["role"] == "tool":
                dbg(f"get_session: tool response follows, will handle in next iteration")
                i += 1
                continue
        elif role == "tool":
            # Handle tool messages - they should be sent as part of the next user message
            # For now, just skip them as they'll be handled by the next user message
            dbg(f"get_session: skipping tool message at [{i}]")
            i += 1
            continue
        else:
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
    return session_id, deepseek_session_id, last_message_id, final_user_message, tools


def update_cache_after_reply(history_before_final, final_user_message, assistant_response, new_message_id):
    """
    After a successful reply, cache the NEW history state so the next turn is a cache hit.
    New history = old history + final user message + assistant reply.
    """
    # Handle both string responses and tool call responses
    if isinstance(assistant_response, dict) and 'tool_calls' in assistant_response:
        # Response with tool calls
        new_history = history_before_final + [
            {"role": "user", "content": final_user_message},
            {"role": "assistant", "content": assistant_response.get('content', ''), "tool_calls": assistant_response['tool_calls']},
        ]
    else:
        # Simple text response
        new_history = history_before_final + [
            {"role": "user", "content": final_user_message},
            {"role": "assistant", "content": assistant_response},
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
        tools = data.get('tools', None)  # Get tools from request
        tool_choice = data.get('tool_choice', 'auto')

        dbg(f"--- NEW REQUEST: {len(messages)} messages, stream={stream}, tools={tools is not None} ---")

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        # If tools are provided in the request, add them to the system message
        if tools:
            # Find or create system message with tools
            system_msg_with_tools = None
            for msg in messages:
                if msg.get("role") == "system":
                    system_msg_with_tools = msg
                    break

            if system_msg_with_tools:
                # Add tools to the system message
                system_msg_with_tools["tools"] = tools

        session_id, deepseek_session_id, last_message_id, user_message, extracted_tools = get_session_for_messages(messages)

        # Use tools from request if available, otherwise use extracted ones
        active_tools = tools or extracted_tools

        if not deepseek_session_id or not user_message:
            return jsonify({"error": "Invalid message structure"}), 400

        # Compute history_before_final for cache update after reply
        conversation = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in ("user", "assistant", "tool")
        ]

        # Find the last user message index
        last_user_index = None
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i]["role"] == "user":
                last_user_index = i
                break

        if last_user_index is not None:
            history_before_final = conversation[:last_user_index]
        else:
            history_before_final = []

        dbg(f"chat_completions: sending final message, parent_message_id={last_message_id}, stream={stream}")

        if stream:
            return handle_streaming_response(deepseek_session_id, user_message, last_message_id, history_before_final, active_tools, tool_choice)
        else:
            return handle_normal_response(deepseek_session_id, user_message, last_message_id, history_before_final, active_tools, tool_choice)

    except Exception as e:
        dbg(f"chat_completions: TOP-LEVEL EXCEPTION: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500


def handle_normal_response(deepseek_session_id, user_message, parent_message_id, history_before_final, tools=None, tool_choice='auto'):
    try:
        # Pass tools to the DeepSeek API - now using native support
        response_data = deepseek_api.chat_completion(
            deepseek_session_id,
            user_message,
            parent_message_id=parent_message_id,
            tools=tools,
            tool_choice=tool_choice
        )

        # Handle different response formats
        if isinstance(response_data, tuple):
            response_text, message_id = response_data
            assistant_message = {"role": "assistant", "content": response_text}
        else:
            # Response with tool calls from native API
            message_id = response_data.get('message_id')
            assistant_message = {
                "role": "assistant",
                "content": response_data.get('content', '')
            }

            # Add tool calls if present - already in OpenAI format
            if 'tool_calls' in response_data:
                assistant_message['tool_calls'] = response_data['tool_calls']
                dbg(f"handle_normal: Native tool calls received: {json.dumps(response_data['tool_calls'], indent=2)}")

        # Show only beginning of response in debug
        content = assistant_message.get('content', '')
        excerpt = content[:150] + "..." if len(content) > 150 else content
        dbg(f"handle_normal: response len={len(content)}, message_id={message_id}")
        if 'tool_calls' in assistant_message:
            dbg(f"handle_normal: contains {len(assistant_message['tool_calls'])} tool calls")
        dbg(f"handle_normal: RESPONSE EXCERPT:\n{excerpt}\n--- END EXCERPT ---")

        update_cache_after_reply(history_before_final, user_message, assistant_message, message_id)

        response_payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": "tool_calls" if 'tool_calls' in assistant_message else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": (len(user_message) + len(content)) // 4
            }
        }

        return jsonify(response_payload)

    except Exception as e:
        dbg(f"handle_normal: EXCEPTION: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": f"DeepSeek API error: {str(e)}"}), 500


def handle_streaming_response(deepseek_session_id, user_message, parent_message_id, history_before_final, tools=None, tool_choice='auto'):
    from flask import Response, stream_with_context

    def generate():
        try:
            # Call DeepSeek API with native tool support
            response_data = deepseek_api.chat_completion(
                deepseek_session_id,
                user_message,
                parent_message_id=parent_message_id,
                tools=tools,
                tool_choice=tool_choice
            )

            # For streaming, we need to handle the response differently
            # The API might return the complete response even in stream mode
            if isinstance(response_data, tuple):
                response_text, message_id = response_data
                has_tool_calls = False
                tool_calls = None
            else:
                response_text = response_data.get('content', '')
                message_id = response_data.get('message_id')
                tool_calls = response_data.get('tool_calls')
                has_tool_calls = tool_calls is not None

            # Show only beginning of response in debug
            excerpt = response_text[:150] + "..." if len(response_text) > 150 else response_text
            dbg(f"handle_streaming: response len={len(response_text)}, message_id={message_id}")
            if has_tool_calls:
                dbg(f"handle_streaming: contains {len(tool_calls)} tool calls")
                dbg(f"handle_streaming: Tool calls: {json.dumps(tool_calls, indent=2)}")
            dbg(f"handle_streaming: RESPONSE EXCERPT:\n{excerpt}\n--- END EXCERPT ---")

            assistant_message = {
                "role": "assistant",
                "content": response_text
            }
            if has_tool_calls:
                assistant_message['tool_calls'] = tool_calls

            update_cache_after_reply(history_before_final, user_message, assistant_message, message_id)

            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            chunks_sent = 0

            if has_tool_calls:
                # For tool calls in streaming mode, we need to send the tool calls in chunks
                # according to the OpenAI streaming format

                # First, send a chunk with just the role
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'deepseek-chat', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                chunks_sent += 1

                # Then send each tool call in separate chunks
                for tool_call in tool_calls:
                    # Send the tool call index and ID
                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'deepseek-chat', 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': tool_call['id'], 'type': 'function'}]}, 'finish_reason': None}]})}\n\n"
                    chunks_sent += 1

                    # Send the function name
                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'deepseek-chat', 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'name': tool_call['function']['name']}}]}, 'finish_reason': None}]})}\n\n"
                    chunks_sent += 1

                    # Send the arguments in chunks if they're long
                    arguments = tool_call['function']['arguments']
                    chunk_size = 20
                    for i in range(0, len(arguments), chunk_size):
                        arg_chunk = arguments[i:i + chunk_size]
                        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'deepseek-chat', 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': arg_chunk}}]}, 'finish_reason': None}]})}\n\n"
                        chunks_sent += 1
                        time.sleep(0.01)  # Small delay for realism
            else:
                # Stream text content in chunks
                for i in range(0, len(response_text), 20):
                    chunk = response_text[i:i + 20]
                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'deepseek-chat', 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                    chunks_sent += 1
                    time.sleep(0.05)

            # Send finish chunk with appropriate finish_reason
            finish_reason = "tool_calls" if has_tool_calls else "stop"
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'deepseek-chat', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
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
    parser.add_argument('--api-key', type=str, help='DeepSeek auth token (can also be set via DEEPSEEK_TOKEN env var or .env file)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    # Get API key from command line args, environment variable, or .env file
    api_key = args.api_key or os.getenv('DEEPSEEK_TOKEN')

    if not api_key:
        print("ERROR: No API key provided. Please provide via:")
        print("  - --api-key command line argument")
        print("  - DEEPSEEK_TOKEN environment variable")
        print("  - DEEPSEEK_TOKEN in .env file")
        sys.exit(1)

    global deepseek_api
    deepseek_api = DeepSeekAPI(api_key)

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
