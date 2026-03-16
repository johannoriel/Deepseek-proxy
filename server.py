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
from typing import Dict, Any, List, Optional, Tuple
from tool_plugin import create_tool_plugin, ToolPlugin, SimulatedToolPlugin, NativeToolPlugin


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

deepseek_api = None
tool_plugin: ToolPlugin = None
tool_placement: str = 'user'


# Cache keyed by history_hash (hash of all turns EXCEPT the last user message).
# Each entry: { "session_id": str, "last_message_id": int|None, "turn_count": int }
session_cache = {}
cache_lock = Lock()


def dbg(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

def init_tool_plugin(plugin_type: str = "simulated", enabled: bool = False, placement: str = 'user'):
    """Initialize the tool plugin."""
    global tool_plugin, tool_placement
    tool_placement = placement
    tool_plugin = create_tool_plugin(plugin_type, enabled, placement)
    dbg(f"Tool plugin initialized: {plugin_type if enabled else 'disabled'} with placement={placement}")

def process_with_plugin(messages: List[Dict], tools: Optional[List[Dict]] = None) -> List[Dict]:
    """Process messages through the plugin before sending to DeepSeek."""
    if not tool_plugin or not tool_plugin.enabled or not tools:
        return messages
    return tool_plugin.prepare_messages(messages, tools)

def handle_tool_response_in_history(messages: List[Dict], tool_call_id: str, tool_name: str, tool_response: str) -> List[Dict]:
    """Convert a tool response message to the format expected by the plugin."""
    if not tool_plugin or not tool_plugin.enabled:
        return messages

    # Find the tool response message and convert it
    converted_messages = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
            # Convert tool message using plugin
            converted = tool_plugin.prepare_tool_response(
                tool_call_id,
                tool_name,
                msg.get("content", ""),
                messages
            )
            converted_messages.append(converted)
        else:
            converted_messages.append(msg)

    return converted_messages

def extract_tool_calls_from_response(response_text: str, original_messages: List[Dict]) -> Tuple[str, Optional[List[Dict]]]:
    """Extract tool calls from response text using the plugin."""
    if not tool_plugin or not tool_plugin.enabled:
        return response_text, None

    # Check if this is a simulated plugin that needs to extract tool calls
    if isinstance(tool_plugin, SimulatedToolPlugin):
        return tool_plugin.process_response(response_text, original_messages)

    return response_text, None

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
    # First, process messages through plugin if needed
    processed_messages = messages

    # Handle tool responses if we're in a tool conversation
    if tool_plugin and tool_plugin.enabled:
        # Check if we have tool messages that need conversion
        has_tool_msgs = any(msg.get("role") == "tool" for msg in messages)
        if has_tool_msgs and isinstance(tool_plugin, SimulatedToolPlugin):
            # Convert tool messages
            converted = []
            for i, msg in enumerate(messages):
                if msg.get("role") == "tool":
                    # Find the corresponding assistant message with tool_calls
                    for j in range(i-1, -1, -1):
                        if messages[j].get("role") == "assistant" and messages[j].get("tool_calls"):
                            tool_call = messages[j]["tool_calls"][0]  # Assume first tool call for now
                            converted_msg = tool_plugin.prepare_tool_response(
                                tool_call["id"],
                                tool_call["function"]["name"],
                                msg.get("content", ""),
                                messages
                            )
                            converted.append(converted_msg)
                            break
                else:
                    converted.append(msg)
            processed_messages = converted

    conversation = [
        {"role": msg["role"], "content": msg.get("content", "")}
        for msg in processed_messages
        if msg.get("role") in ("user", "assistant", "system", "tool")
    ]

    # Extract tools if present
    tools = None
    for msg in messages:  # Use original messages to check for tools
        if msg.get("role") == "system" and "tools" in msg:
            tools = msg.get("tools")
            break
    if not tools:
        for msg in messages:
            if "tools" in msg:
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

    # Create a new session ID for our cache
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
        tools = data.get('tools', None)
        tool_choice = data.get('tool_choice', 'auto')

        dbg(f"--- NEW REQUEST: {len(messages)} messages, stream={stream}, tools={tools is not None}, plugin_enabled={tool_plugin and tool_plugin.enabled} ---")

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        # Store original messages for later reference
        original_messages = messages.copy()

        # Process messages through plugin if tools are present
        if tools and tool_plugin and tool_plugin.enabled:
            # For simulated plugin, we need to add tool definitions to system message
            processed_messages = tool_plugin.prepare_messages(messages, tools)
            dbg(f"Messages processed by plugin: {len(processed_messages)} messages")

            # Update messages for the rest of the function
            messages = processed_messages

        session_id, deepseek_session_id, last_message_id, user_message, extracted_tools = get_session_for_messages(messages)

        # Use tools from request if available
        active_tools = tools if (tools and tool_plugin and isinstance(tool_plugin, NativeToolPlugin)) else None

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
            return handle_streaming_response(deepseek_session_id, user_message, last_message_id, history_before_final, active_tools, tool_choice, original_messages)
        else:
            return handle_normal_response(deepseek_session_id, user_message, last_message_id, history_before_final, active_tools, tool_choice, original_messages)

    except Exception as e:
        dbg(f"chat_completions: TOP-LEVEL EXCEPTION: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500


def handle_normal_response(deepseek_session_id, user_message, parent_message_id, history_before_final, tools=None, tool_choice='auto', original_messages=None):
    try:
        # Call DeepSeek API
        response_data = deepseek_api.chat_completion(
            deepseek_session_id,
            user_message,
            parent_message_id=parent_message_id,
            tools=tools,  # Only used if native plugin is enabled
            tool_choice=tool_choice
        )

        # Handle different response formats
        if isinstance(response_data, tuple):
            response_text, message_id = response_data

            # Process response through plugin to extract tool calls
            if tool_plugin and tool_plugin.enabled and original_messages:
                processed_text, tool_calls = extract_tool_calls_from_response(response_text, original_messages)
                response_text = processed_text
            else:
                tool_calls = None

            assistant_message = {"role": "assistant", "content": response_text}
            if tool_calls:
                assistant_message['tool_calls'] = tool_calls
                dbg(f"handle_normal: Extracted {len(tool_calls)} tool calls via plugin")
        else:
            # Native tool calls from API
            message_id = response_data.get('message_id')
            assistant_message = {
                "role": "assistant",
                "content": response_data.get('content', '')
            }
            if 'tool_calls' in response_data:
                assistant_message['tool_calls'] = response_data['tool_calls']
                dbg(f"handle_normal: Native tool calls received: {len(response_data['tool_calls'])}")

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


def handle_streaming_response(deepseek_session_id, user_message, parent_message_id, history_before_final, tools=None, tool_choice='auto', original_messages=None):
    from flask import Response, stream_with_context

    def generate():
        try:
            # Call DeepSeek API
            response_data = deepseek_api.chat_completion(
                deepseek_session_id,
                user_message,
                parent_message_id=parent_message_id,
                tools=tools,  # Only used if native plugin is enabled
                tool_choice=tool_choice
            )

            # Handle different response formats
            if isinstance(response_data, tuple):
                response_text, message_id = response_data

                # Process response through plugin to extract tool calls
                if tool_plugin and tool_plugin.enabled and original_messages:
                    processed_text, tool_calls = extract_tool_calls_from_response(response_text, original_messages)
                    response_text = processed_text
                else:
                    tool_calls = None

                has_tool_calls = tool_calls is not None
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

    # Tool plugin arguments
    parser.add_argument('--enable-tools', action='store_true', help='Enable tool calling support')
    parser.add_argument('--tool-plugin', type=str, default='simulated', choices=['native', 'simulated'],
                       help='Tool plugin type (native or simulated)')
    parser.add_argument('--tool-placement', type=str, default='user', choices=['user', 'system'],
                       help='Where to place tool definitions (user=working, system=original)')

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

    # Initialize tool plugin
    init_tool_plugin(args.tool_plugin, args.enable_tools, args.tool_placement)

    print(f"Starting DeepSeek OpenAI-compatible server on http://{args.host}:{args.port}")
    print(f"Tool plugin: {args.tool_plugin if args.enable_tools else 'disabled'}")
    if args.enable_tools:
        print(f"Tool placement: {args.tool_placement} (user=working with last user message, system=original approach)")
    print("Endpoints:")
    print(f"  GET  /v1/models")
    print(f"  POST /v1/chat/completions")
    print(f"  POST /v1/completions")
    print(f"  GET  /health")
    print("\nPress Ctrl+C to stop")

    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
