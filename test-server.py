#!/usr/bin/env python3
"""
Comprehensive test suite for proxy-server.py OpenAI compatibility.
Tests: multi-turn conversations, tool calls (simulated), system/user/assistant/tool messages,
streaming and non-streaming modes.

The proxy server should be running on port 5005 with:
  python proxy-server.py --port 5005 --enable-tools --tool-plugin simulated --replace-system
"""

import json
import sys
import time
import requests
from typing import Dict, Any, List, Optional

BASE_URL = "http://127.0.0.1:5005"
MODEL = "deepseek-chat"

# ── Helpers ──────────────────────────────────────────────────────────────────


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def sub_section(title: str):
    print(f"\n--- {title}")


def ok(msg: str):
    print(f"  ✅ {msg}")


def fail(msg: str):
    print(f"  ❌ {msg}")


def chat_request(
    messages: List[Dict],
    stream: bool = False,
    tools: Optional[List[Dict]] = None,
    tool_choice: str = "auto",
    session_id: Optional[str] = None,
) -> requests.Response:
    payload: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    if session_id:
        payload["session_id"] = session_id
    return requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )


def validate_non_streaming_response(resp: requests.Response, label: str = "") -> Dict:
    """Validate a non-streaming JSON response matches OpenAI format."""
    assert resp.status_code == 200, f"{label} HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "id" in data, f"{label} missing 'id'"
    assert data["object"] == "chat.completion", (
        f"{label} wrong object: {data['object']}"
    )
    assert "created" in data, f"{label} missing 'created'"
    assert data["model"] == MODEL, f"{label} wrong model: {data['model']}"
    assert len(data["choices"]) == 1, f"{label} expected 1 choice"
    choice = data["choices"][0]
    assert "index" in choice, f"{label} missing 'index'"
    assert "message" in choice, f"{label} missing 'message'"
    msg = choice["message"]
    assert "role" in msg, f"{label} missing 'role' in message"
    assert msg["role"] == "assistant", f"{label} wrong role: {msg['role']}"
    assert "content" in msg, f"{label} missing 'content' in message"
    assert "finish_reason" in choice, f"{label} missing 'finish_reason'"
    assert choice["finish_reason"] in ("stop", "tool_calls"), (
        f"{label} bad finish_reason: {choice['finish_reason']}"
    )
    if "usage" in data:
        u = data["usage"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert k in u, f"{label} missing usage.{k}"
    ok(f"{label} passed (finish_reason={choice['finish_reason']})")
    return data


def validate_streaming_response(resp: requests.Response, label: str = "") -> List[Dict]:
    """Validate a streaming SSE response matches OpenAI format."""
    assert resp.status_code == 200, f"{label} HTTP {resp.status_code}"
    assert "text/event-stream" in resp.headers.get("Content-Type", ""), (
        f"{label} wrong Content-Type: {resp.headers.get('Content-Type')}"
    )
    chunks: List[Dict] = []
    got_role = False
    got_content = False
    got_finish = False
    got_done = False

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            got_done = True
            continue
        chunk = json.loads(payload)
        chunks.append(chunk)
        assert chunk["object"] == "chat.completion.chunk", (
            f"{label} wrong object in chunk: {chunk['object']}"
        )
        delta = chunk["choices"][0]["delta"]
        if "role" in delta:
            got_role = True
        if "content" in delta and delta["content"]:
            got_content = True
        if "tool_calls" in delta:
            got_content = True  # tool calls also count as content
        if chunk["choices"][0].get("finish_reason"):
            got_finish = True
            assert chunk["choices"][0]["finish_reason"] in ("stop", "tool_calls"), (
                f"{label} bad finish_reason: {chunk['choices'][0]['finish_reason']}"
            )

    assert got_role, f"{label} streaming never sent role"
    assert got_content, f"{label} streaming never sent content/tool_calls"
    assert got_finish, f"{label} streaming never sent finish_reason"
    assert got_done, f"{label} streaming never sent [DONE]"
    ok(f"{label} passed ({len(chunks)} chunks)")
    return chunks


# ── Test definitions ─────────────────────────────────────────────────────────

TOOLS_WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

TOOLS_CALC = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

TOOLS_MULTI = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        },
    },
]


def test_health():
    section("TEST 1: Health check")
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    ok("Health check passed")


def test_list_models():
    section("TEST 2: GET /v1/models")
    resp = requests.get(f"{BASE_URL}/v1/models", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    model_ids = [m["id"] for m in data["data"]]
    assert MODEL in model_ids, f"Model {MODEL} not in list: {model_ids}"
    ok(f"Models listed: {model_ids}")


def test_simple_chat_non_streaming():
    section("TEST 3: Simple chat (non-streaming)")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in exactly 3 words."},
    ]
    resp = chat_request(messages, stream=False)
    data = validate_non_streaming_response(resp, "Simple chat")
    content = data["choices"][0]["message"]["content"]
    print(f"    Response: {content}")
    assert content, "Empty content"


def test_simple_chat_streaming():
    section("TEST 4: Simple chat (streaming)")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 5."},
    ]
    resp = chat_request(messages, stream=True)
    validate_streaming_response(resp, "Simple streaming")


def test_multi_turn_conversation_non_streaming():
    section("TEST 5: Multi-turn conversation (non-streaming)")

    # Turn 1
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Alice."},
    ]
    resp1 = chat_request(messages, stream=False)
    data1 = validate_non_streaming_response(resp1, "Turn 1")
    session_id = data1.get("session_id")
    messages.append(data1["choices"][0]["message"])

    # Turn 2
    messages.append({"role": "user", "content": "What is my name?"})
    resp2 = chat_request(messages, stream=False)
    data2 = validate_non_streaming_response(resp2, "Turn 2")
    content2 = data2["choices"][0]["message"]["content"]
    print(f"    Turn 2 response: {content2}")
    assert "Alice" in content2 or "alice" in content2.lower(), (
        f"Model did not remember name: {content2}"
    )
    messages.append(data2["choices"][0]["message"])

    # Turn 3
    messages.append({"role": "user", "content": "Repeat my name again."})
    resp3 = chat_request(messages, stream=False)
    data3 = validate_non_streaming_response(resp3, "Turn 3")
    content3 = data3["choices"][0]["message"]["content"]
    print(f"    Turn 3 response: {content3}")
    assert "Alice" in content3 or "alice" in content3.lower(), (
        f"Model forgot name on turn 3: {content3}"
    )


def test_multi_turn_conversation_streaming():
    section("TEST 6: Multi-turn conversation (streaming)")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I like apples and bananas."},
    ]
    resp1 = chat_request(messages, stream=True)
    validate_streaming_response(resp1, "Turn 1 streaming")

    messages.append(
        {"role": "assistant", "content": "Got it, you like apples and bananas."}
    )
    messages.append({"role": "user", "content": "What fruits do I like?"})
    resp2 = chat_request(messages, stream=True)
    validate_streaming_response(resp2, "Turn 2 streaming")


def test_session_persistence():
    section("TEST 7: Session persistence via session_id")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "The secret word is PINEAPPLE."},
    ]
    resp1 = chat_request(messages, stream=False)
    data1 = validate_non_streaming_response(resp1, "Turn 1 (create session)")
    session_id = data1.get("session_id")
    assert session_id, "No session_id returned"
    ok(f"Session ID: {session_id}")

    messages.append(data1["choices"][0]["message"])
    messages.append({"role": "user", "content": "What was the secret word?"})
    resp2 = chat_request(messages, stream=False, session_id=session_id)
    data2 = validate_non_streaming_response(resp2, "Turn 2 (reuse session)")
    content2 = data2["choices"][0]["message"]["content"]
    print(f"    Response: {content2}")
    assert "PINEAPPLE" in content2 or "pineapple" in content2.lower(), (
        f"Session did not persist: {content2}"
    )


def test_tool_call_non_streaming():
    section("TEST 8: Tool call (non-streaming)")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
    ]
    resp = chat_request(messages, stream=False, tools=TOOLS_WEATHER)
    data = validate_non_streaming_response(resp, "Tool call request")
    choice = data["choices"][0]
    finish_reason = choice["finish_reason"]
    msg = choice["message"]

    if finish_reason == "tool_calls" or "tool_calls" in msg:
        ok("Model returned tool_calls")
        tool_calls = msg.get("tool_calls", [])
        assert len(tool_calls) > 0, "tool_calls is empty"
        for tc in tool_calls:
            assert "id" in tc, "tool_call missing id"
            assert "type" in tc, "tool_call missing type"
            assert tc["type"] == "function", f"tool_call wrong type: {tc['type']}"
            assert "function" in tc, "tool_call missing function"
            fn = tc["function"]
            assert "name" in fn, "tool_call function missing name"
            assert "arguments" in fn, "tool_call function missing arguments"
            print(f"    Tool: {fn['name']}, Args: {fn['arguments']}")

        # Simulate sending tool results back
        messages.append(msg)
        for tc in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "The weather in Paris is 22°C and sunny.",
                }
            )
        resp2 = chat_request(messages, stream=False, tools=TOOLS_WEATHER)
        data2 = validate_non_streaming_response(resp2, "Tool result follow-up")
        content2 = data2["choices"][0]["message"]["content"]
        print(f"    Final response after tool: {content2}")
    else:
        ok(
            f"Model gave text response instead of tool call (finish_reason={finish_reason})"
        )
        print(f"    Response: {msg.get('content', '')}")


def test_tool_call_streaming():
    section("TEST 9: Tool call (streaming)")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 1234 * 5678?"},
    ]
    resp = chat_request(messages, stream=True, tools=TOOLS_CALC)
    chunks = validate_streaming_response(resp, "Tool call streaming")

    # Check if any chunk contains tool_calls
    has_tool_calls = False
    for chunk in chunks:
        delta = chunk["choices"][0].get("delta", {})
        if "tool_calls" in delta:
            has_tool_calls = True
            for tc in delta["tool_calls"]:
                if "id" in tc:
                    print(f"    Streaming tool_call id: {tc['id']}")
                if "function" in tc:
                    fn = tc["function"]
                    if "name" in fn:
                        print(f"    Streaming tool_call function: {fn['name']}")
                    if "arguments" in fn:
                        print(
                            f"    Streaming tool_call args chunk: {fn['arguments'][:50]}"
                        )

    if has_tool_calls:
        ok("Streaming response contained tool_calls")
    else:
        ok("Streaming response was text (model did not use tools in stream)")


def test_multiple_tool_calls_in_one_response():
    section("TEST 10: Multiple tool calls in one response")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the weather in Paris and also calculate 100 + 200?",
        },
    ]
    resp = chat_request(messages, stream=False, tools=TOOLS_MULTI)
    data = validate_non_streaming_response(resp, "Multi-tool request")
    msg = data["choices"][0]["message"]

    if "tool_calls" in msg:
        tool_calls = msg["tool_calls"]
        print(f"    Number of tool calls: {len(tool_calls)}")
        assert len(tool_calls) >= 1, "Expected at least 1 tool call"
        for tc in tool_calls:
            fn = tc["function"]
            print(f"    Tool: {fn['name']}, Args: {fn['arguments']}")

        # Send back tool results for all
        messages.append(msg)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            if fn_name == "get_weather":
                tool_result = "Paris: 22°C, Sunny"
            elif fn_name == "calculate":
                tool_result = "300"
            else:
                tool_result = "unknown tool"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                }
            )

        resp2 = chat_request(messages, stream=False, tools=TOOLS_MULTI)
        data2 = validate_non_streaming_response(resp2, "Multi-tool follow-up")
        content2 = data2["choices"][0]["message"]["content"]
        print(f"    Final response: {content2}")
    else:
        ok(
            f"Model gave text response (finish_reason={data['choices'][0]['finish_reason']})"
        )
        print(f"    Response: {msg.get('content', '')}")


def test_conversation_with_tools_and_multiple_turns():
    section("TEST 11: Multi-turn conversation WITH tool calls")

    # Turn 1: Ask something that should trigger a tool
    messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What is the weather in Tokyo?"},
    ]
    resp1 = chat_request(messages, stream=False, tools=TOOLS_WEATHER)
    data1 = validate_non_streaming_response(resp1, "Turn 1: weather query")
    msg1 = data1["choices"][0]["message"]
    messages.append(msg1)

    if "tool_calls" in msg1:
        tc = msg1["tool_calls"][0]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": "Tokyo: 18°C, Cloudy",
            }
        )
        resp1b = chat_request(messages, stream=False, tools=TOOLS_WEATHER)
        data1b = validate_non_streaming_response(resp1b, "Turn 1b: tool result")
        messages.append(data1b["choices"][0]["message"])
        print(f"    After tool result: {data1b['choices'][0]['message']['content']}")

    # Turn 2: Follow-up question (should remember context)
    messages.append({"role": "user", "content": "And what about Osaka?"})
    resp2 = chat_request(messages, stream=False, tools=TOOLS_WEATHER)
    data2 = validate_non_streaming_response(resp2, "Turn 2: follow-up weather")
    msg2 = data2["choices"][0]["message"]
    messages.append(msg2)

    if "tool_calls" in msg2:
        tc2 = msg2["tool_calls"][0]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc2["id"],
                "content": "Osaka: 20°C, Partly cloudy",
            }
        )
        resp2b = chat_request(messages, stream=False, tools=TOOLS_WEATHER)
        data2b = validate_non_streaming_response(resp2b, "Turn 2b: tool result")
        messages.append(data2b["choices"][0]["message"])
        print(f"    After tool result: {data2b['choices'][0]['message']['content']}")

    # Turn 3: Non-tool follow-up
    messages.append({"role": "user", "content": "Which city is warmer?"})
    resp3 = chat_request(messages, stream=False, tools=TOOLS_WEATHER)
    data3 = validate_non_streaming_response(resp3, "Turn 3: comparison")
    content3 = data3["choices"][0]["message"]["content"]
    print(f"    Comparison answer: {content3}")


def test_system_message_replacement():
    section("TEST 12: System message handling (with --replace-system)")

    messages = [
        {
            "role": "system",
            "content": "You must always respond in ALL CAPS.",
        },
        {"role": "user", "content": "Hello, how are you?"},
    ]
    resp = chat_request(messages, stream=False)
    data = validate_non_streaming_response(resp, "System message test")
    content = data["choices"][0]["message"]["content"]
    print(f"    Response: {content}")
    # With --replace-system, the system message becomes a user message
    # so the model may or may not follow the instruction
    ok("System message handled without error")


def test_empty_content_assistant_message():
    section("TEST 13: Assistant message with tool_calls but no content")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in London?"},
    ]
    resp = chat_request(messages, stream=False, tools=TOOLS_WEATHER)
    data = validate_non_streaming_response(
        resp, "Tool call with possible empty content"
    )
    msg = data["choices"][0]["message"]
    has_tool_calls = "tool_calls" in msg
    has_content = bool(msg.get("content", ""))
    print(f"    Has tool_calls: {has_tool_calls}")
    print(f"    Has content: {has_content}")
    if has_tool_calls:
        print(f"    Content: '{msg.get('content', '')}'")
        ok("Tool call response valid (content may be empty for tool-only responses)")


def test_tool_choice_auto():
    section("TEST 14: tool_choice=auto")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Just say hi."},
    ]
    resp = chat_request(messages, stream=False, tools=TOOLS_WEATHER, tool_choice="auto")
    data = validate_non_streaming_response(
        resp, "tool_choice=auto (should not force tool)"
    )
    content = data["choices"][0]["message"]["content"]
    print(f"    Response: {content}")
    ok("tool_choice=auto works")


def test_long_conversation():
    section("TEST 15: Long multi-turn conversation (5+ turns)")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses brief.",
        },
        {"role": "user", "content": "I am planning a trip to Japan."},
    ]
    resp = chat_request(messages, stream=False)
    data = validate_non_streaming_response(resp, "Turn 1")
    messages.append(data["choices"][0]["message"])

    turns = [
        "What is the best time to visit?",
        "What about food recommendations?",
        "How should I get around in Tokyo?",
        "Any cultural tips I should know?",
        "Thank you, summarize everything briefly.",
    ]

    for i, turn_text in enumerate(turns, 2):
        messages.append({"role": "user", "content": turn_text})
        resp = chat_request(messages, stream=False)
        data = validate_non_streaming_response(resp, f"Turn {i}")
        content = data["choices"][0]["message"]["content"]
        print(f"    Turn {i} response ({len(content)} chars)")
        messages.append(data["choices"][0]["message"])

    ok(f"Completed {len(turns) + 1} turns successfully")


def test_tool_call_with_streaming_and_followup():
    section("TEST 16: Tool call (streaming) + follow-up (non-streaming)")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Calculate 42 * 17."},
    ]
    resp = chat_request(messages, stream=True, tools=TOOLS_CALC)
    chunks = validate_streaming_response(resp, "Tool call streaming")

    # Reconstruct the full message from chunks
    full_content = ""
    tool_calls_data = []
    for chunk in chunks:
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta and delta["content"]:
            full_content += delta["content"]
        if "tool_calls" in delta:
            for tc in delta["tool_calls"]:
                tool_calls_data.append(tc)

    print(f"    Reconstructed content: {full_content[:100]}")

    # Follow up
    messages.append(
        {"role": "assistant", "content": full_content if full_content else "Done."}
    )
    messages.append({"role": "user", "content": "Now calculate that result + 100."})
    resp2 = chat_request(messages, stream=False, tools=TOOLS_CALC)
    data2 = validate_non_streaming_response(resp2, "Follow-up after streaming tool")
    print(f"    Follow-up response: {data2['choices'][0]['message']['content']}")


def test_mixed_roles_in_messages():
    section("TEST 17: Mixed roles (system, user, assistant, tool)")

    messages = [
        {"role": "system", "content": "You are a calculator assistant."},
        {"role": "user", "content": "What is 10 + 20?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_test_001",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression": "10 + 20"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_test_001",
            "content": "30",
        },
        {"role": "user", "content": "Now add 5 to that result."},
    ]
    resp = chat_request(messages, stream=False, tools=TOOLS_CALC)
    data = validate_non_streaming_response(resp, "Mixed roles")
    content = data["choices"][0]["message"]["content"]
    print(f"    Response: {content}")
    ok("Mixed roles handled correctly")


def test_rapid_succession_requests():
    section("TEST 18: Rapid succession requests (no session reuse)")

    for i in range(3):
        messages = [
            {"role": "user", "content": f"Request number {i + 1}. Just say OK."},
        ]
        resp = chat_request(messages, stream=False)
        data = validate_non_streaming_response(resp, f"Rapid request {i + 1}")
        print(f"    Response {i + 1}: {data['choices'][0]['message']['content'][:50]}")

    ok("All 3 rapid requests succeeded")


def test_streaming_multi_turn_with_tools():
    section("TEST 19: Streaming multi-turn with tool calls")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in New York?"},
    ]
    resp1 = chat_request(messages, stream=True, tools=TOOLS_WEATHER)
    chunks1 = validate_streaming_response(resp1, "Turn 1 streaming with tools")

    # Check if tool calls were in the stream
    has_tool = any(
        "tool_calls" in chunk["choices"][0].get("delta", {}) for chunk in chunks1
    )

    if has_tool:
        ok("Turn 1 streaming contained tool calls")
        # Build assistant message with tool_calls from chunks
        tool_calls_from_stream = []
        for chunk in chunks1:
            delta = chunk["choices"][0].get("delta", {})
            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    tool_calls_from_stream.append(tc)

        # Just verify we got tool call data
        if tool_calls_from_stream:
            print(f"    Extracted {len(tool_calls_from_stream)} tool call chunks")

    # Turn 2: follow up
    messages.append({"role": "assistant", "content": "Let me check the weather."})
    messages.append({"role": "user", "content": "Thanks, also check Boston."})
    resp2 = chat_request(messages, stream=True, tools=TOOLS_WEATHER)
    validate_streaming_response(resp2, "Turn 2 streaming follow-up")


def test_error_handling():
    section("TEST 20: Error handling (empty messages)")

    resp = chat_request([], stream=False)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    ok("Empty messages correctly returns 400")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  Deepseek-proxy OpenAI Compatibility Test Suite")
    print(f"  Target: {BASE_URL}")
    print("=" * 70)

    tests = [
        test_health,
        test_list_models,
        test_simple_chat_non_streaming,
        test_simple_chat_streaming,
        test_multi_turn_conversation_non_streaming,
        test_multi_turn_conversation_streaming,
        test_session_persistence,
        test_tool_call_non_streaming,
        test_tool_call_streaming,
        test_multiple_tool_calls_in_one_response,
        test_conversation_with_tools_and_multiple_turns,
        test_system_message_replacement,
        test_empty_content_assistant_message,
        test_tool_choice_auto,
        test_long_conversation,
        test_tool_call_with_streaming_and_followup,
        test_mixed_roles_in_messages,
        test_rapid_succession_requests,
        test_streaming_multi_turn_with_tools,
        test_error_handling,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            fail(f"{test_fn.__name__}: {e}")

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 70}")

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
