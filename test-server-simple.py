# python proxy-server.py --port 5005 --enable-tools --tool-plugin simulated --replace-system

#!/usr/bin/env python3
"""
OpenAI-compatible server test client.
Tests: multi-turn conversation, tool calls (round-trip), streaming (SSE).

Usage:
    pip install openai rich pytest
    pytest test-server-simple.py -v -s
"""

import argparse
import json
import pytest
import sys
import time
from typing import Any

from openai import OpenAI
from openai import APIConnectionError, APIStatusError


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Paris'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '(3 + 5) * 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    if name == "get_weather":
        city = arguments.get("city", "Unknown")
        unit = arguments.get("unit", "celsius")
        temp = 22 if unit == "celsius" else 72
        return json.dumps(
            {
                "city": city,
                "temperature": temp,
                "unit": unit,
                "condition": "Partly cloudy",
                "humidity": "60%",
            }
        )
    elif name == "calculate":
        expr = arguments.get("expression", "")
        try:
            allowed = set("0123456789+-*/()., ")
            if all(c in allowed for c in expr):
                result = eval(expr)
            else:
                result = "Error: unsupported expression"
        except Exception as e:
            result = f"Error: {e}"
        return json.dumps({"expression": expr, "result": result})
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


@pytest.fixture
def client():
    return OpenAI(
        base_url="http://localhost:5005/v1",
        api_key="not-needed",
        timeout=360.0,
    )


@pytest.fixture
def model():
    return "deepseek-chat"


@pytest.fixture
def system_msg():
    return {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "When the user asks about the weather, ALWAYS use the get_weather tool. "
            "When the user asks to calculate something, ALWAYS use the calculate tool. "
            "Be concise."
        ),
    }


@pytest.fixture
def messages(system_msg):
    return {"messages": [system_msg]}


def add_user(messages, content):
    messages["messages"].append({"role": "user", "content": content})


def add_assistant(messages, content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    messages["messages"].append(msg)


def add_tool(messages, tool_call_id, content):
    messages["messages"].append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
    )


class TestTurn1BasicNonStreaming:
    def test_http_200_no_exception(self, client, model, messages, system_msg):
        add_user(messages, "Hello! What can you help me with?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            stream=False,
        )
        assert resp is not None

    def test_finish_reason_is_stop(self, client, model, messages, system_msg):
        add_user(messages, "Hello! What can you help me with?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            stream=False,
        )
        choice = resp.choices[0]
        assert choice.finish_reason == "stop", f"got '{choice.finish_reason}'"

    def test_assistant_content_non_empty(self, client, model, messages, system_msg):
        add_user(messages, "Hello! What can you help me with?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            stream=False,
        )
        content = resp.choices[0].message.content
        assert content and content.strip(), f"got: {content}"

    def test_usage_block_present(self, client, model, messages, system_msg):
        add_user(messages, "Hello! What can you help me with?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            stream=False,
        )
        assert resp.usage is not None, "usage block missing"


class TestTurn2ToolCallGetWeather:
    def test_finish_reason_is_tool_calls(self, client, model, messages, system_msg):
        add_user(messages, "What is the weather like in Paris right now?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls", f"got '{choice.finish_reason}'"

    def test_tool_calls_list_present(self, client, model, messages, system_msg):
        add_user(messages, "What is the weather like in Paris right now?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls" and choice.message.tool_calls

    def test_correct_tool_name(self, client, model, messages, system_msg):
        add_user(messages, "What is the weather like in Paris right now?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        choice = resp.choices[0]
        tool_call = choice.message.tool_calls[0]
        assert tool_call.function.name == "get_weather"

    def test_tool_arguments_valid_json(self, client, model, messages, system_msg):
        add_user(messages, "What is the weather like in Paris right now?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        choice = resp.choices[0]
        tool_call = choice.message.tool_calls[0]
        fn_args = json.loads(tool_call.function.arguments)
        assert fn_args.get("city") == "Paris"

    def test_followup_after_tool_result_returns_stop(
        self, client, model, messages, system_msg
    ):
        add_user(messages, "What is the weather like in Paris right now?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        choice = resp.choices[0]
        tool_call = choice.message.tool_calls[0]
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        tool_result = execute_tool(fn_name, fn_args)

        add_assistant(
            messages,
            choice.message.content,
            [tc.model_dump() for tc in choice.message.tool_calls],
        )
        add_tool(messages, tool_call.id, tool_result)

        resp2 = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            stream=False,
        )
        choice2 = resp2.choices[0]
        assert choice2.finish_reason == "stop"


class TestTurn3ToolCallCalculate:
    def test_tool_calls_for_calculate(self, client, model, messages, system_msg):
        add_user(messages, "Can you calculate (123 * 456) + 789 for me?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls" and choice.message.tool_calls


class TestTurn4Streaming:
    def test_streaming_receives_chunks(self, client, model, messages, system_msg):
        add_user(messages, "Summarize what we talked about in 2-3 sentences.")
        stream = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            stream=True,
        )

        chunk_count = 0
        for chunk in stream:
            if chunk.choices:
                chunk_count += 1

        assert chunk_count > 1, f"got {chunk_count} chunks"

    def test_streamed_content_non_empty(self, client, model, messages, system_msg):
        add_user(messages, "Summarize what we talked about in 2-3 sentences.")
        stream = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            stream=True,
        )

        collected_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                collected_text += chunk.choices[0].delta.content

        assert collected_text.strip(), f"got: {collected_text}"

    def test_stream_finish_reason_is_stop(self, client, model, messages, system_msg):
        add_user(messages, "Summarize what we talked about in 2-3 sentences.")
        stream = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            stream=True,
        )

        finish_reason = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        assert finish_reason == "stop", f"got '{finish_reason}'"


class TestTurn5StreamingToolCall:
    def test_streaming_tool_call_detected(self, client, model, messages, system_msg):
        add_user(messages, "One more thing: what's the weather in Tokyo?")
        stream = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
        )

        finish_reason = None
        tool_call_chunks = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_call_chunks[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_chunks[idx]["function"]["name"] += (
                                tc_delta.function.name
                            )
                        if tc_delta.function.arguments:
                            tool_call_chunks[idx]["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )

        assert finish_reason == "tool_calls" and tool_call_chunks

    def test_streamed_tool_name_correct(self, client, model, messages, system_msg):
        add_user(messages, "One more thing: what's the weather in Tokyo?")
        stream = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
        )

        finish_reason = None
        tool_call_chunks = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_call_chunks[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_chunks[idx]["function"]["name"] += (
                                tc_delta.function.name
                            )
                        if tc_delta.function.arguments:
                            tool_call_chunks[idx]["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )

        if finish_reason == "tool_calls" and tool_call_chunks:
            fn_name = tool_call_chunks[0]["function"]["name"]
            assert fn_name == "get_weather"


class TestTurn6JSONResponseFormat:
    def test_http_200_no_exception(self, client, model, messages, system_msg):
        add_user(
            messages,
            "Return a JSON object with keys 'name' and 'role' for a developer.",
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            response_format={"type": "json_object"},
            stream=False,
        )
        assert resp is not None

    def test_finish_reason_is_stop(self, client, model, messages, system_msg):
        add_user(
            messages,
            "Return a JSON object with keys 'name' and 'role' for a developer.",
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            response_format={"type": "json_object"},
            stream=False,
        )
        assert resp.choices[0].finish_reason == "stop"

    def test_response_is_valid_json(self, client, model, messages, system_msg):
        add_user(
            messages,
            "Return a JSON object with keys 'name' and 'role' for a developer.",
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            response_format={"type": "json_object"},
            stream=False,
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        assert parsed is not None

    def test_json_has_required_keys(self, client, model, messages, system_msg):
        add_user(
            messages,
            "Return a JSON object with keys 'name' and 'role' for a developer.",
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            response_format={"type": "json_object"},
            stream=False,
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        assert "name" in parsed and "role" in parsed


class TestTurn7ContinueAfterJSON:
    def test_http_200_no_exception(self, client, model, messages, system_msg):
        add_user(messages, "What is my name from the JSON I gave you?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            stream=False,
        )
        assert resp is not None

    def test_finish_reason_is_stop(self, client, model, messages, system_msg):
        add_user(messages, "What is my name from the JSON I gave you?")
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages["messages"],
            stream=False,
        )
        assert resp.choices[0].finish_reason == "stop"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
