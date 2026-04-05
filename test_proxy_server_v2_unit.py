import json

import pytest
from tool_parser import clean_text_response, extract_tool_calls

flask = pytest.importorskip("flask")
import proxy_server_v2
from proxy_server_v2 import create_app


class FakeDeepSeekAPI:
    def __init__(self, _api_key: str):
        self.calls = []
        self._session_counter = 0

    def create_chat_session(self):
        self._session_counter += 1
        return f"backend-session-{self._session_counter}"

    def chat_completion(self, session_id, prompt_text, parent_message_id=None):
        self.calls.append(
            {
                "session_id": session_id,
                "prompt_text": prompt_text,
                "parent_message_id": parent_message_id,
            }
        )

        if len(self.calls) == 1:
            response = (
                'I will run both tools now. '\
                '{"tool_calls": ['
                '{"name": "get_weather", "arguments": {"city": "Paris"}}, '
                '{"name": "calculate", "arguments": {"expression": "2+2"}}'
                ']} FINISHED'
            )
            return response, "msg-1"

        return "Done. Paris weather + math completed. FINISHED", "msg-2"


def _build_client(monkeypatch):
    holder = {}

    def _factory(api_key):
        api = FakeDeepSeekAPI(api_key)
        holder["api"] = api
        return api

    monkeypatch.setattr(proxy_server_v2, "DeepSeekAPI", _factory)
    app = create_app("test-key")
    return app.test_client(), holder


def test_extract_tool_calls_multiple_and_mixed_text():
    text = (
        "Let's do this. "
        '{"tool_call": {"name": "foo", "arguments": {"x": 1}}} '
        "Then this too: "
        '{"tool_calls": [{"name": "bar", "arguments": {"y": 2}}]}'
    )

    calls = extract_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["tool_call"]["name"] == "foo"
    assert calls[1]["tool_call"]["name"] == "bar"

    cleaned = clean_text_response(text)
    assert "Let's do this." in cleaned
    assert "Then this too:" in cleaned
    assert "tool_call" not in cleaned


def test_chat_completion_extracts_parallel_tool_calls_and_preserves_text(monkeypatch):
    client, _ = _build_client(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Need weather in Paris and calculate 2+2"}],
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "calculate", "parameters": {"type": "object"}}},
            ],
        },
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["content"] == "I will run both tools now."
    tool_calls = body["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 2
    assert [tc["function"]["name"] for tc in tool_calls] == ["get_weather", "calculate"]


def test_multi_tool_followup_keeps_session_history(monkeypatch):
    client, holder = _build_client(monkeypatch)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Need weather in Paris and calculate 2+2"}],
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "calculate", "parameters": {"type": "object"}}},
            ],
        },
    )
    first_body = first.get_json()
    session_id = first_body["session_id"]
    tool_calls = first_body["choices"][0]["message"]["tool_calls"]

    messages = [
        {"role": "user", "content": "Need weather in Paris and calculate 2+2"},
        {"role": "assistant", "content": first_body["choices"][0]["message"]["content"], "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": json.dumps({"city": "Paris", "temp": 20})},
        {"role": "tool", "tool_call_id": tool_calls[1]["id"], "content": json.dumps({"result": 4})},
        {"role": "user", "content": "Great, summarize both results in one line."},
    ]

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "session_id": session_id,
            "messages": messages,
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "calculate", "parameters": {"type": "object"}}},
            ],
        },
    )

    second_body = second.get_json()
    assert second.status_code == 200
    assert second_body["choices"][0]["finish_reason"] == "stop"
    assert "completed" in second_body["choices"][0]["message"]["content"].lower()

    api = holder["api"]
    assert len(api.calls) == 2
    assert api.calls[1]["session_id"] == api.calls[0]["session_id"]
    assert api.calls[1]["parent_message_id"] == "msg-1"
    assert "summarize both results" in api.calls[1]["prompt_text"].lower()


def test_slowdown_scales_with_prompt_length(monkeypatch):
    client, _ = _build_client(monkeypatch)
    sleep_calls = []

    monkeypatch.setattr(proxy_server_v2, "slowdown_per_1000_chars", 1.0)
    monkeypatch.setattr(proxy_server_v2.time, "sleep", lambda value: sleep_calls.append(value))

    client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "a" * 1200}],
        },
    )

    assert sleep_calls, "Expected slowdown sleep call"
    assert sleep_calls[0] >= 1.0
