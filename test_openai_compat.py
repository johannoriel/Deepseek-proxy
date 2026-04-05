import json
import logging
from typing import Any
from urllib.request import urlopen

import pytest
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment dependency
    OpenAI = None

pytestmark = pytest.mark.skipif(OpenAI is None, reason="openai package is not installed")

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("test")

BASE_URL = "http://localhost:5005/v1"
MODEL = "deepseek-chat"


@pytest.fixture(scope="session")
def client():
    return OpenAI(base_url=BASE_URL, api_key="unused", timeout=120)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(3+5)*2'"}
                },
                "required": ["expression"],
            },
        },
    },
]


def tool_result(name, args):
    if name == "get_weather":
        return json.dumps({"city": args.get("city"), "temp": 22, "unit": args.get("unit", "celsius"), "condition": "sunny"})
    if name == "calculate":
        try:
            return json.dumps({"result": eval(args["expression"])})
        except Exception:
            return json.dumps({"error": "invalid expression"})
    return json.dumps({"error": "unknown tool"})


session_state: dict[str, Any] = {}


def _dump(obj: Any) -> dict[str, Any]:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def _chat(client: OpenAI, *, messages: list[dict], tools=None, stream=False, session_id=None):
    req = {"model": MODEL, "messages": messages, "stream": stream}
    if tools is not None:
        req["tools"] = tools
    extra_body = None
    if session_id is not None:
        # OpenAI SDK does not accept arbitrary top-level kwargs; pass custom fields via extra_body.
        extra_body = {"session_id": session_id}

    log.info("REQUEST: %s", json.dumps(req, ensure_ascii=False, indent=2))
    if extra_body:
        log.info("REQUEST extra_body: %s", json.dumps(extra_body, ensure_ascii=False, indent=2))
        return client.chat.completions.create(**req, extra_body=extra_body)
    return client.chat.completions.create(**req)


@pytest.mark.phase1
class TestPhase1_Connectivity:
    def test_01_health_check(self):
        log.info("=== test_01_health_check ===")
        with urlopen("http://localhost:5005/health") as resp:
            body = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert body["status"] == "ok"

    def test_02_list_models(self):
        log.info("=== test_02_list_models ===")
        with urlopen("http://localhost:5005/v1/models") as resp:
            body = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert any(m["id"] == MODEL for m in body["data"])

    def test_03_simple_text_response(self, client):
        log.info("=== test_03_simple_text_response ===")
        response = _chat(client, messages=[{"role": "user", "content": "Say hello"}])
        raw = _dump(response)
        log.info("RESPONSE: %s", json.dumps(raw, ensure_ascii=False, indent=2))
        assert raw["choices"][0]["finish_reason"] == "stop"
        assert raw["choices"][0]["message"]["content"]
        assert raw.get("usage")

    def test_04_system_message(self, client):
        log.info("=== test_04_system_message ===")
        messages = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Introduce yourself"},
        ]
        response = _chat(client, messages=messages)
        raw = _dump(response)
        log.info("RESPONSE_CONTENT: %s", raw["choices"][0]["message"]["content"])
        assert raw["choices"][0]["finish_reason"] == "stop"
        assert raw["choices"][0]["message"]["content"]

    def test_05_session_id_returned(self, client):
        log.info("=== test_05_session_id_returned ===")
        response = _chat(client, messages=[{"role": "user", "content": "Create a session please."}])
        raw = _dump(response)
        log.info("RESPONSE: %s", json.dumps(raw, ensure_ascii=False, indent=2))
        assert raw.get("session_id")
        session_state["session_1"] = raw["session_id"]
        session_state["history_1"] = [{"role": "user", "content": "Create a session please."}, {"role": "assistant", "content": raw["choices"][0]["message"]["content"]}]


@pytest.mark.phase2
class TestPhase2_MultiTurn:
    def test_06_turn1_establish_fact(self, client):
        log.info("=== test_06_turn1_establish_fact ===")
        sid = session_state["session_1"]
        history = session_state["history_1"]
        history.append({"role": "user", "content": "My name is Alice and I love astronomy."})
        response = _chat(client, messages=history, session_id=sid)
        raw = _dump(response)
        log.info("RESPONSE: %s", json.dumps(raw, ensure_ascii=False, indent=2))
        history.append({"role": "assistant", "content": raw["choices"][0]["message"]["content"]})

    def test_07_turn2_recall_name(self, client):
        log.info("=== test_07_turn2_recall_name ===")
        sid = session_state["session_1"]
        history = session_state["history_1"]
        history.append({"role": "user", "content": "What is my name?"})
        response = _chat(client, messages=history, session_id=sid)
        raw = _dump(response)
        reply = raw["choices"][0]["message"]["content"]
        log.info("FULL_REPLY: %s", reply)
        assert "alice" in reply.lower()
        history.append({"role": "assistant", "content": reply})

    def test_08_turn3_recall_interest(self, client):
        log.info("=== test_08_turn3_recall_interest ===")
        sid = session_state["session_1"]
        history = session_state["history_1"]
        history.append({"role": "user", "content": "What topic did I say I love?"})
        response = _chat(client, messages=history, session_id=sid)
        raw = _dump(response)
        reply = raw["choices"][0]["message"]["content"]
        log.info("FULL_REPLY: %s", reply)
        assert "astronomy" in reply.lower()
        history.append({"role": "assistant", "content": reply})

    def test_09_new_session_no_memory(self, client):
        log.info("=== test_09_new_session_no_memory ===")
        response = _chat(client, messages=[{"role": "user", "content": "What is my name?"}])
        raw = _dump(response)
        reply = raw["choices"][0]["message"]["content"]
        log.info("FULL_REPLY: %s", reply)
        assert "alice" not in reply.lower()


@pytest.mark.phase3
class TestPhase3_ToolCalls:
    def test_10_tool_call_triggered(self, client):
        log.info("=== test_10_tool_call_triggered ===")
        messages = [
            {"role": "system", "content": "Use tools when appropriate."},
            {"role": "user", "content": "What is the weather in Paris?"},
        ]
        response = _chat(client, messages=messages, tools=TOOLS)
        raw = _dump(response)
        log.info("TOOL_CALL_RESPONSE: %s", json.dumps(raw, ensure_ascii=False, indent=2))
        assert raw["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = raw["choices"][0]["message"]["tool_calls"]
        assert tool_calls and len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "get_weather"
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["city"].lower() == "paris"
        session_state["phase3_session"] = raw["session_id"]
        session_state["phase3_messages"] = messages + [{"role": "assistant", "content": None, "tool_calls": tool_calls}]

    def test_11_tool_result_sends_stop(self, client):
        log.info("=== test_11_tool_result_sends_stop ===")
        sid = session_state["phase3_session"]
        msgs = session_state["phase3_messages"]
        tc = msgs[-1]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result(tc["function"]["name"], args)})
        response = _chat(client, messages=msgs, tools=TOOLS, session_id=sid)
        raw = _dump(response)
        reply = raw["choices"][0]["message"]["content"]
        log.info("FULL_RESPONSE: %s", reply)
        assert raw["choices"][0]["finish_reason"] == "stop"
        assert any(k in reply.lower() for k in ["weather", "temp", "22"])


@pytest.mark.phase4
class TestPhase4_MultipleTools:
    def test_12_calculate_tool_triggered(self, client):
        log.info("=== test_12_calculate_tool_triggered ===")
        response = _chat(client, messages=[{"role": "user", "content": "Calculate (123 * 456) + 789"}], tools=TOOLS)
        raw = _dump(response)
        tc = raw["choices"][0]["message"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        log.info("EXPRESSION_FOUND: %s", args)
        assert raw["choices"][0]["finish_reason"] == "tool_calls"
        assert tc["function"]["name"] == "calculate"
        assert "expression" in args
        session_state["phase4_session"] = raw["session_id"]
        session_state["phase4_messages"] = [{"role": "user", "content": "Calculate (123 * 456) + 789"}, {"role": "assistant", "content": None, "tool_calls": raw["choices"][0]["message"]["tool_calls"]}]

    def test_13_calculate_result_sent(self, client):
        log.info("=== test_13_calculate_result_sent ===")
        sid = session_state["phase4_session"]
        msgs = session_state["phase4_messages"]
        tc = msgs[-1]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result("calculate", args)})
        response = _chat(client, messages=msgs, tools=TOOLS, session_id=sid)
        raw = _dump(response)
        reply = raw["choices"][0]["message"]["content"]
        assert raw["choices"][0]["finish_reason"] == "stop"
        assert "56877" in reply


@pytest.mark.phase5
class TestPhase5_MultiTurnWithTools:
    def test_14_turn1_normal(self, client):
        log.info("=== test_14_turn1_normal ===")
        messages = [
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": "Hello, I'm Bob. Nice to meet you."},
        ]
        response = _chat(client, messages=messages, tools=TOOLS)
        raw = _dump(response)
        assert raw["choices"][0]["finish_reason"] == "stop"
        assert raw["choices"][0]["message"]["content"]
        session_state["phase5_session"] = raw["session_id"]
        session_state["phase5_messages"] = messages + [{"role": "assistant", "content": raw["choices"][0]["message"]["content"]}]

    def test_15_turn2_tool_call(self, client):
        log.info("=== test_15_turn2_tool_call ===")
        sid = session_state["phase5_session"]
        msgs = session_state["phase5_messages"]
        msgs.append({"role": "user", "content": "What's the weather in Tokyo?"})
        response = _chat(client, messages=msgs, tools=TOOLS, session_id=sid)
        raw = _dump(response)
        tc = raw["choices"][0]["message"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        assert raw["choices"][0]["finish_reason"] == "tool_calls"
        assert args["city"].lower() == "tokyo"
        msgs.append({"role": "assistant", "content": None, "tool_calls": raw["choices"][0]["message"]["tool_calls"]})

    def test_16_turn3_tool_result(self, client):
        log.info("=== test_16_turn3_tool_result ===")
        sid = session_state["phase5_session"]
        msgs = session_state["phase5_messages"]
        tc = msgs[-1]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result(tc["function"]["name"], args)})
        response = _chat(client, messages=msgs, tools=TOOLS, session_id=sid)
        raw = _dump(response)
        reply = raw["choices"][0]["message"]["content"]
        assert raw["choices"][0]["finish_reason"] == "stop"
        assert any(k in reply.lower() for k in ["tokyo", "temp", "weather", "22"])
        msgs.append({"role": "assistant", "content": reply})

    def test_17_turn4_memory_check(self, client):
        log.info("=== test_17_turn4_memory_check ===")
        sid = session_state["phase5_session"]
        msgs = session_state["phase5_messages"]
        msgs.append({"role": "user", "content": "Who am I? And what city's weather did I just ask about?"})
        response = _chat(client, messages=msgs, tools=TOOLS, session_id=sid)
        raw = _dump(response)
        reply = raw["choices"][0]["message"]["content"]
        log.info("FULL_RESPONSE: %s", reply)
        assert raw["choices"][0]["finish_reason"] == "stop"
        assert "bob" in reply.lower()
        assert "tokyo" in reply.lower()


@pytest.mark.phase6
class TestPhase6_Streaming:
    def test_18_streaming_basic(self, client):
        log.info("=== test_18_streaming_basic ===")
        stream = _chat(client, messages=[{"role": "user", "content": "Count from 1 to 5."}], stream=True)
        chunks = list(stream)
        assembled = "".join((c.choices[0].delta.content or "") for c in chunks if c.choices)
        final_reason = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
        log.info("chunk_count=%s assembled=%s", len(chunks), assembled)
        assert len(chunks) >= 3
        assert assembled.strip()
        assert final_reason[-1] == "stop"

    def test_19_streaming_tool_call(self, client):
        log.info("=== test_19_streaming_tool_call ===")
        stream = _chat(client, messages=[{"role": "user", "content": "What's the weather in London?"}], tools=TOOLS, stream=True)
        chunks = list(stream)
        tool_name = ""
        arg_parts = []
        finish = None
        for ch in chunks:
            if not ch.choices:
                continue
            choice = ch.choices[0]
            if choice.delta and choice.delta.tool_calls:
                tc = choice.delta.tool_calls[0]
                if tc.function and tc.function.name:
                    tool_name = tc.function.name
                if tc.function and tc.function.arguments:
                    arg_parts.append(tc.function.arguments)
            if choice.finish_reason:
                finish = choice.finish_reason
        args = json.loads("".join(arg_parts))
        log.info("STREAM_TOOL: name=%s args=%s finish=%s", tool_name, args, finish)
        assert finish == "tool_calls"
        assert tool_name == "get_weather"
        assert "city" in args


@pytest.mark.phase7
class TestPhase7_ParallelSessions:
    def test_20_session_A_setup(self, client):
        log.info("=== test_20_session_A_setup ===")
        raw = _dump(_chat(client, messages=[{"role": "user", "content": "I am Carol and I like jazz."}]))
        session_state["A_sid"] = raw["session_id"]
        session_state["A_history"] = [{"role": "user", "content": "I am Carol and I like jazz."}, {"role": "assistant", "content": raw["choices"][0]["message"]["content"]}]

    def test_21_session_B_setup(self, client):
        log.info("=== test_21_session_B_setup ===")
        raw = _dump(_chat(client, messages=[{"role": "user", "content": "I am Dave and I like chess."}]))
        session_state["B_sid"] = raw["session_id"]
        session_state["B_history"] = [{"role": "user", "content": "I am Dave and I like chess."}, {"role": "assistant", "content": raw["choices"][0]["message"]["content"]}]

    def test_22_session_A_recall(self, client):
        log.info("=== test_22_session_A_recall ===")
        h = session_state["A_history"]
        h.append({"role": "user", "content": "What is my name and hobby?"})
        raw = _dump(_chat(client, messages=h, session_id=session_state["A_sid"]))
        reply = raw["choices"][0]["message"]["content"].lower()
        assert "carol" in reply and "jazz" in reply

    def test_23_session_B_recall(self, client):
        log.info("=== test_23_session_B_recall ===")
        h = session_state["B_history"]
        h.append({"role": "user", "content": "What is my name and hobby?"})
        raw = _dump(_chat(client, messages=h, session_id=session_state["B_sid"]))
        reply = raw["choices"][0]["message"]["content"].lower()
        assert "dave" in reply and "chess" in reply

    def test_24_cross_session_isolation(self, client):
        log.info("=== test_24_cross_session_isolation ===")
        h = session_state["A_history"]
        h.append({"role": "user", "content": "Do you know anyone named Dave?"})
        raw = _dump(_chat(client, messages=h, session_id=session_state["A_sid"]))
        reply = raw["choices"][0]["message"]["content"]
        log.info("MANUAL_REVIEW_RESPONSE: %s", reply)
        assert "dave" not in reply.lower()


@pytest.mark.phase8
class TestPhase8_EdgeCases:
    def test_25_empty_tools_list(self, client):
        log.info("=== test_25_empty_tools_list ===")
        raw = _dump(_chat(client, messages=[{"role": "user", "content": "Hello with empty tools."}], tools=[]))
        assert raw["choices"][0]["finish_reason"] == "stop"

    def test_26_no_tool_call_needed(self, client):
        log.info("=== test_26_no_tool_call_needed ===")
        raw = _dump(_chat(client, messages=[{"role": "user", "content": "What is 2+2?"}], tools=TOOLS))
        reason = raw["choices"][0]["finish_reason"]
        log.info("MODEL_CHOICE_REASON: %s response=%s", reason, json.dumps(raw, ensure_ascii=False))
        assert reason in ["stop", "tool_calls"]

    def test_27_long_system_message(self, client):
        log.info("=== test_27_long_system_message ===")
        long_system = "A" * 500
        raw = _dump(_chat(client, messages=[{"role": "system", "content": long_system}, {"role": "user", "content": "Summarize your instructions."}]))
        assert raw["choices"][0]["finish_reason"] == "stop"
        assert raw["choices"][0]["message"]["content"]

    def test_28_unicode_content(self, client):
        log.info("=== test_28_unicode_content ===")
        raw = _dump(_chat(client, messages=[{"role": "user", "content": "Tell me about café culture ☕"}]))
        assert raw["choices"][0]["finish_reason"] == "stop"
        assert raw["choices"][0]["message"]["content"]
