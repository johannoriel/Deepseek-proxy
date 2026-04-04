# python proxy-server.py --port 5005 --enable-tools --tool-plugin simulated --replace-system

#!/usr/bin/env python3
"""
OpenAI-compatible server test client.
Tests: multi-turn conversation, tool calls (round-trip), streaming (SSE).

Usage:
    pip install openai rich
    python test_openai_server.py --base-url http://localhost:8000/v1 --model your-model-name
"""

import argparse
import json
import sys
import time
from typing import Any

from openai import OpenAI
from openai import APIConnectionError, APIStatusError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# ---------------------------------------------------------------------------
# Tool definitions (sent to the server)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Fake tool executor (runs locally, simulates real tool results)
# ---------------------------------------------------------------------------


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
            # Safe eval for simple arithmetic
            allowed = set("0123456789+-*/()., ")
            if all(c in allowed for c in expr):
                result = eval(expr)  # noqa: S307
            else:
                result = "Error: unsupported expression"
        except Exception as e:
            result = f"Error: {e}"
        return json.dumps({"expression": expr, "result": result})
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


class TestResults:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []  # (label, passed, detail)

    def record(self, label: str, passed: bool, detail: str = ""):
        self.checks.append((label, passed, detail))

    def summary(self, console: Console):
        console.print(Rule("Test Summary"))
        passed = sum(1 for _, ok, _ in self.checks if ok)
        total = len(self.checks)
        for label, ok, detail in self.checks:
            icon = "✅" if ok else "❌"
            line = f"{icon} {label}"
            if detail:
                line += f"  — {detail}"
            console.print(line)
        console.print()
        color = "green" if passed == total else "red"
        console.print(f"[{color}]{passed}/{total} checks passed[/{color}]")


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------


def run_tests(client: OpenAI, model: str, console: Console, results: TestResults):
    messages: list[dict[str, Any]] = []

    # ── System message ──────────────────────────────────────────────────────
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "When the user asks about the weather, ALWAYS use the get_weather tool. "
            "When the user asks to calculate something, ALWAYS use the calculate tool. "
            "Be concise."
        ),
    }

    # ═══════════════════════════════════════════════════════════════════════
    # TURN 1 — Basic non-streaming reply
    # ═══════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Turn 1 — Basic non-streaming reply"))
    messages.append({"role": "user", "content": "Hello! What can you help me with?"})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages,
            tools=TOOLS,
            stream=False,
        )
        choice = resp.choices[0]
        assistant_msg = choice.message

        # Checks
        results.record("Turn 1: HTTP 200 / no exception", True)
        results.record(
            "Turn 1: finish_reason is 'stop'",
            choice.finish_reason == "stop",
            f"got '{choice.finish_reason}'",
        )
        results.record(
            "Turn 1: assistant content non-empty",
            bool(assistant_msg.content and assistant_msg.content.strip()),
            (assistant_msg.content or "")[:80],
        )
        results.record(
            "Turn 1: usage block present",
            resp.usage is not None,
            str(resp.usage) if resp.usage else "missing",
        )

        console.print(
            Panel(assistant_msg.content or "(empty)", title="Assistant", style="blue")
        )
        messages.append({"role": "assistant", "content": assistant_msg.content})

    except (APIConnectionError, APIStatusError) as e:
        results.record("Turn 1: HTTP 200 / no exception", False, str(e))
        console.print(f"[red]Turn 1 failed: {e}[/red]")
        _abort(console)

    # ═══════════════════════════════════════════════════════════════════════
    # TURN 2 — Tool call round-trip (get_weather)
    # ═══════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Turn 2 — Tool call: get_weather"))
    messages.append(
        {"role": "user", "content": "What is the weather like in Paris right now?"}
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        choice = resp.choices[0]
        assistant_msg = choice.message

        called_tool = (
            choice.finish_reason == "tool_calls"
            and assistant_msg.tool_calls
            and len(assistant_msg.tool_calls) > 0
        )
        results.record(
            "Turn 2: finish_reason is 'tool_calls'",
            choice.finish_reason == "tool_calls",
            f"got '{choice.finish_reason}'",
        )
        results.record("Turn 2: tool_calls list present", called_tool)

        if not called_tool:
            console.print(
                "[yellow]Server did not issue a tool call — skipping tool round-trip.[/yellow]"
            )
            # Still add assistant message and continue
            messages.append(
                {"role": "assistant", "content": assistant_msg.content or ""}
            )
        else:
            tool_call = assistant_msg.tool_calls[0]
            fn_name = tool_call.function.name
            fn_args_raw = tool_call.function.arguments

            results.record(
                "Turn 2: correct tool name (get_weather)",
                fn_name == "get_weather",
                f"got '{fn_name}'",
            )

            # Parse args
            try:
                fn_args = json.loads(fn_args_raw)
                args_valid = True
            except json.JSONDecodeError:
                fn_args = {}
                args_valid = False
            results.record(
                "Turn 2: tool arguments are valid JSON", args_valid, fn_args_raw[:80]
            )

            console.print(f"[yellow]Tool call:[/yellow] {fn_name}({fn_args})")

            # Execute tool locally
            tool_result = execute_tool(fn_name, fn_args)
            console.print(f"[yellow]Tool result:[/yellow] {tool_result}")

            # Append assistant tool-call message + tool result message
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

            # Follow-up: let assistant incorporate the tool result
            resp2 = client.chat.completions.create(
                model=model,
                messages=[system_msg] + messages,
                tools=TOOLS,
                stream=False,
            )
            choice2 = resp2.choices[0]
            final_content = choice2.message.content or ""

            results.record(
                "Turn 2: follow-up after tool result returns 'stop'",
                choice2.finish_reason == "stop",
                f"got '{choice2.finish_reason}'",
            )
            results.record(
                "Turn 2: follow-up assistant content non-empty",
                bool(final_content.strip()),
                final_content[:80],
            )
            console.print(
                Panel(final_content, title="Assistant (after tool)", style="blue")
            )
            messages.append({"role": "assistant", "content": final_content})

    except (APIConnectionError, APIStatusError) as e:
        results.record("Turn 2: HTTP 200 / no exception", False, str(e))
        console.print(f"[red]Turn 2 failed: {e}[/red]")

    # ═══════════════════════════════════════════════════════════════════════
    # TURN 3 — Second tool call (calculate) to confirm tool infra is stable
    # ═══════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Turn 3 — Tool call: calculate"))
    messages.append(
        {"role": "user", "content": "Can you calculate (123 * 456) + 789 for me?"}
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        choice = resp.choices[0]
        assistant_msg = choice.message
        called_tool = choice.finish_reason == "tool_calls" and bool(
            assistant_msg.tool_calls
        )

        results.record("Turn 3: tool_calls for calculate", called_tool)

        if called_tool:
            tool_call = assistant_msg.tool_calls[0]
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            tool_result = execute_tool(fn_name, fn_args)
            console.print(f"[yellow]Tool call:[/yellow] {fn_name}({fn_args})")
            console.print(f"[yellow]Tool result:[/yellow] {tool_result}")

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

            resp2 = client.chat.completions.create(
                model=model,
                messages=[system_msg] + messages,
                tools=TOOLS,
                stream=False,
            )
            final_content = resp2.choices[0].message.content or ""
            console.print(
                Panel(final_content, title="Assistant (after tool)", style="blue")
            )
            messages.append({"role": "assistant", "content": final_content})
        else:
            # Model answered directly
            content = assistant_msg.content or ""
            console.print(Panel(content, title="Assistant", style="blue"))
            messages.append({"role": "assistant", "content": content})

    except (APIConnectionError, APIStatusError) as e:
        results.record("Turn 3: HTTP 200 / no exception", False, str(e))
        console.print(f"[red]Turn 3 failed: {e}[/red]")

    # ═══════════════════════════════════════════════════════════════════════
    # TURN 4 — Streaming (SSE) reply
    # ═══════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Turn 4 — Streaming (SSE)"))
    messages.append(
        {"role": "user", "content": "Summarize what we talked about in 2-3 sentences."}
    )

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages,
            tools=TOOLS,
            stream=True,
        )

        collected_text = ""
        chunk_count = 0
        finish_reason = None
        first_chunk_time = None
        t0 = time.time()

        console.print("[dim]Streaming: [/dim]", end="")
        for chunk in stream:
            if first_chunk_time is None:
                first_chunk_time = time.time() - t0

            chunk_count += 1
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                collected_text += delta.content
                console.print(delta.content, end="", highlight=False)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        console.print()  # newline after stream
        console.print(
            f"[dim]Chunks received: {chunk_count} | "
            f"First chunk latency: {first_chunk_time:.3f}s | "
            f"finish_reason: {finish_reason}[/dim]"
        )

        results.record(
            "Turn 4: streaming received chunks",
            chunk_count > 1,
            f"{chunk_count} chunks",
        )
        results.record(
            "Turn 4: streamed content non-empty",
            bool(collected_text.strip()),
            collected_text[:80],
        )
        results.record(
            "Turn 4: stream finish_reason is 'stop'",
            finish_reason == "stop",
            f"got '{finish_reason}'",
        )
        results.record(
            "Turn 4: first chunk latency < 10s",
            first_chunk_time is not None and first_chunk_time < 10,
            f"{first_chunk_time:.3f}s" if first_chunk_time else "N/A",
        )

        messages.append({"role": "assistant", "content": collected_text})

    except (APIConnectionError, APIStatusError) as e:
        results.record("Turn 4: streaming / no exception", False, str(e))
        console.print(f"[red]Turn 4 (streaming) failed: {e}[/red]")

    # ═══════════════════════════════════════════════════════════════════════
    # TURN 5 — Streaming tool call
    # ═══════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Turn 5 — Streaming tool call"))
    messages.append(
        {
            "role": "user",
            "content": "One more thing: what's the weather in Tokyo? (stream this)",
        }
    )

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
        )

        # Accumulate streamed tool call
        finish_reason = None
        tool_call_chunks: dict[int, dict] = {}
        text_chunks = ""

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta

            if delta.content:
                text_chunks += delta.content

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

        got_streaming_tool_call = finish_reason == "tool_calls" and bool(
            tool_call_chunks
        )
        results.record(
            "Turn 5: streaming tool call detected",
            got_streaming_tool_call,
            f"finish_reason={finish_reason}, tool_calls={len(tool_call_chunks)}",
        )

        if got_streaming_tool_call:
            tc = tool_call_chunks[0]
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            tool_result = execute_tool(fn_name, fn_args)
            console.print(f"[yellow]Streamed tool call:[/yellow] {fn_name}({fn_args})")
            console.print(f"[yellow]Tool result:[/yellow] {tool_result}")

            results.record(
                "Turn 5: streamed tool name correct (get_weather)",
                fn_name == "get_weather",
                f"got '{fn_name}'",
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": text_chunks or None,
                    "tool_calls": list(tool_call_chunks.values()),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                }
            )

            # Final follow-up (non-streaming)
            resp_final = client.chat.completions.create(
                model=model,
                messages=[system_msg] + messages,
                stream=False,
            )
            final_text = resp_final.choices[0].message.content or ""
            console.print(
                Panel(final_text, title="Assistant (after streamed tool)", style="blue")
            )

    except (APIConnectionError, APIStatusError) as e:
        results.record("Turn 5: streaming tool call / no exception", False, str(e))
        console.print(f"[red]Turn 5 failed: {e}[/red]")

    # ═══════════════════════════════════════════════════════════════════════
    # TURN 6 — JSON response format (response_format)
    # ═══════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Turn 6 — JSON response format"))
    messages.append(
        {
            "role": "user",
            "content": "Return a JSON object with keys 'name' and 'role' for a developer.",
        }
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages,
            response_format={"type": "json_object"},
            stream=False,
        )
        choice = resp.choices[0]
        assistant_msg = choice.message

        results.record("Turn 6: HTTP 200 / no exception", True)
        results.record(
            "Turn 6: finish_reason is 'stop'",
            choice.finish_reason == "stop",
            f"got '{choice.finish_reason}'",
        )

        response_text = assistant_msg.content or ""
        console.print(Panel(response_text, title="Assistant (JSON)", style="blue"))

        # Verify response is valid JSON
        try:
            parsed_json = json.loads(response_text)
            results.record(
                "Turn 6: response is valid JSON", True, str(parsed_json)[:50]
            )
            has_required_keys = "name" in parsed_json and "role" in parsed_json
            results.record(
                "Turn 6: JSON has 'name' and 'role' keys",
                has_required_keys,
                str(parsed_json)[:50],
            )
        except json.JSONDecodeError as e:
            results.record("Turn 6: response is valid JSON", False, str(e))
            results.record(
                "Turn 6: JSON has 'name' and 'role' keys", False, "parse failed"
            )

        messages.append({"role": "assistant", "content": response_text})

    except (APIConnectionError, APIStatusError) as e:
        results.record("Turn 6: HTTP 200 / no exception", False, str(e))
        console.print(f"[red]Turn 6 failed: {e}[/red]")
        _abort(console)

    # ═══════════════════════════════════════════════════════════════════════
    # TURN 7 — Continue conversation after JSON response (session integrity)
    # ═══════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Turn 7 — Continue after JSON response"))
    messages.append(
        {"role": "user", "content": "What is my name from the JSON I gave you?"}
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg] + messages,
            stream=False,
        )
        choice = resp.choices[0]
        assistant_msg = choice.message

        results.record("Turn 7: HTTP 200 / no exception", True)
        results.record(
            "Turn 7: finish_reason is 'stop'",
            choice.finish_reason == "stop",
            f"got '{choice.finish_reason}'",
        )

        response_text = assistant_msg.content or ""
        console.print(Panel(response_text, title="Assistant (follow-up)", style="blue"))

        # Verify model can "see" the JSON response from Turn 6
        # The response should reference something from the JSON
        response_lower = response_text.lower()
        # Check if response mentions something related to the JSON keys
        # (e.g., "name", "developer", or the actual value from Turn 6)
        references_json = any(kw in response_lower for kw in ["name", "developer"])
        results.record(
            "Turn 7: conversation continued (references Turn 6)",
            references_json,
            response_text[:80],
        )

        messages.append({"role": "assistant", "content": response_text})

    except (APIConnectionError, APIStatusError) as e:
        results.record("Turn 7: HTTP 200 / no exception", False, str(e))
        console.print(f"[red]Turn 7 failed: {e}[/red]")
        _abort(console)


def _abort(console: Console):
    console.print("[red]Aborting remaining tests.[/red]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server test client")
    parser.add_argument(
        "--base-url",
        default="http://localhost:5005/v1",
        help="Base URL of the OpenAI-compatible server (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="Model name to use for requests (default: deepseek-chat)",
    )
    parser.add_argument(
        "--api-key",
        default="not-needed",
        help="API key (default: 'not-needed', for servers that don't require auth)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=360.0,
        help="Request timeout in seconds (default: 360)",
    )
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel(
            f"[bold]OpenAI-compatible server test[/bold]\n"
            f"Base URL : {args.base_url}\n"
            f"Model    : {args.model}\n"
            f"Timeout  : {args.timeout}s",
            title="Config",
            style="cyan",
        )
    )

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    results = TestResults()
    run_tests(client, args.model, console, results)
    results.summary(console)


if __name__ == "__main__":
    main()
