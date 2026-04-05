from tool_parser import clean_text_response, extract_tool_calls


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
