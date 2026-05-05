"""Tests for Anthropic -> OpenAI request conversion."""

from claude_relay.convert_request import convert_request


def test_tool_choice_tool_converts_to_openai_function_choice():
    body = {
        "messages": [{"role": "user", "content": "Use the tool."}],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather.",
            "input_schema": {"type": "object"},
        }],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }

    converted = convert_request(body)

    assert converted["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_tool_choice_any_converts_to_required():
    body = {
        "messages": [{"role": "user", "content": "Use a tool."}],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather.",
            "input_schema": {"type": "object"},
        }],
        "tool_choice": {"type": "any"},
    }

    converted = convert_request(body)

    assert converted["tool_choice"] == "required"
