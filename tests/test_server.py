"""Tests for server-side request handling helpers."""

from claude_relay.config import ProxyConfig
from claude_relay.server import (
    _cap_max_completion_tokens,
    _completion_token_margin,
    _context_overflow_retry_from_error,
)


def test_completion_token_margin_scales_with_messages_and_tools():
    assert _completion_token_margin(num_messages=1, num_tools=0) == 130
    assert _completion_token_margin(num_messages=41, num_tools=0) == 210
    assert _completion_token_margin(num_messages=2, num_tools=50) == 1132
    assert _completion_token_margin(num_messages=2, num_tools=134) == 2812


def test_cap_max_completion_tokens_uses_dynamic_margin():
    config = ProxyConfig(min_completion_tokens=4096)

    capped, margin = _cap_max_completion_tokens(
        64000,
        ctx_limit=65536,
        input_tokens=1537,
        num_messages=2,
        num_tools=0,
        config=config,
    )

    assert margin == 132
    assert capped == 63867


def test_cap_max_completion_tokens_keeps_lower_request():
    config = ProxyConfig(min_completion_tokens=4096)

    capped, margin = _cap_max_completion_tokens(
        32000,
        ctx_limit=65536,
        input_tokens=1537,
        num_messages=2,
        num_tools=50,
        config=config,
    )

    assert margin == 1132
    assert capped == 32000


def test_cap_max_completion_tokens_covers_many_tool_overhead():
    config = ProxyConfig(min_completion_tokens=4096)

    capped, margin = _cap_max_completion_tokens(
        64000,
        ctx_limit=65536,
        input_tokens=53479,
        num_messages=2,
        num_tools=134,
        config=config,
    )

    assert margin == 2812
    assert capped == 9245


def test_cap_max_completion_tokens_respects_min_completion_floor():
    config = ProxyConfig(min_completion_tokens=4096)

    capped, margin = _cap_max_completion_tokens(
        64000,
        ctx_limit=65536,
        input_tokens=64000,
        num_messages=2,
        num_tools=0,
        config=config,
    )

    assert margin == 132
    assert capped == 4096


def test_non_200_retry_detects_completion_token_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = "max_completion_tokens=250000 cannot be greater than max_model_len=202,752"

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "completion_limit"
    assert retry.ctx_limit == 202752
    assert retry.input_tokens is None
    assert retry.max_completion_tokens == 202652


def test_non_200_retry_uses_available_context_for_completion_token_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = "max_completion_tokens=250000 cannot be greater than max_model_len=202,752"

    retry = _context_overflow_retry_from_error(error_body, config, estimated_input_tokens=139000)

    assert retry is not None
    assert retry.reason == "completion_limit"
    assert retry.ctx_limit == 202752
    assert retry.input_tokens == 139000
    assert retry.max_completion_tokens == 63652


def test_non_200_retry_detects_vllm_context_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "This model's maximum context length is 202752 tokens. "
        "However, you requested 64000 output tokens and your prompt contains 139000 input tokens."
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "context_limit"
    assert retry.ctx_limit == 202752
    assert retry.input_tokens == 139000
    assert retry.max_completion_tokens == 63652


def test_non_200_retry_detects_vllm_at_least_context_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "This model's maximum context length is 65536 tokens. "
        "However, you requested 64000 output tokens and your prompt contains at least 1537 input tokens, "
        "for a total of at least 65537 tokens. Please reduce the length of the input prompt or the number "
        "of requested output tokens. (parameter=input_tokens, value=1537)"
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "context_limit"
    assert retry.ctx_limit == 65536
    assert retry.input_tokens == 1537
    assert retry.max_completion_tokens == 63899


def test_non_200_retry_detects_openai_compatible_context_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "This model's maximum context length is 202752 tokens. "
        "However, you requested 203000 tokens (139000 in the messages, 64000 in the completion)."
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "context_limit"
    assert retry.ctx_limit == 202752
    assert retry.input_tokens == 139000
    assert retry.max_completion_tokens == 63652


def test_non_200_retry_respects_min_completion_token_floor():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "This model's maximum context length is 202752 tokens. "
        "However, you requested 64000 output tokens and your prompt contains 202000 input tokens."
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.max_completion_tokens == 4096


def test_non_200_retry_ignores_unrelated_errors():
    config = ProxyConfig(min_completion_tokens=4096)

    retry = _context_overflow_retry_from_error("backend is unavailable", config)

    assert retry is None
