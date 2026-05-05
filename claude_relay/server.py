"""aiohttp server: POST /v1/messages handler."""

import json
import os
import logging
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import AsyncIterator

import aiohttp
from aiohttp import web

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug")

from .config import ProxyConfig
from .convert_request import convert_request
from .convert_stream import StreamResult, convert_openai_stream_to_anthropic
from .sse import SSEEvent, parse_sse_stream
from .backend import send_to_backend, detect_backend, get_state
from .image_agent import has_images, strip_and_cache_images, image_agent_stream, ImageCache
from .normalize import normalize_request

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextOverflowRetry:
    max_completion_tokens: int
    ctx_limit: int
    input_tokens: int | None
    reason: str


def _summarize_messages(messages: list) -> str:
    """One-line summary of message roles and content types."""
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(f"{role}(text:{len(content)})")
        elif isinstance(content, list):
            types = []
            for b in content:
                bt = b.get("type", "?")
                if bt == "image":
                    types.append("img")
                elif bt == "tool_result":
                    sub = b.get("content")
                    has_img = False
                    if isinstance(sub, list):
                        has_img = any(isinstance(x, dict) and x.get("type") == "image" for x in sub)
                    types.append(f"tool_result{'(img)' if has_img else ''}")
                elif bt == "tool_use":
                    types.append(f"tool_use:{b.get('name', '?')}")
                elif bt == "thinking":
                    types.append("think")
                elif bt == "text":
                    types.append(f"text:{len(b.get('text', ''))}")
                else:
                    types.append(bt)
            parts.append(f"{role}[{','.join(types)}]")
        else:
            parts.append(role)
    return " | ".join(parts[-6:])  # last 6 messages


def validate_request(body: dict, config: ProxyConfig) -> None:
    """Validate Anthropic API request body."""
    # Messages validation
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise web.HTTPBadRequest(text='{"error":"messages must be a non-empty array"}')

    # Max tokens validation
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise web.HTTPBadRequest(text='{"error":"max_tokens must be a positive integer"}')

    # Tools length validation
    tools = body.get("tools", [])
    if len(tools) > config.max_tools:
        raise web.HTTPBadRequest(text=f'{{"error":"tools array exceeds maximum length of {config.max_tools}"}}')

    # Per-image size validation
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image":
                    src = block.get("source", {})
                    data = src.get("data", "")
                    if len(data) > config.max_image_b64_chars:
                        raise web.HTTPRequestEntityTooLarge(
                            text=f'{{"error":"image exceeds maximum size of {config.max_image_b64_chars} characters"}}'
                        )


def _tool_names(tools: list[dict]) -> list[str]:
    names = []
    for tool in tools:
        if isinstance(tool, dict):
            names.append(str(tool.get("name") or "?"))
    return names


def _openai_tool_names(tools: list[dict]) -> list[str]:
    names = []
    for tool in tools:
        if isinstance(tool, dict):
            func = tool.get("function") or {}
            names.append(str(func.get("name") or "?"))
    return names


def _parse_token_count(value: str) -> int:
    return int(value.replace(",", ""))


def _completion_token_margin(num_messages: int, num_tools: int) -> int:
    """Estimate backend chat-template overhead not captured by local token counting."""
    return 128 + (2 * num_messages) + (20 * num_tools)


def _cap_max_completion_tokens(
    requested_max: int,
    *,
    ctx_limit: int,
    input_tokens: int,
    num_messages: int,
    num_tools: int,
    config: ProxyConfig,
) -> tuple[int, int]:
    margin = _completion_token_margin(num_messages, num_tools)
    capped_max = max(config.min_completion_tokens, ctx_limit - input_tokens - margin)
    return min(requested_max, capped_max), margin


def _context_overflow_retry_from_error(
    error_body: str,
    config: ProxyConfig,
    estimated_input_tokens: int | None = None,
) -> ContextOverflowRetry | None:
    """Detect non-streaming backend context errors and compute the retry budget."""
    flags = re.IGNORECASE | re.DOTALL

    # vLLM/SGLang output-only limit:
    # "max_completion_tokens=X cannot be greater than max_model_len=Y"
    match = re.search(r"max_completion_tokens\s*=\s*([\d,]+).*?max_model_len\D*([\d,]+)", error_body, flags)
    if match:
        ctx_limit = _parse_token_count(match.group(2))
        available_tokens = ctx_limit - 100
        if estimated_input_tokens is not None:
            available_tokens -= estimated_input_tokens
        return ContextOverflowRetry(
            max_completion_tokens=max(config.min_completion_tokens, available_tokens),
            ctx_limit=ctx_limit,
            input_tokens=estimated_input_tokens,
            reason="completion_limit",
        )

    # vLLM combined input+output limit:
    # "maximum context length is X. However, you requested Y output tokens and your prompt contains Z input tokens"
    match = re.search(
        r"maximum context length is\s*([\d,]+).*?"
        r"requested\s*([\d,]+)\s*output tokens.*?"
        r"prompt contains(?:\s+at least)?\s*([\d,]+)\s*input tokens",
        error_body,
        flags,
    )
    if not match:
        # OpenAI-compatible/SGLang shape:
        # "maximum context length is X tokens. However, you requested Y tokens (Z in the messages, W in the completion)."
        match = re.search(
            r"maximum context length is\s*([\d,]+).*?"
            r"requested\s*([\d,]+)\s*tokens.*?"
            r"\(([\d,]+)\s*in (?:the )?(?:messages|prompt).*?([\d,]+)\s*in (?:the )?(?:completion|output)",
            error_body,
            flags,
        )

    if match:
        ctx_limit = _parse_token_count(match.group(1))
        input_tokens = _parse_token_count(match.group(3))
        return ContextOverflowRetry(
            max_completion_tokens=max(config.min_completion_tokens, ctx_limit - input_tokens - 100),
            ctx_limit=ctx_limit,
            input_tokens=input_tokens,
            reason="context_limit",
        )

    return None


def _tool_state_summary(tool_states: dict[str, dict]) -> list[dict]:
    summary = []
    for index, state in sorted(tool_states.items(), key=lambda item: item[0]):
        args = state.get("arguments", "")
        item = {
            "index": index,
            "id": state.get("id"),
            "name": state.get("name"),
            "chunks": state.get("chunks", 0),
            "arguments_chars": len(args),
            "arguments_preview": args[:500],
        }
        if args:
            try:
                json.loads(args)
                item["arguments_json_valid"] = True
            except json.JSONDecodeError as e:
                item["arguments_json_valid"] = False
                item["arguments_json_error"] = str(e)
        summary.append(item)
    return summary


async def _debug_sse_stream(
    sse_events: AsyncIterator[SSEEvent],
    req_id: str,
    *,
    enabled: bool,
    expected_tool_names: list[str] | None = None,
) -> AsyncIterator[SSEEvent]:
    """Optionally dump raw backend SSE and summarize tool-call parser signals."""
    if not enabled:
        async for event in sse_events:
            yield event
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)
    stream_path = os.path.join(DEBUG_DIR, f"{req_id}_backend_stream.ndjson")
    summary_path = os.path.join(DEBUG_DIR, f"{req_id}_tool_debug.json")
    delta_keys: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    unknown_tool_names: Counter[str] = Counter()
    tail = deque(maxlen=12)
    counts = Counter()
    content_chars = 0
    reasoning_chars = 0
    tool_states: dict[str, dict] = {}
    tool_like_content = []
    expected_tools = set(expected_tool_names or [])
    stream_file = None

    try:
        stream_file = open(stream_path, "w")
        log.info("[%s] tool_debug: dumping backend SSE to %s", req_id, stream_path)
    except OSError as e:
        log.warning("[%s] tool_debug: failed to open stream dump %s: %s", req_id, stream_path, e)

    try:
        async for event in sse_events:
            counts["events"] += 1
            seq = counts["events"]
            record = {
                "seq": seq,
                "event": event.event,
                "raw": event.data,
            }

            if event.data == "[DONE]":
                counts["done"] += 1
                tail.append({"seq": seq, "done": True})
                if stream_file:
                    stream_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                yield event
                continue

            event_summary = {"seq": seq, "event": event.event}
            try:
                data = json.loads(event.data)
            except (json.JSONDecodeError, TypeError) as e:
                counts["bad_json"] += 1
                event_summary["bad_json"] = str(e)
                tail.append(event_summary)
                if stream_file:
                    stream_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                yield event
                continue

            choices = data.get("choices") or []
            if choices:
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                finish_reason = choice.get("finish_reason")
                keys = sorted(delta.keys())
                for key in keys:
                    delta_keys[key] += 1

                if finish_reason:
                    finish_reasons[str(finish_reason)] += 1

                content = delta.get("content")
                if content:
                    content_text = str(content)
                    content_chars += len(content_text)
                    event_summary["content_preview"] = content_text[:200]
                    lower_content = content_text.lower()
                    looks_like_tool = any(
                        marker in lower_content
                        for marker in ("tool_call", "function_call", "<tool", "<function", '"arguments"')
                    )
                    if not looks_like_tool and expected_tools:
                        looks_like_tool = any(f"{name}(" in content_text for name in expected_tools)
                    if looks_like_tool:
                        counts["tool_like_content_events"] += 1
                        tool_like_content.append({
                            "seq": seq,
                            "preview": content_text[:300],
                        })

                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if not reasoning and isinstance(delta.get("thinking"), dict):
                    reasoning = delta["thinking"].get("content")
                if reasoning:
                    reasoning_text = str(reasoning)
                    reasoning_chars += len(reasoning_text)
                    event_summary["reasoning_preview"] = reasoning_text[:200]

                tool_calls = delta.get("tool_calls") or []
                if tool_calls:
                    counts["tool_call_events"] += 1
                    counts["tool_call_deltas"] += len(tool_calls)
                    event_summary["tool_calls"] = []
                    for tool_call in tool_calls:
                        func = (tool_call or {}).get("function") or {}
                        name = func.get("name")
                        if name:
                            tool_names[str(name)] += 1
                            if expected_tools and str(name) not in expected_tools:
                                unknown_tool_names[str(name)] += 1
                        args = func.get("arguments")
                        index = str(tool_call.get("index", 0))
                        state = tool_states.setdefault(index, {
                            "id": None,
                            "name": None,
                            "arguments": "",
                            "chunks": 0,
                        })
                        if tool_call.get("id"):
                            state["id"] = tool_call.get("id")
                        if name:
                            state["name"] = name
                        state["chunks"] += 1
                        if isinstance(args, str):
                            state["arguments"] += args
                        event_summary["tool_calls"].append({
                            "index": tool_call.get("index"),
                            "id": tool_call.get("id"),
                            "name": name,
                            "args_chars": len(args) if isinstance(args, str) else 0,
                            "args_preview": args[:200] if isinstance(args, str) else None,
                        })

                event_summary["delta_keys"] = keys
                event_summary["finish_reason"] = finish_reason
                tail.append(event_summary)

                if finish_reason or tool_calls:
                    log.info(
                        "[%s] tool_debug: event #%d finish=%s keys=%s tool_call_deltas=%d",
                        req_id,
                        seq,
                        finish_reason,
                        ",".join(keys),
                        len(tool_calls),
                    )

            if stream_file:
                stream_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            yield event
    finally:
        if stream_file:
            stream_file.close()

        summary = {
            "request_id": req_id,
            "events": counts["events"],
            "done_events": counts["done"],
            "bad_json_events": counts["bad_json"],
            "tool_call_events": counts["tool_call_events"],
            "tool_call_deltas": counts["tool_call_deltas"],
            "tool_names": dict(tool_names),
            "expected_tool_names": sorted(expected_tools),
            "unknown_tool_names": dict(unknown_tool_names),
            "tool_calls_by_index": _tool_state_summary(tool_states),
            "finish_reasons": dict(finish_reasons),
            "delta_keys": dict(delta_keys),
            "content_chars": content_chars,
            "reasoning_chars": reasoning_chars,
            "tool_like_content_events": counts["tool_like_content_events"],
            "tool_like_content": tool_like_content[-5:],
            "stream_path": stream_path if stream_file else None,
            "tail": list(tail),
        }

        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except OSError as e:
            log.warning("[%s] tool_debug: failed to write summary %s: %s", req_id, summary_path, e)

        if finish_reasons.get("tool_calls", 0) and counts["tool_call_deltas"] == 0:
            log.error(
                "[%s] tool_debug: backend reported finish_reason=tool_calls but emitted no delta.tool_calls; summary=%s",
                req_id,
                summary_path,
            )
        elif unknown_tool_names:
            log.error(
                "[%s] tool_debug: backend emitted unknown tool name(s) %s; expected=%s summary=%s",
                req_id,
                dict(unknown_tool_names),
                sorted(expected_tools),
                summary_path,
            )
        elif any(item.get("arguments_json_valid") is False for item in summary["tool_calls_by_index"]):
            log.error("[%s] tool_debug: backend emitted invalid tool argument JSON; summary=%s", req_id, summary_path)
        elif counts["tool_like_content_events"] and counts["tool_call_deltas"] == 0:
            log.warning(
                "[%s] tool_debug: backend emitted tool-looking plain content but no delta.tool_calls; summary=%s",
                req_id,
                summary_path,
            )
        else:
            log.info(
                "[%s] tool_debug: events=%d tool_call_deltas=%d finish=%s summary=%s",
                req_id,
                counts["events"],
                counts["tool_call_deltas"],
                dict(finish_reasons),
                summary_path,
            )


async def handle_messages(request: web.Request) -> web.StreamResponse:
    config: ProxyConfig = request.app["config"]
    session: aiohttp.ClientSession = request.app["session"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    validate_request(body, config)

    # Session ID for image agent: use user_id if provided, otherwise req_id (not "default") to prevent collisions
    req_id = os.urandom(3).hex()
    session_id = (body.get("metadata") or {}).get("user_id", req_id)
    use_image_agent = config.image_agent_enabled and has_images(body)

    msgs = body.get("messages", [])
    num_tools = len(body.get("tools", []))
    tool_names = _tool_names(body.get("tools", []))
    system_len = len(body.get("system", "")) if isinstance(body.get("system"), str) else sum(
        len(b.get("text", "")) for b in body.get("system", []) if isinstance(b, dict)
    )
    thinking = body.get("thinking", {})
    thinking_info = f"budget={thinking.get('budget_tokens')}" if thinking.get("type") == "enabled" else "off"

    log.info(
        "[%s] >>> POST /v1/messages msgs=%d tools=%d(%s) system=%d thinking=%s image_agent=%s",
        req_id, len(msgs), num_tools,
        ",".join(tool_names[:5]) + ("..." if len(tool_names) > 5 else ""),
        system_len, thinking_info, use_image_agent,
    )
    log.info("[%s]     last_msgs: %s", req_id, _summarize_messages(msgs))

    # Dump full request body for debugging (opt-in via --dump-requests)
    if config.dump_requests:
        try:
            debug_file = os.path.join(DEBUG_DIR, f"{req_id}_anthropic.json")
            with open(debug_file, "w") as f:
                json.dump(body, f, indent=2, ensure_ascii=False, default=str)
            log.info("[%s]     dumped anthropic body to %s", req_id, debug_file)
        except Exception as e:
            log.warning("[%s]     failed to dump body: %s", req_id, e)

    if use_image_agent:
        strip_and_cache_images(body, session_id, request.app["image_cache"])

    # Normalize request for KV cache stability
    body, norm_changes = normalize_request(body, config)
    if norm_changes:
        log.info("[%s]     normalize: %s", req_id, "; ".join(norm_changes))

    # Convert Anthropic → OpenAI
    openai_body = convert_request(body)

    # Force analyzeImage tool_choice when image agent is active
    if use_image_agent and config.force_vision:
        openai_body["tool_choice"] = {"type": "function", "function": {"name": "analyzeImage"}}
        log.info("[%s]     force_vision: tool_choice set to analyzeImage", req_id)

    openai_msgs = openai_body.get("messages", [])
    openai_tool_names = _openai_tool_names(openai_body.get("tools", []))
    tool_debug_enabled = config.tool_debug and bool(openai_tool_names)
    log.info("[%s]     openai: %d msgs, model=%s, max_tokens=%s, tools=%d tool_choice=%s",
             req_id, len(openai_msgs), openai_body.get("model"),
             openai_body.get("max_completion_tokens"),
             len(openai_tool_names), openai_body.get("tool_choice"))
    if openai_tool_names:
        log.info(
            "[%s]     openai_tools: %s%s",
            req_id,
            ",".join(openai_tool_names[:12]),
            "..." if len(openai_tool_names) > 12 else "",
        )
    if tool_debug_enabled:
        log.info("[%s]     tool_debug enabled for this request", req_id)

    if config.dump_requests:
        try:
            debug_file = os.path.join(DEBUG_DIR, f"{req_id}_openai.json")
            with open(debug_file, "w") as f:
                json.dump(openai_body, f, indent=2, ensure_ascii=False, default=str)
            log.info("[%s]     dumped openai body to %s", req_id, debug_file)
        except Exception as e:
            log.warning("[%s]     failed to dump openai body: %s", req_id, e)

    # Count input tokens via tiktoken for message_start
    input_token_count = len(_get_tiktoken().encode(_serialize_for_counting(body)))
    state = get_state()
    requested_max = openai_body.get("max_completion_tokens")
    if isinstance(requested_max, int) and requested_max > 0:
        capped_max, margin = _cap_max_completion_tokens(
            requested_max,
            ctx_limit=state.context_limit,
            input_tokens=input_token_count,
            num_messages=len(openai_msgs),
            num_tools=len(openai_tool_names),
            config=config,
        )
        if capped_max < requested_max:
            log.warning(
                "[%s] Capping max_completion_tokens %d → %d (input≈%d, limit=%d, margin=%d)",
                req_id,
                requested_max,
                capped_max,
                input_token_count,
                state.context_limit,
                margin,
            )
            openai_body["max_completion_tokens"] = capped_max

    # Send main streaming request to backend (with auto-retry on context overflow)
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            resp = await send_to_backend(session, config, openai_body, req_id=req_id)
        except Exception as e:
            log.error("[%s] Backend request failed: %s", req_id, e)
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": str(e)}},
                status=502,
            )

        if resp.status != 200:
            error_body = await resp.text()
            log.error("[%s] Backend returned %d: %s", req_id, resp.status, error_body[:500])

            retry = _context_overflow_retry_from_error(error_body, config, input_token_count)
            if retry is not None and attempt < max_retries:
                if retry.reason == "completion_limit":
                    log.warning("[%s] max_completion_tokens exceeds model limit: input≈%d, limit=%d → retrying with max_completion_tokens=%d (attempt %d)",
                                req_id, retry.input_tokens or 0, retry.ctx_limit, retry.max_completion_tokens, attempt + 1)
                else:
                    log.warning("[%s] Context overflow on non-200: input=%d, limit=%d → retrying with max_completion_tokens=%d (attempt %d)",
                                req_id, retry.input_tokens, retry.ctx_limit, retry.max_completion_tokens, attempt + 1)
                resp.close()
                openai_body["max_completion_tokens"] = retry.max_completion_tokens
                continue

            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": error_body}},
                status=resp.status,
            )

        log.info("[%s]     backend responded 200, streaming...", req_id)

        # For non-image-agent: peek at first SSE events to detect errors/empty responses
        if not use_image_agent:
            sse_events = _debug_sse_stream(
                parse_sse_stream(resp),
                req_id,
                enabled=tool_debug_enabled,
                expected_tool_names=openai_tool_names,
            )
            buffered = []
            has_content = False
            async for event in sse_events:
                buffered.append(event)
                if event.data == "[DONE]":
                    break
                try:
                    data = json.loads(event.data)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta:
                            has_content = True
                            break
                except (json.JSONDecodeError, TypeError):
                    pass

            if not has_content:
                # Check if backend returned a context-overflow error
                error_msg = "Backend returned empty response"
                context_overflow = False
                for ev in buffered:
                    if ev.data == "[DONE]":
                        continue
                    try:
                        ev_data = json.loads(ev.data)
                        if "error" in ev_data:
                            error_msg = ev_data["error"].get("message", str(ev_data["error"]))
                            m = re.search(r"(\d+) tokens from the input.*?(\d+) tokens for the completion", error_msg)
                            if m:
                                context_overflow = True
                                input_tokens = int(m.group(1))
                                m2 = re.search(r"maximum context length of (\d+)", error_msg)
                                ctx_limit = int(m2.group(1)) if m2 else get_state().context_limit
                                break
                    except (json.JSONDecodeError, TypeError):
                        pass

                if context_overflow and attempt < max_retries:
                    new_max = max(config.min_completion_tokens, ctx_limit - input_tokens - 100)
                    log.warning("[%s] Context overflow: input=%d, limit=%d → retrying with max_completion_tokens=%d (attempt %d)",
                                req_id, input_tokens, ctx_limit, new_max, attempt + 1)
                    openai_body["max_completion_tokens"] = new_max
                    continue

                log.error("[%s] EMPTY/ERROR response from backend — got %d events, error: %s",
                          req_id, len(buffered), error_msg)
                for i, ev in enumerate(buffered):
                    log.error("[%s]   event[%d]: %s", req_id, i, ev.data[:500])
                return web.json_response(
                    {"type": "error", "error": {"type": "overloaded_error",
                     "message": error_msg}},
                    status=529,
                )

            async def _replay_and_continue():
                for ev in buffered:
                    yield ev
                async for ev in sse_events:
                    yield ev

            sse_source = _replay_and_continue()
        else:
            sse_source = None

        break  # success

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    bytes_sent = 0
    try:
        if use_image_agent:
            async for chunk in image_agent_stream(resp, openai_body, session_id, session, config, request.app["image_cache"], req_id):
                await response.write(chunk)
                bytes_sent += len(chunk)
        else:
            async for chunk in convert_openai_stream_to_anthropic(sse_source, req_id=req_id, input_tokens=input_token_count):
                await response.write(chunk)
                bytes_sent += len(chunk)
    except Exception as e:
        err_str = str(e).lower()
        if "closing transport" in err_str or "connection reset" in err_str:
            log.warning("[%s] Client disconnected during streaming: %s", req_id, e)
        else:
            log.error("[%s] Streaming error: %s", req_id, e, exc_info=True)

    log.info("[%s] <<< done, %d bytes sent", req_id, bytes_sent)
    await response.write_eof()
    return response


def _serialize_for_counting(body: dict) -> str:
    """Serialize Anthropic count_tokens body to plain text for tokenization."""
    parts = []

    # System prompt
    system = body.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))

    # Messages
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                bt = block.get("type", "")
                if bt == "text":
                    parts.append(block.get("text", ""))
                elif bt == "tool_use":
                    parts.append(json.dumps(block.get("input", {}), ensure_ascii=False))
                elif bt == "tool_result":
                    sub = block.get("content", "")
                    if isinstance(sub, str):
                        parts.append(sub)
                    elif isinstance(sub, list):
                        for sb in sub:
                            if isinstance(sb, dict):
                                if sb.get("type") == "text":
                                    parts.append(sb.get("text", ""))
                                elif sb.get("type") == "image":
                                    parts.append("[image]")  # ~1600 tokens per image, rough estimate
                elif bt == "thinking":
                    parts.append(block.get("thinking", ""))

    # Tools
    for tool in body.get("tools", []):
        parts.append(tool.get("name", ""))
        parts.append(tool.get("description", ""))
        schema = tool.get("input_schema")
        if schema:
            parts.append(json.dumps(schema, ensure_ascii=False))

    return "\n".join(parts)


_tiktoken_enc = None

def _get_tiktoken():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_enc


async def handle_count_tokens(request: web.Request) -> web.Response:
    """POST /v1/messages/count_tokens — token counting for Claude Code /context."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    text = _serialize_for_counting(body)
    enc = _get_tiktoken()
    count = len(enc.encode(text))
    # Debug: log what we're counting
    num_msgs = len(body.get("messages", []))
    num_tools = len(body.get("tools", []))
    sys_len = len(body.get("system", "")) if isinstance(body.get("system"), str) else sum(
        len(b.get("text", "")) for b in body.get("system", []) if isinstance(b, dict)
    )
    log.debug("count_tokens: msgs=%d tools=%d system=%d text_len=%d → %d tokens",
              num_msgs, num_tools, sys_len, len(text), count)
    return web.json_response({"input_tokens": count})


async def health_check(request: web.Request) -> web.Response:
    state = get_state()
    config: ProxyConfig = request.app["config"]
    return web.json_response({
        **state.info(),
        "backend_url": config.backend_url,
        "image_agent_enabled": config.image_agent_enabled,
        "vision_url": config.vision_url,
        "vision_model": config.vision_model,
    })


def _cleanup_debug_files(max_age_hours: int) -> int:
    """Delete debug dump files older than max_age_hours. Returns count deleted."""
    import time
    deleted = 0
    now = time.time()
    max_age_sec = max_age_hours * 3600

    if not os.path.isdir(DEBUG_DIR):
        return 0

    for fname in os.listdir(DEBUG_DIR):
        if not fname.endswith((".json", ".ndjson")):
            continue
        fpath = os.path.join(DEBUG_DIR, fname)
        try:
            mtime = os.path.getmtime(fpath)
            if now - mtime > max_age_sec:
                os.remove(fpath)
                deleted += 1
        except OSError:
            pass  # race: file may have been deleted concurrently

    return deleted


async def on_startup(app: web.Application):
    app["session"] = aiohttp.ClientSession()
    config: ProxyConfig = app["config"]
    app["image_cache"] = ImageCache(config.image_cache_max_size, config.image_cache_ttl)

    if config.dump_requests or config.tool_debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    # Cleanup old debug files before starting
    if config.debug_max_age_hours is not None:
        deleted = _cleanup_debug_files(config.debug_max_age_hours)
        if deleted > 0:
            log.info("Cleaned up %d old debug file(s) from %s", deleted, DEBUG_DIR)

    await detect_backend(app["session"], config.backend_url)


async def on_cleanup(app: web.Application):
    await app["session"].close()


def create_app(config: ProxyConfig) -> web.Application:
    app = web.Application(client_max_size=config.client_max_size)
    app["config"] = config
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/messages", handle_messages)
    app.router.add_post("/v1/messages/count_tokens", handle_count_tokens)
    app.router.add_get("/health", health_check)
    return app
