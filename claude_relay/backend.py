import json
import os
import time
import logging
from copy import deepcopy

import aiohttp

from .config import BackendTarget, ProxyConfig

log = logging.getLogger(__name__)

# Mirror server.DEBUG_DIR without importing server (server imports backend).
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug")


DEFAULT_CONTEXT_LIMIT = 131072  # 128k fallback if backend doesn't report


def _lookup_ci(mapping: dict, key: str | None):
    """Case-insensitive dict lookup. Exact match wins, then case-folded."""
    if not key:
        return None
    if key in mapping:
        return mapping[key]
    folded = key.lower()
    for k, v in mapping.items():
        if k.lower() == folded:
            return v
    return None


def _template_family(resolved_model: str | None, target: BackendTarget) -> str:
    """Which chat-template contract the backend speaks. An explicit route
    `template_family` wins; otherwise sniff the model name.

    The concretely resolved model is authoritative — a stale `upstream_model`
    must not override it (else a Kimi-named-but-unserved route would push real
    DeepSeek traffic through Kimi mutation). Only fall back to the configured
    `upstream_model` when resolution is unavailable (probe failed ->
    resolve_model returns "default"/empty). Defaults to "deepseek" (the
    original behavior) for anything unrecognized."""
    declared = (getattr(target, "template_family", "auto") or "auto").lower()
    if declared in ("kimi", "deepseek"):
        return declared
    name = (resolved_model or "").strip()
    if not name or name.lower() == "default":
        name = getattr(target, "upstream_model", "") or ""
    return "kimi" if "kimi" in name.lower() else "deepseek"


def _kimi_reshape_reasoning(body: dict) -> None:
    """Remap Anthropic nested thinking -> flat reasoning_content on prior
    assistant turns so Kimi's `preserve_thinking` has reasoning to replay.
    Kimi templates read `message.reasoning`/`reasoning_content` (a string),
    not Anthropic's nested {"thinking": {"content", "signature"}}. Covers
    tool-call turns (filters on role only). Always drops the nested thinking
    dict (Kimi ignores it); a usable existing reasoning_content string wins."""
    for msg in body.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        thinking = msg.pop("thinking", None)
        existing = msg.get("reasoning_content")
        if isinstance(existing, str) and existing.strip():
            continue
        if isinstance(thinking, dict) and thinking.get("content"):
            msg["reasoning_content"] = thinking["content"]


class BackendState:
    def __init__(self, ttl: int = 30):
        self.model: str | None = None
        self.models: list[str] = []  # all model ids the backend serves
        self.backend_type: str | None = None  # "sglang" or "vllm"
        self.context_limit: int = DEFAULT_CONTEXT_LIMIT
        self.last_check: float = 0
        self.ttl = ttl

    @property
    def stale(self) -> bool:
        return self.model is None or (time.time() - self.last_check) >= self.ttl

    def resolve_model(self, requested: str | None) -> str | None:
        """Match `requested` against served models case-insensitively, returning
        the backend's own casing. Fall back to the first served model when there
        is no match (e.g. a stale or wrong upstream_model in config)."""
        if requested:
            folded = requested.strip().lower()
            for served in self.models:
                if served.lower() == folded:
                    return served
        return self.model

    def info(self) -> dict:
        return {
            "model": self.model,
            "models": list(self.models),
            "backend_type": self.backend_type,
            "context_limit": self.context_limit,
            "last_check_ago": f"{time.time() - self.last_check:.0f}s"
            if self.last_check
            else "never",
        }


_states: dict[str, BackendState] = {}
_ttl = 30


async def detect_backend(session: aiohttp.ClientSession, backend_url: str) -> BackendState:
    state = _states.get(backend_url)
    if state is None:
        state = BackendState(ttl=_ttl)
        _states[backend_url] = state

    if not state.stale:
        return state

    try:
        async with session.get(
            f"{backend_url}/v1/models",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            data = await resp.json()
            served = data.get("data") or []
            valid = [m for m in served if isinstance(m, dict) and m.get("id")]
            if not valid:
                # Don't clobber cached state on an empty/garbage response.
                raise ValueError(f"no models reported by {backend_url}")
            state.models = [m["id"] for m in valid]
            model_info = valid[0]
            state.model = model_info["id"]
            owned_by = model_info.get("owned_by", "").lower()

            # Detect backend type and context limit from model info
            max_model_len = model_info.get("max_model_len")  # vLLM includes this

            if "sglang" in owned_by:
                state.backend_type = "sglang"
            elif "vllm" in owned_by:
                state.backend_type = "vllm"
            else:
                # Fallback: try SGLang-specific endpoint
                try:
                    async with session.get(
                        f"{backend_url}/get_model_info",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as r2:
                        state.backend_type = "sglang" if r2.status == 200 else "vllm"
                except Exception:
                    state.backend_type = "vllm"

            # Query context limit from backend-specific endpoints
            if max_model_len:
                state.context_limit = int(max_model_len)
            elif state.backend_type == "sglang":
                try:
                    async with session.get(
                        f"{backend_url}/get_model_info",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as r2:
                        if r2.status == 200:
                            info = await r2.json()
                            ctx = info.get("max_total_num_tokens") or info.get("context_length")
                            if ctx:
                                state.context_limit = int(ctx)
                except Exception:
                    pass  # keep default

            state.last_check = time.time()
            log.info(
                "backend_url=%s backend=%s model=%s context_limit=%d (cached %ds)",
                backend_url,
                state.backend_type,
                state.model,
                state.context_limit,
                state.ttl,
            )
    except Exception as e:
        log.warning("backend probe failed for %s: %s, using cached values", backend_url, e)
        if not state.model:
            state.model = "default"
            state.backend_type = "vllm"

    return state


def init_state(ttl: int):
    global _states, _ttl
    _ttl = ttl
    _states = {}


def get_state(backend_url: str | None = None) -> BackendState:
    if backend_url is not None:
        return _states.setdefault(backend_url, BackendState(ttl=_ttl))
    if _states:
        return next(iter(_states.values()))
    return BackendState(ttl=_ttl)


def get_states_info() -> dict[str, dict]:
    return {backend_url: state.info() for backend_url, state in _states.items()}


async def send_to_backend(
    session: aiohttp.ClientSession,
    config: ProxyConfig,
    openai_body: dict,
    req_id: str = "",
    backend_target: BackendTarget | None = None,
) -> aiohttp.ClientResponse:
    """Detect backend, remap model, inject sglang kwargs, POST to backend.

    Returns the raw aiohttp response (caller must consume it).
    """
    _r = f"[{req_id}] " if req_id else ""
    request_model = str(openai_body.get("model") or "auto")
    target = backend_target or config.resolve_backend(request_model)
    state = await detect_backend(session, target.backend_url)

    backend_body = deepcopy(openai_body)
    # Match the requested upstream model against what the backend actually serves
    # (case-insensitively), falling back to the first served model when there is
    # no match. This tolerates casing drift and stale upstream_model config.
    requested_model = target.upstream_model
    resolved_model = state.resolve_model(requested_model)
    if requested_model and (resolved_model or "").lower() != requested_model.strip().lower():
        log.warning(
            "%smodel %r not served by %s (serving: %s); falling back to %r",
            _r, requested_model, target.backend_url,
            ", ".join(state.models) or "?", resolved_model,
        )
    backend_body["model"] = resolved_model

    # Apply per-model sampling defaults keyed by the model actually being sent.
    # Only fills params not already set by client (setdefault).
    sampling = _lookup_ci(config.model_sampling, resolved_model)
    if sampling:
        for k, v in sampling.items():
            backend_body.setdefault(k, v)
        log.info("%ssampling: %s (model=%s)", _r, sampling, resolved_model)

    # When thinking is active: use route/effort default. When inactive: send "none"
    # so vLLM skips reasoning compute entirely. This changes the prompt slightly
    # (costing a KV cache refill on toggle) but avoids burning compute on hidden reasoning.
    thinking_active = backend_body.pop("_thinking_active", False)
    family = _template_family(resolved_model, target)
    kwargs = backend_body.setdefault("chat_template_kwargs", {})
    if family == "kimi":
        # Kimi K2.6/K2.7 templates read `thinking` (bool) + `preserve_thinking`
        # (bool), NOT the DeepSeek vars. Strip the DeepSeek keys and the
        # conversion-only top-level fields Kimi ignores, then set Kimi's own.
        for k in ("enable_thinking", "clear_thinking", "reasoning_effort"):
            kwargs.pop(k, None)
        backend_body.pop("reasoning", None)
        backend_body.pop("enable_thinking", None)
        # Always keep thinking ENABLED for Kimi. Setting thinking=false makes the
        # vLLM kimi_k2 reasoning parser fall through to IdentityReasoningParser
        # (pure passthrough); K2.7-Code keeps reasoning anyway, so its raw chain
        # of thought plus a stray </think> token leak into `content`. With
        # thinking=true the parser splits reasoning into delta.reasoning, and the
        # server's `emit_thinking` flag (driven by the client's request) decides
        # whether the client actually sees thinking blocks or they're dropped.
        kwargs["thinking"] = True
        kwargs["preserve_thinking"] = True           # replay prior reasoning; version-independent
        _kimi_reshape_reasoning(backend_body)
        log.info("%skimi thinking=True preserve_thinking=True (client_thinking=%s, route=%s, model=%s)",
                 _r, thinking_active, target.route_name, resolved_model)
    else:
        re = target.reasoning_effort
        if thinking_active:
            kwargs.setdefault("reasoning_effort", re or "max")
            kwargs.setdefault("enable_thinking", True)
        else:
            kwargs.setdefault("reasoning_effort", "none")
        log.info("%sreasoning_effort: %s (route=%s, active=%s)", _r, kwargs["reasoning_effort"], target.route_name, thinking_active)

    # Request usage stats in streaming mode so we can report token counts to Claude Code
    if backend_body.get("stream"):
        backend_body["stream_options"] = {"include_usage": True}

    # Dump the POST-mutation backend body (the openai dump in server.py is taken
    # before this function rewrites model/kwargs/messages, so it cannot show the
    # transform). Reuses the dump_requests flag.
    if config.dump_requests and req_id:
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            with open(os.path.join(DEBUG_DIR, f"{req_id}_backend_request.json"), "w") as f:
                json.dump(backend_body, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            log.warning("%sfailed to dump backend request: %s", _r, e)

    url = f"{target.backend_url}/v1/chat/completions"
    num_msgs = len(backend_body.get("messages", []))
    num_tools = len(backend_body.get("tools", []))
    log.info(
        "%sbackend: POST %s route=%s request_model=%s model=%s msgs=%d tools=%d max_tokens=%s stream=%s",
        _r, url, target.route_name, target.request_model, backend_body.get("model"),
        num_msgs, num_tools, backend_body.get("max_completion_tokens"), backend_body.get("stream"),
    )

    resp = await session.post(
        url,
        json=backend_body,
        headers={"Connection": "close"},
        timeout=aiohttp.ClientTimeout(
            total=config.request_timeout,
            sock_read=config.sock_read_timeout,
        ),
    )
    log.info("%sbackend: response status=%d content_type=%s", _r, resp.status, resp.content_type)
    return resp
