"""Tests for backend request routing."""

from types import SimpleNamespace

import pytest

from claude_relay import backend as backend_module
from claude_relay.backend import BackendState, send_to_backend
from claude_relay.config import ModelRoute, ProxyConfig


class FakeSession:
    def __init__(self):
        self.posts = []

    async def post(self, url, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return SimpleNamespace(status=200, content_type="text/event-stream")


def _state(model, models=None, backend_type="vllm"):
    """Build a BackendState as detect_backend would, for monkeypatching."""
    state = BackendState()
    state.model = model
    state.models = models if models is not None else ([model] if model else [])
    state.backend_type = backend_type
    return state


def _patch_detect(monkeypatch, state, calls=None):
    async def fake_detect_backend(session, backend_url):
        if calls is not None:
            calls.append(backend_url)
        return state

    monkeypatch.setattr(backend_module, "detect_backend", fake_detect_backend)


@pytest.mark.asyncio
async def test_send_to_backend_uses_configured_model_route(monkeypatch):
    calls = []
    # Backend serves Kimi-K2.6 (plus another); the route forces it as upstream.
    _patch_detect(monkeypatch, _state("base-model", ["base-model", "Kimi-K2.6"]), calls)

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_routes={
            "claude-opus": ModelRoute(
                backend_url="http://opus:8000",
                upstream_model="Kimi-K2.6",
            ),
        },
    )
    body = {"model": "claude-opus", "messages": [], "stream": True}

    await send_to_backend(session, config, body)

    assert calls == ["http://opus:8000"]
    assert session.posts[0]["url"] == "http://opus:8000/v1/chat/completions"
    assert session.posts[0]["json"]["model"] == "Kimi-K2.6"
    assert session.posts[0]["json"]["stream_options"] == {"include_usage": True}
    assert body == {"model": "claude-opus", "messages": [], "stream": True}


@pytest.mark.asyncio
async def test_send_to_backend_default_route_uses_detected_model(monkeypatch):
    _patch_detect(monkeypatch, _state("detected-default"))

    session = FakeSession()
    config = ProxyConfig(backend_url="http://default:8000")

    await send_to_backend(session, config, {"model": "claude-sonnet", "messages": []})

    assert session.posts[0]["url"] == "http://default:8000/v1/chat/completions"
    assert session.posts[0]["json"]["model"] == "detected-default"


@pytest.mark.asyncio
async def test_send_to_backend_matches_upstream_model_case_insensitively(monkeypatch):
    # Config carries lowercase upstream_model; backend serves mixed-case id.
    _patch_detect(monkeypatch, _state("DeepSeek-V4-Flash", ["DeepSeek-V4-Flash"]))

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_routes={
            "*opus*": ModelRoute(
                backend_url="http://default:8000",
                upstream_model="deepseek-v4-flash",
            ),
        },
    )

    await send_to_backend(session, config, {"model": "claude-opus-4-8", "messages": []})

    # Sent with the backend's own casing, not the lowercase config value.
    assert session.posts[0]["json"]["model"] == "DeepSeek-V4-Flash"


@pytest.mark.asyncio
async def test_send_to_backend_falls_back_to_first_served_when_unknown(monkeypatch):
    # upstream_model names a model the backend does not serve at all.
    _patch_detect(monkeypatch, _state("DeepSeek-V4-Flash", ["DeepSeek-V4-Flash", "other"]))

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_routes={
            "*opus*": ModelRoute(
                backend_url="http://default:8000",
                upstream_model="ghost-model-9000",
            ),
        },
    )

    await send_to_backend(session, config, {"model": "claude-opus-4-8", "messages": []})

    # Falls back to the first served model instead of forwarding a 404.
    assert session.posts[0]["json"]["model"] == "DeepSeek-V4-Flash"


@pytest.mark.asyncio
async def test_send_to_backend_injects_sampling_from_detected_model(monkeypatch):
    _patch_detect(monkeypatch, _state("deepseek-v4-flash"))

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_sampling={
            "deepseek-v4-flash": {"temperature": 1.0, "top_p": 1.0},
        },
    )
    body = {"model": "claude-opus", "messages": [], "stream": True}

    await send_to_backend(session, config, body)

    assert session.posts[0]["json"]["temperature"] == 1.0
    assert session.posts[0]["json"]["top_p"] == 1.0
    # Params not in model_sampling should not appear
    assert "top_k" not in session.posts[0]["json"]
    assert "presence_penalty" not in session.posts[0]["json"]
    assert body == {"model": "claude-opus", "messages": [], "stream": True}


@pytest.mark.asyncio
async def test_send_to_backend_sampling_lookup_is_case_insensitive(monkeypatch):
    # Detected model and config key differ only by case.
    _patch_detect(monkeypatch, _state("DeepSeek-V4-Flash"))

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_sampling={
            "deepseek-v4-flash": {"temperature": 1.0, "top_p": 1.0},
        },
    )

    await send_to_backend(session, config, {"model": "claude-opus", "messages": []})

    assert session.posts[0]["json"]["temperature"] == 1.0
    assert session.posts[0]["json"]["top_p"] == 1.0


@pytest.mark.asyncio
async def test_send_to_backend_client_params_beat_sampling_defaults(monkeypatch):
    _patch_detect(monkeypatch, _state("deepseek-v4-flash"))

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_sampling={
            "deepseek-v4-flash": {"temperature": 1.0, "top_p": 1.0},
        },
    )
    # Client already set temperature — should not be overridden
    body = {"model": "claude-opus", "messages": [], "temperature": 0.7, "stream": True}

    await send_to_backend(session, config, body)

    assert session.posts[0]["json"]["temperature"] == 0.7
    assert session.posts[0]["json"]["top_p"] == 1.0


def test_resolve_model_matches_case_insensitively_and_falls_back():
    state = _state("DeepSeek-V4-Flash", ["DeepSeek-V4-Flash", "Kimi-K2.6"])
    # Case-insensitive match returns the backend's own casing.
    assert state.resolve_model("deepseek-v4-flash") == "DeepSeek-V4-Flash"
    assert state.resolve_model("KIMI-K2.6") == "Kimi-K2.6"
    # Unknown model falls back to the first served model.
    assert state.resolve_model("does-not-exist") == "DeepSeek-V4-Flash"
    # No requested model also yields the first served model.
    assert state.resolve_model(None) == "DeepSeek-V4-Flash"


@pytest.mark.asyncio
async def test_sampling_follows_resolved_model_not_first_served(monkeypatch):
    # Backend serves two models; a route forces the second one. Sampling must
    # key off the model actually sent, not the first-served detected model.
    _patch_detect(monkeypatch, _state("ModelA", ["ModelA", "ModelB"]))

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_routes={
            "*opus*": ModelRoute(
                backend_url="http://default:8000",
                upstream_model="modelb",
            ),
        },
        model_sampling={
            "ModelA": {"temperature": 0.1},
            "ModelB": {"temperature": 0.9},
        },
    )

    await send_to_backend(session, config, {"model": "claude-opus-4-8", "messages": []})

    assert session.posts[0]["json"]["model"] == "ModelB"
    # ModelB's sampling, not first-served ModelA's 0.1.
    assert session.posts[0]["json"]["temperature"] == 0.9


class _FakeGetResp:
    def __init__(self, payload):
        self._payload = payload
        self.status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload


class FakeGetSession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, url, **kwargs):
        return _FakeGetResp(self._payload)


@pytest.mark.asyncio
async def test_detect_backend_empty_models_preserves_cached_state():
    backend_module.init_state(30)
    url = "http://probe:8000"

    good = {"data": [{"id": "RealModel", "owned_by": "vllm", "max_model_len": 1000}]}
    state = await backend_module.detect_backend(FakeGetSession(good), url)
    assert state.models == ["RealModel"]
    assert state.model == "RealModel"

    # Force a stale re-detect that transiently returns no models.
    state.last_check = 0
    state2 = await backend_module.detect_backend(FakeGetSession({"data": []}), url)

    # Cached values survive — not clobbered to [] / "default".
    assert state2.models == ["RealModel"]
    assert state2.model == "RealModel"


@pytest.mark.asyncio
async def test_detect_backend_skips_entries_without_id():
    backend_module.init_state(30)
    url = "http://probe-noid:8000"

    payload = {"data": [{"object": "model"}, {"id": "RealModel", "owned_by": "vllm", "max_model_len": 1000}]}
    state = await backend_module.detect_backend(FakeGetSession(payload), url)

    # First valid (id-bearing) entry drives state.model; id-less entry dropped.
    assert state.models == ["RealModel"]
    assert state.model == "RealModel"
