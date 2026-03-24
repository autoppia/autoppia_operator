from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import llm_gateway


def test_is_sandbox_gateway_base_url_honors_env_override(monkeypatch) -> None:
    """Is sandbox gateway base URL honors env override."""
    monkeypatch.setenv("SANDBOX_GATEWAY_URL", "http://custom-gateway")

    assert llm_gateway.is_sandbox_gateway_base_url("https://api.openai.com/v1") is True


def test_is_sandbox_gateway_base_url_detects_local_hosts(monkeypatch) -> None:
    """Is sandbox gateway base URL detects local hosts."""
    monkeypatch.delenv("SANDBOX_GATEWAY_URL", raising=False)

    assert llm_gateway.is_sandbox_gateway_base_url("http://sandbox-gateway:9000/openai/v1") is True
    assert llm_gateway.is_sandbox_gateway_base_url("http://localhost:9000/openai/v1") is True
    assert llm_gateway.is_sandbox_gateway_base_url("http://127.0.0.1:9000/openai/v1") is True
    assert llm_gateway.is_sandbox_gateway_base_url("not a url") is False


def test_is_sandbox_gateway_base_url_handles_parse_errors(monkeypatch) -> None:
    """Is sandbox gateway base URL handles parse errors."""
    monkeypatch.delenv("SANDBOX_GATEWAY_URL", raising=False)
    monkeypatch.setattr(llm_gateway.httpx, "URL", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")))

    assert llm_gateway.is_sandbox_gateway_base_url("http://example.com") is False


def test_anthropic_helpers_normalize_response() -> None:
    """Anthropic helpers normalize response."""
    raw = {
        "model": "claude-sonnet-4",
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image", "text": "ignore"},
            {"type": "text", "text": " world"},
        ],
        "usage": {"input_tokens": 12, "output_tokens": 4},
    }

    text = llm_gateway._anthropic_extract_text(raw)
    inp, out = llm_gateway._anthropic_usage(raw)
    normalized = llm_gateway._anthropic_to_openai_format(raw, text, inp, out)

    assert text == "Hello world"
    assert (inp, out) == (12, 4)
    assert normalized == {
        "choices": [{"message": {"content": "Hello world"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        "model": "claude-sonnet-4",
    }


def test_anthropic_helpers_handle_invalid_values() -> None:
    """Anthropic helpers handle invalid values."""
    assert llm_gateway._anthropic_extract_text(None) == ""
    assert llm_gateway._anthropic_extract_text({"content": []}) == ""
    assert llm_gateway._anthropic_usage(None) == (0, 0)
    assert llm_gateway._anthropic_usage({"usage": None}) == (0, 0)


def test_openai_chat_completions_uses_anthropic_provider(monkeypatch) -> None:
    """Openai chat completions uses anthropic provider."""
    captured: dict[str, Any] = {}

    def fake_messages(*, task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["task_id"] = task_id
        captured["body"] = body
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(llm_gateway, "_anthropic", type("FakeAnthropic", (), {"messages": staticmethod(fake_messages)})())
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    resp = llm_gateway.openai_chat_completions(
        task_id="task-1",
        model="claude-sonnet-4",
        messages=[
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "User request"},
        ],
        temperature=0.7,
        max_tokens=123,
    )

    assert resp == {"choices": [{"message": {"content": "ok"}}]}
    assert captured["task_id"] == "task-1"
    assert captured["body"] == {
        "model": "claude-sonnet-4",
        "max_tokens": 123,
        "messages": [{"role": "user", "content": "User request"}],
        "temperature": 0.7,
        "system": "System instruction",
    }


def test_openai_chat_completions_handles_broken_anthropic_messages(monkeypatch) -> None:
    """Openai chat completions handles broken anthropic messages."""
    captured: dict[str, Any] = {}

    def fake_messages(*, task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return {"ok": True}

    class BrokenMessages:
        def __iter__(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(llm_gateway, "_anthropic", type("FakeAnthropic", (), {"messages": staticmethod(fake_messages)})())
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    resp = llm_gateway.openai_chat_completions(
        task_id="task-broken",
        model="claude-sonnet-4",
        messages=BrokenMessages(),  # type: ignore[arg-type]
    )

    assert resp == {"ok": True}
    assert captured["body"]["messages"] == [{"role": "user", "content": ""}]
    assert "system" not in captured["body"]


def test_openai_chat_completions_uses_gpt5_parameters(monkeypatch) -> None:
    """Openai chat completions uses gpt5 parameters."""
    captured: dict[str, Any] = {}

    def fake_chat_completions(*, task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["task_id"] = task_id
        captured["body"] = body
        return {"ok": True}

    monkeypatch.setattr(llm_gateway, "_openai", type("FakeOpenAI", (), {"chat_completions": staticmethod(fake_chat_completions)})())
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    resp = llm_gateway.openai_chat_completions(
        task_id="task-2",
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=321,
    )

    assert resp == {"ok": True}
    assert captured["task_id"] == "task-2"
    assert captured["body"] == {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 321,
    }


def test_openai_chat_completions_retries_without_response_format(monkeypatch) -> None:
    """Openai chat completions retries without response format."""
    calls: list[dict[str, Any]] = []

    def fake_chat_completions(*, task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        if len(calls) == 1:
            raise RuntimeError("unsupported_parameter: response_format")
        return {"ok": True}

    monkeypatch.setattr(llm_gateway, "_openai", type("FakeOpenAI", (), {"chat_completions": staticmethod(fake_chat_completions)})())
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    resp = llm_gateway.openai_chat_completions(
        task_id="task-3",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.3,
        max_tokens=99,
    )

    assert resp == {"ok": True}
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
    assert calls[1]["max_tokens"] == 99
    assert calls[1]["temperature"] == 0.3


def test_openai_chat_completions_uses_chutes_provider(monkeypatch) -> None:
    """Openai chat completions uses chutes provider."""
    captured: dict[str, Any] = {}

    def fake_chat_completions(*, task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["task_id"] = task_id
        captured["body"] = body
        return {"provider": "chutes"}

    monkeypatch.setattr(llm_gateway, "_chutes", type("FakeChutes", (), {"chat_completions": staticmethod(fake_chat_completions)})())
    monkeypatch.setenv("LLM_PROVIDER", "chutes")

    resp = llm_gateway.openai_chat_completions(
        task_id="task-4",
        model="model-x",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert resp == {"provider": "chutes"}
    assert captured["body"]["model"] == "model-x"
    assert captured["body"]["max_tokens"] == 300


def test_openai_gateway_chat_completions_success(monkeypatch) -> None:
    """Openai gateway chat completions success."""
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], headers: dict[str, Any]) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(llm_gateway.httpx, "Client", FakeClient)

    gateway = llm_gateway.OpenAIGateway(base_url="http://localhost:9000/openai/v1", api_key="key", timeout_seconds=9)
    resp = gateway.chat_completions(task_id="task-5", body={"model": "gpt-4o"})

    assert resp["choices"][0]["message"]["content"] == "ok"
    assert captured["url"] == "http://localhost:9000/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer key"
    assert captured["headers"]["IWA-Task-ID"] == "task-5"


def test_openai_gateway_chat_completions_sandbox_omits_auth(monkeypatch) -> None:
    """Openai gateway chat completions sandbox omits auth."""
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"ok": True}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], headers: dict[str, Any]) -> FakeResponse:
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(llm_gateway.httpx, "Client", FakeClient)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    gateway = llm_gateway.OpenAIGateway(base_url="http://localhost:9000/openai/v1")
    resp = gateway.chat_completions(task_id="task-sandbox", body={"model": "gpt-4o"})

    assert resp == {"ok": True}
    assert "Authorization" not in captured["headers"]


def test_openai_gateway_formats_http_error(monkeypatch) -> None:
    """Openai gateway formats http error."""
    request = httpx.Request("POST", "http://localhost/chat/completions")
    response = httpx.Response(400, json={"error": {"type": "bad_request", "code": "invalid", "message": "bad input"}})

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], headers: dict[str, Any]):
            return SimpleNamespace(
                raise_for_status=lambda: (_ for _ in ()).throw(httpx.HTTPStatusError("boom", request=request, response=response)),
                json=lambda: {},
            )

    monkeypatch.setattr(llm_gateway.httpx, "Client", FakeClient)

    gateway = llm_gateway.OpenAIGateway(base_url="http://localhost:9000/openai/v1", api_key="key")

    try:
        gateway.chat_completions(task_id="task-6", body={"model": "gpt-4o"})
    except RuntimeError as exc:
        assert "OpenAI error (400)" in str(exc)
        assert "bad_request" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_openai_gateway_formats_http_error_text_fallback(monkeypatch) -> None:
    """Openai gateway formats http error text fallback."""
    request = httpx.Request("POST", "http://localhost/chat/completions")
    response = httpx.Response(500, text="plain failure")

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], headers: dict[str, Any]):
            return SimpleNamespace(
                raise_for_status=lambda: (_ for _ in ()).throw(httpx.HTTPStatusError("boom", request=request, response=response)),
                json=lambda: (_ for _ in ()).throw(ValueError("bad json")),
            )

    monkeypatch.setattr(llm_gateway.httpx, "Client", FakeClient)

    gateway = llm_gateway.OpenAIGateway(base_url="http://localhost:9000/openai/v1", api_key="key")

    with pytest.raises(RuntimeError, match="plain failure"):
        gateway.chat_completions(task_id="task-6b", body={"model": "gpt-4o"})


def test_anthropic_gateway_messages_success(monkeypatch) -> None:
    """Anthropic gateway messages success."""

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], headers: dict[str, Any]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(llm_gateway.httpx, "Client", FakeClient)

    gateway = llm_gateway.AnthropicGateway(api_key="anthropic-key")
    resp = gateway.messages(task_id="task-7", body={"model": "claude-sonnet-4"})

    assert resp["usage"]["total_tokens"] == 3
    assert resp["choices"][0]["message"]["content"] == "hello"


def test_gateway_missing_keys_raise_errors(monkeypatch) -> None:
    """Gateway missing keys raise errors."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set"):
        llm_gateway.OpenAIGateway(base_url="https://api.openai.com/v1").chat_completions(task_id="x", body={})

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
        llm_gateway.AnthropicGateway().messages(task_id="x", body={})


def test_anthropic_gateway_error_is_wrapped(monkeypatch) -> None:
    """Anthropic gateway error is wrapped."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(500, text="server down")

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], headers: dict[str, Any]):
            return SimpleNamespace(
                raise_for_status=lambda: (_ for _ in ()).throw(httpx.HTTPStatusError("boom", request=request, response=response)),
                json=lambda: {},
            )

    monkeypatch.setattr(llm_gateway.httpx, "Client", FakeClient)

    with pytest.raises(RuntimeError, match="Anthropic error \\(500\\)"):
        llm_gateway.AnthropicGateway(api_key="key").messages(task_id="task-8", body={"model": "claude-sonnet-4"})


def test_openai_chat_completions_removes_temperature_on_unsupported_value(monkeypatch) -> None:
    """Openai chat completions removes temperature on unsupported value."""
    calls: list[dict[str, Any]] = []

    def fake_chat_completions(*, task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        calls.append(dict(body))
        if len(calls) == 1:
            raise RuntimeError("unsupported_value: temperature")
        return {"ok": True}

    monkeypatch.setattr(llm_gateway, "_openai", type("FakeOpenAI", (), {"chat_completions": staticmethod(fake_chat_completions)})())
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    resp = llm_gateway.openai_chat_completions(
        task_id="task-9",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.5,
        max_tokens=25,
    )

    assert resp == {"ok": True}
    assert calls[0]["max_tokens"] == 25
    assert "temperature" not in calls[1]


def test_openai_chat_completions_reraises_unhandled_runtime_error(monkeypatch) -> None:
    """Openai chat completions reraises unhandled runtime error."""

    def fake_chat_completions(*, task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("other failure")

    monkeypatch.setattr(llm_gateway, "_openai", type("FakeOpenAI", (), {"chat_completions": staticmethod(fake_chat_completions)})())
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(RuntimeError, match="other failure"):
        llm_gateway.openai_chat_completions(
            task_id="task-err-reraise",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
