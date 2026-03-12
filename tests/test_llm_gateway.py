from __future__ import annotations

from typing import Any, Dict

import pytest

from infra import llm_gateway


def test_chutes_404_falls_back_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "chutes")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHUTES_MODEL_FALLBACK_TO_OPENAI", "1")

    def _raise_404(*, task_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("OpenAI error (404): type=None code=None message=model_not_found")

    def _openai_ok(*, task_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"type":"browser","tool_call":{"name":"browser.wait","arguments":{"time_seconds":1}}}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-5.2",
        }

    monkeypatch.setattr(llm_gateway._chutes, "chat_completions", _raise_404)
    monkeypatch.setattr(llm_gateway._openai, "chat_completions", _openai_ok)

    out = llm_gateway.openai_chat_completions(
        task_id="test-task",
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5.2",
        max_tokens=100,
    )
    assert isinstance(out, dict)
    assert out.get("_provider_fallback") == "openai"
    usage = out.get("usage") if isinstance(out.get("usage"), dict) else {}
    assert int(usage.get("total_tokens") or 0) > 0


def test_chutes_404_respects_disable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "chutes")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHUTES_MODEL_FALLBACK_TO_OPENAI", "0")

    def _raise_404(*, task_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("OpenAI error (404): type=None code=None message=model_not_found")

    monkeypatch.setattr(llm_gateway._chutes, "chat_completions", _raise_404)

    with pytest.raises(RuntimeError):
        llm_gateway.openai_chat_completions(
            task_id="test-task",
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-5.2",
            max_tokens=100,
        )


def test_openai_vision_chat_completions_uses_openai_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Dict[str, Any] = {}

    def _openai_ok(*, task_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        seen["task_id"] = task_id
        seen["body"] = body
        return {
            "choices": [{"message": {"content": '{"answer":"ok","element_ids":[],"signals":[],"confidence":"high"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o-mini",
        }

    monkeypatch.setattr(llm_gateway._openai, "chat_completions", _openai_ok)

    out = llm_gateway.openai_vision_chat_completions(
        task_id="vision-task",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe the screenshot"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8=", "detail": "low"}},
                ],
            }
        ],
        model="gpt-4o-mini",
        max_tokens=120,
    )
    assert seen["task_id"] == "vision-task"
    assert isinstance(seen["body"].get("messages"), list)
    usage = out.get("usage") if isinstance(out.get("usage"), dict) else {}
    assert int(usage.get("total_tokens") or 0) == 15
