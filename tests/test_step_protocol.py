from __future__ import annotations

from src.operator.api.step_protocol import _act_http_response, _normalize_allowed_tool_names, _step_request_from_payload


def test_step_request_from_payload_normalizes_legacy_fields() -> None:
    request = _step_request_from_payload(
        {
            "task_id": "t-1",
            "prompt": "do something",
            "url": "https://example.com",
            "snapshot_html": "<html></html>",
            "allowed_tools": [{"name": "browser.click", "description": "Click", "parameters": {"type": "object"}}],
            "step_index": 2,
            "include_reasoning": True,
        }
    )

    assert request.task_id == "t-1"
    assert request.html == "<html></html>"
    assert request.step_index == 2
    assert request.include_reasoning is True
    assert len(request.tools) == 1
    assert request.tools[0].name == "browser.click"


def test_normalize_allowed_tool_names_accepts_step_tool_shape() -> None:
    allowed = _normalize_allowed_tool_names(
        [
            {"name": "browser.click", "description": "Click", "parameters": {"type": "object"}},
            {"function": {"name": "request_user_input"}},
        ]
    )
    assert "browser.click" in allowed
    assert "user.request_input" in allowed


def test_act_http_response_validates_canonical_step_shape() -> None:
    payload = _act_http_response(
        {
            "protocol_version": "1.0",
            "tool_calls": [{"name": "browser.navigate", "arguments": {"url": "https://example.com"}}],
            "done": False,
        },
        [],
    )

    assert payload["tool_calls"] == [{"name": "browser.navigate", "arguments": {"url": "https://example.com"}}]
    assert payload["done"] is False
