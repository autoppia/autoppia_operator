from __future__ import annotations

from fastapi.testclient import TestClient

import agent


def test_act_http_response_is_canonical_single_step(monkeypatch) -> None:
    async def _fake_act_from_payload(payload):
        return {
            "protocol_version": "1.0",
            "actions": [
                {"type": "ClickAction", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "go"}},
                {"type": "DoneAction", "reason": "finished"},
            ],
            "reasoning": "pick next",
        }

    monkeypatch.setattr(agent.OPERATOR, "act_from_payload", _fake_act_from_payload)
    client = TestClient(agent.app)

    resp = client.post(
        "/act",
        json={
            "task_id": "t-1",
            "prompt": "click go",
            "url": "https://example.com",
            "snapshot_html": "<html><button id='go'>Go</button></html>",
            "step_index": 0,
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["protocol_version"] == "1.0"
    assert isinstance(body.get("tool_calls"), list)
    assert len(body["tool_calls"]) == 1
    assert body["tool_calls"][0]["name"] == "browser.click"
    assert isinstance(body.get("state_out"), dict)
    assert body["done"] is True


def test_act_http_response_sets_done_from_done_action(monkeypatch) -> None:
    async def _fake_act_from_payload(payload):
        return {
            "actions": [{"type": "DoneAction", "reason": "already done"}],
        }

    monkeypatch.setattr(agent.OPERATOR, "act_from_payload", _fake_act_from_payload)
    client = TestClient(agent.app)

    resp = client.post(
        "/act",
        json={
            "task_id": "t-2",
            "prompt": "done",
            "url": "https://example.com",
            "snapshot_html": "<html></html>",
            "step_index": 0,
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["done"] is True
    assert body["tool_calls"] == []


def test_act_http_response_sets_done_and_result_from_report_result_action(monkeypatch) -> None:
    async def _fake_act_from_payload(payload):
        return {
            "actions": [{"type": "ReportResultAction", "content": "Portfolio value is $120"}],
        }

    monkeypatch.setattr(agent.OPERATOR, "act_from_payload", _fake_act_from_payload)
    client = TestClient(agent.app)

    resp = client.post(
        "/act",
        json={
            "task_id": "t-3",
            "prompt": "fetch portfolio",
            "url": "https://example.com",
            "snapshot_html": "<html></html>",
            "step_index": 0,
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["done"] is True
    assert body["tool_calls"] == []
    assert body["content"] == "Portfolio value is $120"


def test_act_http_response_passthroughs_canonical_tool_calls(monkeypatch) -> None:
    async def _fake_act_from_payload(payload):
        return {
            "protocol_version": "1.0",
            "tool_calls": [
                {
                    "name": "browser.navigate",
                    "arguments": {"url": "https://example.com/docs"},
                }
            ],
            "content": "Navigating to docs",
            "done": False,
            "state_out": {"phase": "navigate"},
        }

    monkeypatch.setattr(agent.OPERATOR, "act_from_payload", _fake_act_from_payload)
    client = TestClient(agent.app)

    resp = client.post(
        "/act",
        json={
            "task_id": "t-4",
            "prompt": "open docs",
            "url": "https://example.com",
            "snapshot_html": "<html></html>",
            "step_index": 0,
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["tool_calls"] == [{"name": "browser.navigate", "arguments": {"url": "https://example.com/docs"}}]
    assert body["content"] == "Navigating to docs"
    assert body["done"] is False
    assert body["state_out"] == {"phase": "navigate"}


def test_capabilities_exposes_protocol_and_tools() -> None:
    client = TestClient(agent.app)
    resp = client.get("/capabilities")
    assert resp.status_code == 200
    body = resp.json()

    assert isinstance(body.get("protocol_version"), str)
    assert body.get("act_endpoint") == "/act"
    assert isinstance(body.get("tool_definitions"), list)
    assert body.get("supports_request_user_input") is True
    assert body.get("supports_state_roundtrip") is True
