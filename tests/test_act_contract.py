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


def test_act_http_response_sets_done_and_result_from_done_action_content(monkeypatch) -> None:
    async def _fake_act_from_payload(payload):
        return {
            "actions": [{"type": "DoneAction", "content": "Portfolio value is $120"}],
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
    assert body["content"] is None
    assert body["done"] is False
    assert body["state_out"] == {"phase": "navigate"}


def test_act_http_response_normalizes_browser_select_tool_call_to_value(monkeypatch) -> None:
    async def _fake_act_from_payload(payload):
        return {
            "protocol_version": "1.0",
            "tool_calls": [
                    {
                        "name": "browser.select_dropdown",
                        "arguments": {
                            "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "genre"},
                            "text": "Comedy",
                    },
                }
            ],
            "done": False,
            "state_out": {"phase": "filter"},
        }

    monkeypatch.setattr(agent.OPERATOR, "act_from_payload", _fake_act_from_payload)
    client = TestClient(agent.app)

    resp = client.post(
        "/act",
        json={
            "task_id": "t-select",
            "prompt": "filter by comedy",
            "url": "https://example.com/search",
            "snapshot_html": "<html></html>",
            "step_index": 0,
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool_calls"] == [
        {
            "name": "browser.select_dropdown",
            "arguments": {
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "genre", "case_sensitive": False},
                "text": "Comedy",
            },
        }
    ]


def test_act_http_response_passthroughs_metrics_and_usage(monkeypatch) -> None:
    async def _fake_act_from_payload(payload):
        return {
            "protocol_version": "1.0",
            "tool_calls": [{"name": "browser.scroll", "arguments": {"direction": "down"}}],
            "done": False,
            "state_out": {"mode": "NAV"},
            "metrics": {"llm": {"llm_calls": 1, "llm_usages": [{"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}], "model": "gpt-5.2"}},
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            "total_tokens": 18,
            "model": "gpt-5.2",
            "helper_models": ["gpt-4o-mini-2024-07-18"],
        }

    monkeypatch.setattr(agent.OPERATOR, "act_from_payload", _fake_act_from_payload)
    client = TestClient(agent.app)

    resp = client.post(
        "/act",
        json={
            "task_id": "t-metrics",
            "prompt": "scroll",
            "url": "https://example.com",
            "snapshot_html": "<html></html>",
            "step_index": 0,
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("metrics"), dict)
    assert isinstance(body.get("usage"), dict)
    assert int(body.get("total_tokens") or 0) == 18
    assert body.get("model") == "gpt-5.2"
    assert body.get("metrics", {}).get("llm", {}).get("helper_models") == ["gpt-4o-mini-2024-07-18"]


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


def test_step_endpoint_aliases_act_behavior(monkeypatch) -> None:
    async def _fake_step_from_payload(payload):
        return {
            "protocol_version": "1.0",
            "tool_calls": [{"name": "browser.navigate", "arguments": {"url": "https://example.com"}}],
            "content": "navigating",
            "done": False,
            "state_out": {"phase": "nav"},
        }

    monkeypatch.setattr(agent.OPERATOR, "step_from_payload", _fake_step_from_payload)
    client = TestClient(agent.app)

    payload = {
        "task_id": "t-step",
        "prompt": "go",
        "url": "https://start.example",
        "snapshot_html": "<html></html>",
        "step_index": 0,
        "history": [],
    }
    body_act = client.post("/act", json=payload).json()
    body_step = client.post("/step", json=payload).json()

    assert body_step == body_act
    assert body_step["tool_calls"] == [{"name": "browser.navigate", "arguments": {"url": "https://example.com"}}]


def test_agent_step_method_aliases_act(monkeypatch) -> None:
    captured = {}

    async def _fake_act(*, task, snapshot_html, screenshot=None, url, step_index, history=None, state=None):
        captured["task"] = task
        captured["snapshot_html"] = snapshot_html
        captured["screenshot"] = screenshot
        captured["url"] = url
        captured["step_index"] = step_index
        captured["history"] = history
        captured["state"] = state
        return ["ok"]

    monkeypatch.setattr(agent.OPERATOR, "act", _fake_act)

    import asyncio
    from types import SimpleNamespace

    out = asyncio.run(
        agent.OPERATOR.step(
            task=SimpleNamespace(id="t", prompt="p"),
            snapshot_html="<html></html>",
            screenshot=None,
            url="https://example.com",
            step_index=3,
            history=[{"type": "NavigateAction"}],
            state={"phase": "x"},
        )
    )

    assert out == ["ok"]
    assert captured["url"] == "https://example.com"
    assert captured["step_index"] == 3
    assert captured["history"] == [{"type": "NavigateAction"}]
    assert captured["state"] == {"phase": "x"}


def test_act_from_payload_forwards_screenshot_to_fsm(monkeypatch) -> None:
    captured = {}

    class _DummyFSM:
        def run(self, *, payload, model_override=""):
            captured["payload"] = payload
            return {"protocol_version": "1.0", "actions": [], "state_out": {}}

    monkeypatch.setattr(agent, "_FSM_OPERATOR", _DummyFSM())

    import asyncio

    asyncio.run(
        agent.OPERATOR.act_from_payload(
            {
                "task_id": "t-shot",
                "prompt": "inspect screenshot",
                "url": "https://example.com",
                "snapshot_html": "<html></html>",
                "screenshot": "data:image/png;base64,abc123",
                "step_index": 0,
                "history": [],
            }
        )
    )

    assert captured["payload"]["screenshot"] == "data:image/png;base64,abc123"
