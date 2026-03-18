from fastapi.testclient import TestClient

import agent
import main


def _payload() -> dict:
    return {
        "task_id": "task-123",
        "prompt": "Open the product page",
        "url": "https://demo.example/products?seed=7",
        "snapshot_html": "<html><body><a href='/next?seed=7'>Next</a></body></html>",
        "step_index": 0,
        "history": [],
        "relevant_data": {},
    }


def test_health_endpoint_returns_healthy_status() -> None:
    """Health endpoint returns healthy status."""
    client = TestClient(agent.app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_main_reexports_agent_app() -> None:
    """Main reexports agent app."""
    assert main.app is agent.app


def test_act_endpoint_normalizes_actions_and_keeps_metrics(monkeypatch) -> None:
    """Act endpoint normalizes actions and keeps metrics."""
    client = TestClient(agent.app)

    async def fake_act_from_payload(payload: dict) -> dict:
        assert payload["task_id"] == "task-123"
        return {
            "actions": [
                {"type": "NavigateAction", "url": "https://demo.example/catalog?seed=7"},
                {"type": "ClickAction", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "buy"}},
            ],
            "metrics": {"decision": "mocked"},
        }

    monkeypatch.setattr(agent.OPERATOR, "act_from_payload", fake_act_from_payload)

    resp = client.post("/act", json=_payload())

    assert resp.status_code == 200
    assert resp.json() == {
        "actions": [
            {"type": "NavigateAction", "url": "http://localhost/catalog?seed=7"},
            {"type": "ClickAction", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "buy"}},
        ],
        "metrics": {"decision": "mocked"},
    }


def test_step_endpoint_aliases_act(monkeypatch) -> None:
    """Step endpoint aliases act."""
    client = TestClient(agent.app)

    async def fake_act_from_payload(_payload: dict) -> dict:
        return {"actions": [{"type": "WaitAction", "time_seconds": 0.25}]}

    monkeypatch.setattr(agent.OPERATOR, "act_from_payload", fake_act_from_payload)

    resp = client.post("/step", json=_payload())

    assert resp.status_code == 200
    assert resp.json() == {"actions": [{"type": "WaitAction", "time_seconds": 0.25}]}


def test_act_endpoint_drops_unserializable_actions(monkeypatch) -> None:
    """Act endpoint drops unserializable actions."""
    client = TestClient(agent.app)

    class SerializableAction:
        def model_dump(self, *, exclude_none: bool = True) -> dict:
            assert exclude_none is True
            return {"type": "NavigateAction", "url": "/checkout?seed=7"}

    class BrokenAction:
        def model_dump(self, *, exclude_none: bool = True) -> dict:
            raise RuntimeError("boom")

    async def fake_act_from_payload(_payload: dict) -> dict:
        return {"actions": [SerializableAction(), BrokenAction()]}

    monkeypatch.setattr(agent.OPERATOR, "act_from_payload", fake_act_from_payload)

    resp = client.post("/act", json=_payload())

    assert resp.status_code == 200
    assert resp.json() == {
        "actions": [{"type": "NavigateAction", "url": "http://localhost/checkout?seed=7"}]
    }
