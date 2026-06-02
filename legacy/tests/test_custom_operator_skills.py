from __future__ import annotations

import pytest

from src.operator.custom_operator import CustomOperator
from src.operator.skills import load_skill_registry


PACKAGE = "examples/custom_operators/autocinema"


def _payload(prompt: str, *, state: dict | None = None) -> dict:
    payload = {
        "task_id": "custom-1",
        "prompt": prompt,
        "url": "http://84.247.180.192:8000",
        "snapshot_html": "<html><body><a id='home-link' href='/'>Home</a></body></html>",
        "step_index": 0,
        "history": [],
    }
    if state is not None:
        payload["_internal_state"] = state
    return payload


@pytest.mark.asyncio
async def test_custom_operator_replays_matching_login_skill() -> None:
    operator = CustomOperator(registry=load_skill_registry(PACKAGE))

    first = await operator.act_from_payload(_payload("login in autocinema"))
    assert first["actions"][0]["type"] == "NavigateAction"
    assert first["actions"][0]["url"].endswith("/login")
    state = first["internal_state"]

    second = await operator.act_from_payload(_payload("login in autocinema", state=state))
    assert second["actions"][0]["type"] == "TypeAction"
    assert second["actions"][0]["selector"]["value"] == "login-username-input"
    assert second["actions"][0]["text"] == "user1"


@pytest.mark.asyncio
async def test_custom_operator_falls_back_to_generalist_when_no_skill_matches() -> None:
    class FakeGeneralist:
        async def act_from_payload(self, payload):
            return {
                "actions": [{"type": "NavigateAction", "url": "http://84.247.180.192:8000/"}],
                "done": False,
                "reasoning": "go home",
                "internal_state": {"mode": "NAV"},
            }

    operator = CustomOperator(registry=load_skill_registry(PACKAGE), generalist=FakeGeneralist())

    out = await operator.act_from_payload(_payload("now go to home"))
    assert out["actions"][0]["type"] == "NavigateAction"
    assert out["actions"][0]["url"].endswith("/")
    assert "Generalist fallback" in out["reasoning"]
    assert out["internal_state"]["custom_operator"]["generalist_internal_state"] == {"mode": "NAV"}
