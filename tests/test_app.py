from fastapi.testclient import TestClient

import autoppia_harvester.app as app_module
from autoppia_harvester.app import app
from autoppia_harvester.models import FindTrayectoryResponse, ToolCall


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_find_trayectory_requires_prompt_and_url():
    client = TestClient(app)
    response = client.post("/find_trayectory", json={"prompt": ""})
    assert response.status_code == 400


def test_find_trayectory_returns_subnet_shape(monkeypatch):
    class FakeHarvester:
        async def find_trayectory(self, request):
            return FindTrayectoryResponse(
                web_agent_id="fake",
                task_id=request.canonical_task_id,
                trajectory=[ToolCall(name="navigate", arguments={"url": request.url})],
                actions=[ToolCall(name="navigate", arguments={"url": request.url})],
                success=True,
                summary="ok",
            )

    monkeypatch.setattr(app_module, "ClaudeCodeHarvester", FakeHarvester)
    client = TestClient(app)
    response = client.post(
        "/find_trayectory",
        json={"id": "task-1", "prompt": "Open page", "url": "https://example.com", "web_project_id": "demo"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["web_agent_id"] == "fake"
    assert body["task_id"] == "task-1"
    assert body["trajectory"] == [{"name": "navigate", "arguments": {"url": "https://example.com"}, "reasoning": None}]
    assert body["actions"] == body["trajectory"]
