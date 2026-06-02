from fastapi.testclient import TestClient

from autoppia_harvester.app import app


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_harvest_requires_prompt_and_url():
    client = TestClient(app)
    response = client.post("/harvest", json={"prompt": ""})
    assert response.status_code == 400
