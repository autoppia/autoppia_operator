import json

import httpx
import pytest

from autoppia_harvester.models import FindTrayectoryRequest
from autoppia_harvester.openai_provider import OpenAIHarvester


@pytest.mark.asyncio
async def test_openai_harvester_returns_trajectory(monkeypatch):
    original_async_client = httpx.AsyncClient
    seen_headers = {}

    class AsyncClientWrapper:
        def __new__(cls, *args, **kwargs):
            return original_async_client(
                transport=httpx.MockTransport(
                    lambda request: (
                        seen_headers.update(dict(request.headers))
                        or httpx.Response(
                            200,
                            json={
                                "choices": [
                                    {
                                        "message": {
                                            "content": json.dumps(
                                                {
                                                    "success": True,
                                                    "summary": "ok",
                                                    "trajectory": [
                                                        {"name": "navigate", "arguments": {"url": "https://example.com?seed=1"}},
                                                        {"name": "done", "arguments": {"summary": "complete"}},
                                                    ],
                                                }
                                            )
                                        }
                                    }
                                ],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                            },
                        )
                    )
                ),
                timeout=kwargs.get("timeout"),
            )

    monkeypatch.setattr(httpx, "AsyncClient", AsyncClientWrapper)
    monkeypatch.setenv("AUTOPPIA_HARVESTER_OPENAI_MODEL", "gpt-5")

    result = await OpenAIHarvester().find_trayectory(
        FindTrayectoryRequest(id="task-1", prompt="Open page", url="https://example.com?seed=1", web_project_id="demo")
    )

    assert result.success is True
    assert result.model_used == "openai:gpt-5"
    assert [tool.name for tool in result.trajectory] == ["navigate", "done"]
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert seen_headers["iwa-task-id"] == "task-1"


@pytest.mark.asyncio
async def test_openai_harvester_uses_nested_subnet_task(monkeypatch):
    original_async_client = httpx.AsyncClient
    seen_request = {}

    class AsyncClientWrapper:
        def __new__(cls, *args, **kwargs):
            return original_async_client(
                transport=httpx.MockTransport(
                    lambda request: (
                        seen_request.update({"headers": dict(request.headers), "body": json.loads(request.content.decode("utf-8"))})
                        or httpx.Response(
                            200,
                            json={
                                "choices": [
                                    {
                                        "message": {
                                            "content": json.dumps(
                                                {
                                                    "success": True,
                                                    "trajectory": [
                                                        {"name": "navigate", "arguments": {"url": "https://example.com/about?seed=1"}},
                                                        {"name": "done", "arguments": {}},
                                                    ],
                                                }
                                            )
                                        }
                                    }
                                ],
                                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                            },
                        )
                    )
                ),
                timeout=kwargs.get("timeout"),
            )

    monkeypatch.setattr(httpx, "AsyncClient", AsyncClientWrapper)
    monkeypatch.setenv("AUTOPPIA_HARVESTER_OPENAI_MODEL", "gpt-5")

    result = await OpenAIHarvester().find_trayectory(
        FindTrayectoryRequest(
            project_name="Autoppia Dining",
            task={
                "id": "nested-task",
                "url": "https://example.com?seed=1",
                "prompt": "Navigate to the About page.",
                "web_project_id": "autodining",
            },
        )
    )

    prompt = seen_request["body"]["messages"][1]["content"]
    assert result.task_id == "nested-task"
    assert seen_request["headers"]["iwa-task-id"] == "nested-task"
    assert '"start_url": "https://example.com?seed=1"' in prompt
    assert '"prompt": "Navigate to the About page."' in prompt
