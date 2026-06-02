import json

import httpx
import pytest

from autoppia_harvester.models import FindTrayectoryRequest
from autoppia_harvester.openai_provider import OpenAIHarvester


@pytest.mark.asyncio
async def test_openai_harvester_returns_trajectory(monkeypatch):
    original_async_client = httpx.AsyncClient

    class AsyncClientWrapper:
        def __new__(cls, *args, **kwargs):
            return original_async_client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
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
