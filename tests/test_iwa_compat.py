from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import autoppia_harvester.app as app_module
from autoppia_harvester.app import app
from autoppia_harvester.models import FindTrayectoryResponse, ToolCall


def test_iwa_apified_harvester_can_parse_find_trayectory(monkeypatch, unused_tcp_port):
    pytest.importorskip("autoppia_iwa")
    from autoppia_iwa.src.data_generation.tasks.classes import Task
    from autoppia_iwa.src.web_agents.apified_harvester import ApifiedHarvester

    class FakeHarvester:
        async def find_trayectory(self, request):
            return FindTrayectoryResponse(
                web_agent_id="autoppia-harvester-test",
                task_id=request.canonical_task_id,
                trajectory=[ToolCall(name="browser.navigate", arguments={"url": "/dashboard"})],
                actions=[ToolCall(name="browser.navigate", arguments={"url": "/dashboard"})],
                success=True,
                summary="ok",
            )

    monkeypatch.setattr(app_module, "ClaudeCodeHarvester", FakeHarvester)

    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=unused_tcp_port, log_level="error")
    server = uvicorn.Server(config)

    async def run_check():
        task = asyncio.create_task(server.serve())
        try:
            for _ in range(100):
                if server.started:
                    break
                await asyncio.sleep(0.01)
            client = ApifiedHarvester(base_url=f"http://127.0.0.1:{unused_tcp_port}", endpoint_path="/find_trayectory")
            solution = await client.find_trayectory(Task(id="iwa-task", url="https://example.com", prompt="Open dashboard", web_project_id="demo"))
            assert solution.web_agent_id == "autoppia-harvester-test"
            assert len(solution.actions) == 1
            assert solution.actions[0].type == "NavigateAction"
        finally:
            server.should_exit = True
            await task

    asyncio.run(run_check())


def test_find_trayectory_route_exists_for_testclient():
    client = TestClient(app)
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/find_trayectory" in paths
    assert client.get("/health").status_code == 200
