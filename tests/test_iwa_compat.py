from __future__ import annotations

import asyncio
import json

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


def test_full_http_smoke_with_fake_claude_and_iwa_client(monkeypatch, tmp_path, unused_tcp_port):
    pytest.importorskip("autoppia_iwa")
    from autoppia_iwa.src.data_generation.tasks.classes import Task
    from autoppia_iwa.src.web_agents.apified_harvester import ApifiedHarvester

    fake_claude = tmp_path / "claude"
    fake_claude.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "_ = sys.stdin.read()",
                "print(json.dumps({",
                "    'type': 'result',",
                "    'structured_output': {",
                "        'success': True,",
                "        'summary': 'fake claude produced a replayable trajectory',",
                "        'trajectory': [",
                "            {'name': 'browser.navigate', 'arguments': {'url': '/dashboard'}}",
                "        ],",
                "    },",
                "}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)

    workdir = tmp_path / "runs"
    monkeypatch.setenv("AUTOPPIA_HARVESTER_CLAUDE_BIN", str(fake_claude))
    monkeypatch.setenv("AUTOPPIA_HARVESTER_WORKDIR", str(workdir))
    monkeypatch.setenv("AUTOPPIA_HARVESTER_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("AUTOPPIA_HARVESTER_WEB_AGENT_ID", "fake-claude-harvester")

    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=unused_tcp_port, log_level="error")
    server = uvicorn.Server(config)

    async def run_check():
        server_task = asyncio.create_task(server.serve())
        try:
            for _ in range(100):
                if server.started:
                    break
                await asyncio.sleep(0.01)
            assert server.started

            client = ApifiedHarvester(base_url=f"http://127.0.0.1:{unused_tcp_port}", endpoint_path="/find_trayectory")
            solution = await client.find_trayectory(
                Task(
                    id="iwa-smoke-task",
                    url="https://example.com",
                    prompt="Open dashboard",
                    web_project_id="demo",
                )
            )

            assert solution.web_agent_id == "fake-claude-harvester"
            assert solution.task_id == "iwa-smoke-task"
            assert len(solution.actions) == 1
            assert solution.actions[0].type == "NavigateAction"
            assert solution.trajectory[0]["name"] == "navigate"

            run_dirs = [path for path in workdir.iterdir() if path.is_dir()]
            assert len(run_dirs) == 1
            task_payload = json.loads((run_dirs[0] / "task.json").read_text(encoding="utf-8"))
            assert task_payload["iwa_contract"]["endpoint"] == "/find_trayectory"
            assert task_payload["task"]["prompt"] == "Open dashboard"
        finally:
            server.should_exit = True
            await server_task

    asyncio.run(run_check())
