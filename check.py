#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
import os
import py_compile
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _load_main_app():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("main", ROOT / "main.py")
    if spec is None or spec.loader is None:
        _fail("Cannot load main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = getattr(module, "app", None)
    if app is None:
        _fail("main.py does not expose app")
    return app


def _find_route(app: Any, path: str, method: str) -> bool:
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) != path:
            continue
        methods = {str(item).upper() for item in getattr(route, "methods", [])}
        if method.upper() in methods:
            return True
    return False


def _find_endpoint(app: Any, path: str, method: str):
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) != path:
            continue
        methods = {str(item).upper() for item in getattr(route, "methods", [])}
        if method.upper() in methods:
            return getattr(route, "endpoint", None)
    return None


def _validate_trajectory_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return f"response must be an object, got {type(payload).__name__}"
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list):
        return "response must include trajectory list"
    for idx, item in enumerate(trajectory):
        if not isinstance(item, dict):
            return f"trajectory[{idx}] must be an object"
        if not isinstance(item.get("name"), str) or not item["name"]:
            return f"trajectory[{idx}].name must be a non-empty string"
        if not isinstance(item.get("arguments"), dict):
            return f"trajectory[{idx}].arguments must be an object"
    return None


def _call_find_trayectory_shape(app: Any) -> None:
    endpoint = _find_endpoint(app, "/find_trayectory", "POST")
    if endpoint is None:
        _fail("POST /find_trayectory route not found")

    from autoppia_harvester.models import FindTrayectoryResponse, ToolCall

    async def fake_find_trayectory(_request):
        return FindTrayectoryResponse(
            web_agent_id="check",
            task_id="check-task",
            trajectory=[ToolCall(name="navigate", arguments={"url": "https://example.com"})],
            actions=[ToolCall(name="navigate", arguments={"url": "https://example.com"})],
            success=True,
            summary="check",
        )

    import autoppia_harvester.app as app_module

    class FakeHarvester:
        async def find_trayectory(self, request):
            return await fake_find_trayectory(request)

    original = app_module.ClaudeCodeHarvester
    app_module.ClaudeCodeHarvester = FakeHarvester
    try:
        from autoppia_harvester.models import FindTrayectoryRequest

        request = FindTrayectoryRequest(id="check-task", prompt="Open page", url="https://example.com", web_project_id="demo")
        if inspect.iscoroutinefunction(endpoint):
            import asyncio

            response = asyncio.run(endpoint(request))
        else:
            response = endpoint(request)
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
    finally:
        app_module.ClaudeCodeHarvester = original

    err = _validate_trajectory_payload(payload)
    if err:
        _fail(f"/find_trayectory response shape invalid: {err}")
    _ok("/find_trayectory response shape is subnet-compatible")


def main() -> None:
    for path in [
        ROOT / "main.py",
        ROOT / "autoppia_harvester" / "app.py",
        ROOT / "autoppia_harvester" / "claude_code.py",
        ROOT / "autoppia_harvester" / "models.py",
        ROOT / "autoppia_harvester" / "trajectory.py",
    ]:
        if not path.exists():
            _fail(f"Missing {path.relative_to(ROOT)}")
        py_compile.compile(str(path), doraise=True)
        _ok(f"Python compile OK: {path.relative_to(ROOT)}")

    app = _load_main_app()
    if not _find_route(app, "/health", "GET"):
        _fail("GET /health route not found")
    _ok("GET /health route found")

    if not _find_route(app, "/find_trayectory", "POST"):
        _fail("POST /find_trayectory route not found")
    _ok("POST /find_trayectory route found")

    if _find_route(app, "/harvest", "POST"):
        _warn("POST /harvest alias is present; keep only if legacy compatibility is desired")

    claude_bin = shutil.which(os.getenv("AUTOPPIA_HARVESTER_CLAUDE_BIN", "claude"))
    if claude_bin:
        _ok(f"Claude Code CLI found: {claude_bin}")
    else:
        _warn("Claude Code CLI not found on PATH; set AUTOPPIA_HARVESTER_CLAUDE_BIN or install claude")

    if os.getenv("ANTHROPIC_API_KEY"):
        _ok("ANTHROPIC_API_KEY is set")
    else:
        _warn("ANTHROPIC_API_KEY not set; Claude Code must have existing auth or the service will fail at runtime")

    _call_find_trayectory_shape(app)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
