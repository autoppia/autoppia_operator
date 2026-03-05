#!/usr/bin/env python3
"""Deployment preflight check for autoppia_operator.

This validates the exact things that cause subnet disqualifications before deploy:
- FastAPI routes exist and return valid /act payload shape.
- Operator metadata required by the subnet handshake is present.
- URL normalization to localhost for demo/task payloads and NavigateAction-like actions.
- Optional live-HTTP smoke checks when --live-url is provided.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from typing import Any
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _load_module(path: Path, name: str):
    if not path.exists():
        _fail(f"Missing file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"Cannot import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _find_route(app, path: str, method: str) -> bool:
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) != path:
            continue
        methods = {m.upper() for m in getattr(route, "methods", [])}
        if method.upper() in methods:
            return True
    return False


def _invoke_route_function(app, path: str, payload: dict[str, Any]) -> Any:
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) != path:
            continue
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            _fail(f"Route {path} has no endpoint callable")
        if asyncio.iscoroutinefunction(endpoint):
            return asyncio.run(endpoint(payload))  # type: ignore[arg-type]
        return endpoint(payload)  # type: ignore[arg-type]
    _fail(f"Route not found: {path}")


def _invoke_route_no_payload(app, path: str) -> Any:
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) != path:
            continue
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            _fail(f"Route {path} has no endpoint callable")
        if asyncio.iscoroutinefunction(endpoint):
            return asyncio.run(endpoint())  # type: ignore[misc]
        return endpoint()  # type: ignore[misc]
    _fail(f"Route not found: {path}")


def _check_action_payload_shape(resp: dict[str, Any]) -> str | None:
    if not isinstance(resp, dict):
        return f"expected dict response, got {type(resp).__name__}"
    if "tool_calls" not in resp:
        return "missing 'tool_calls'"
    tool_calls = resp.get("tool_calls")
    if not isinstance(tool_calls, list):
        return f"'tool_calls' should be list, got {type(tool_calls).__name__}"
    for idx, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            return f"tool_calls[{idx}] should be object, got {type(call).__name__}"
        if not isinstance(call.get("name"), str) or not str(call.get("name") or "").strip():
            return f"tool_calls[{idx}] missing valid 'name'"
        if "arguments" in call and not isinstance(call.get("arguments"), dict):
            return f"tool_calls[{idx}].arguments should be object when present"
    if "protocol_version" in resp:
        pv = resp.get("protocol_version")
        if not isinstance(pv, str) or not pv.strip():
            return "protocol_version should be a non-empty string when present"
    if "state_out" not in resp or not isinstance(resp.get("state_out"), dict):
        return "state_out should be object"
    if "done" in resp and not isinstance(resp.get("done"), bool):
        return "done should be bool when present"
    return None


def _check_url_normalizer(agent_mod) -> None:
    normalize_fn = getattr(agent_mod, "_normalize_demo_url", None)
    if not callable(normalize_fn):
        _fail("agent._normalize_demo_url not found")
    # Default behavior preserves real URLs unless AGENT_FORCE_LOCALHOST_URLS=1.
    normed = normalize_fn("84.247.180.192")
    if normed != "84.247.180.192":
        _fail(f"_normalize_demo_url default mode should preserve raw URL. got {normed!r}")
    _ok("_normalize_demo_url default mode preserves incoming URL")

    prev = os.getenv("AGENT_FORCE_LOCALHOST_URLS")
    os.environ["AGENT_FORCE_LOCALHOST_URLS"] = "1"
    try:
        normed = normalize_fn("84.247.180.192")
        if normed != "http://localhost":
            _fail(f"_normalize_demo_url failed to rewrite bare host. expected http://localhost, got {normed!r}")

        normed = normalize_fn("84.247.180.192/task?a=1")
        if normed != "http://localhost/task?a=1":
            _fail(
                "_normalize_demo_url failed for path with query. "
                f"expected http://localhost/task?a=1, got {normed!r}"
            )

        normed = normalize_fn("http://84.247.180.192/task?a=1")
        if normed != "http://localhost/task?a=1":
            _fail(
                "_normalize_demo_url failed to rewrite scheme+host. "
                f"expected http://localhost/task?a=1, got {normed!r}"
            )
    finally:
        if prev is None:
            os.environ.pop("AGENT_FORCE_LOCALHOST_URLS", None)
        else:
            os.environ["AGENT_FORCE_LOCALHOST_URLS"] = prev

    _ok("_normalize_demo_url localhost-rewrite mode works")

    sanitize_action = getattr(agent_mod, "_sanitize_action_payload", None)
    if callable(sanitize_action):
        prev = os.getenv("AGENT_FORCE_LOCALHOST_URLS")
        os.environ["AGENT_FORCE_LOCALHOST_URLS"] = "1"
        try:
            nav_action = sanitize_action({"type": "NavigateAction", "url": "84.247.180.192/foo?x=1"})
            if nav_action.get("url") != "http://localhost/foo?x=1":
                _fail(
                    "_sanitize_action_payload did not sanitize NavigateAction url. "
                    f"got: {nav_action.get('url')!r}"
                )
            _ok("_sanitize_action_payload rewrites NavigateAction URL when localhost mode is enabled")
        finally:
            if prev is None:
                os.environ.pop("AGENT_FORCE_LOCALHOST_URLS", None)
            else:
                os.environ["AGENT_FORCE_LOCALHOST_URLS"] = prev


def _check_task_payload_task_from_payload(agent_mod) -> None:
    task_fn = getattr(agent_mod, "_task_from_payload", None)
    if not callable(task_fn):
        _fail("agent._task_from_payload not found")
    task = task_fn({"task_id": "check", "url": "http://84.247.180.192/test", "prompt": "open page"})
    if not hasattr(task, "url"):
        _fail("_task_from_payload did not return task-like object")
    if str(getattr(task, "url")) != "http://84.247.180.192/test":
        _fail(f"_task_from_payload should preserve incoming URL by default. got {getattr(task, 'url')!r}")
    _ok("_task_from_payload preserves incoming task URL by default")


def _check_handshake_fields() -> None:
    try:
        from autoppia_web_agents_subnet.miner.config import AGENT_NAME, GITHUB_URL, AGENT_IMAGE, AGENT_VERSION
    except Exception as exc:
        _warn("autoppia_web_agents_subnet package not importable for handshake checks")
        print(f"       detail: {exc}")
        return

    name = str(AGENT_NAME or "").strip()
    github_url = str(GITHUB_URL or "").strip()
    version = str(AGENT_VERSION or "").strip()
    image = str(AGENT_IMAGE or "").strip()

    if not name or name in {"AGENT_NAME", "MINER_AGENT_NAME"}:
        _fail("MINER_AGENT_NAME/AGENT_NAME must be set to a non-empty agent name")
    if not github_url or "GITHUB_URL" in github_url or "AGENT_GITHUB_URL" in github_url:
        _fail("MINER_GITHUB_URL/AGENT_GITHUB_URL/GITHUB_URL must be set to a real repository URL")

    parsed = urlparse(github_url)
    if parsed.scheme not in {"http", "https"}:
        _warn(f"GITHUB_URL does not look like an HTTP URL: {github_url}")

    if not version:
        _warn("AGENT_VERSION not set, default/empty may still pass but hurts observability")

    if not name or not github_url:
        _fail("validator-side handshake fields would be considered missing")

    _ok(f"Handshake fields ready: name={name!r}, github_url={github_url!r}")
    if image:
        _ok(f"agent_image present: {image}")
    if version:
        _ok(f"agent_version present: {version}")


def _check_act_response(app) -> None:
    payload = {
        "task_id": "check",
        "prompt": "open homepage",
        "url": "http://84.247.180.192",
        "snapshot_html": "<html><body><button>go</button></body></html>",
        "screenshot": None,
        "step_index": 0,
        "history": [],
    }
    resp = _invoke_route_function(app, "/act", payload)
    if not isinstance(resp, dict):
        _fail(f"/act returned non-dict: {type(resp).__name__}")

    err = _check_action_payload_shape(resp)
    if err:
        _fail(f"/act response invalid: {err}")

    _ok("/act responds with subnet-compatible action payload")


def _check_capabilities_response(app) -> None:
    if not _find_route(app, "/capabilities", "GET"):
        _warn("GET /capabilities route not found")
        return
    resp = _invoke_route_no_payload(app, "/capabilities")
    if not isinstance(resp, dict):
        _fail(f"/capabilities returned non-dict: {type(resp).__name__}")
    if not isinstance(resp.get("protocol_version"), str) or not str(resp.get("protocol_version")).strip():
        _fail("/capabilities missing valid protocol_version")
    _ok("/capabilities responds with protocol metadata")


def _check_http_live(url: str) -> None:
    import httpx

    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.get(f"{url}/health")
            if r.status_code != 200:
                _fail(f"Live /health failed with status {r.status_code}: {r.text[:200]}")
            _ok("Live GET /health 200")

            r = client.post(
                f"{url}/act",
                json={
                    "task_id": "check",
                    "prompt": "open homepage",
                    "url": "http://84.247.180.192",
                    "snapshot_html": "<html><body><button>go</button></body></html>",
                    "screenshot": None,
                    "step_index": 0,
                    "history": [],
                },
            )
            if r.status_code != 200:
                _fail(f"Live /act failed with status {r.status_code}: {r.text[:200]}")
            act_resp = r.json()
            err = _check_action_payload_shape(act_resp)
            if err:
                _fail(f"Live /act response invalid: {err}")
            _ok("Live POST /act 200 and payload shape valid")
            if any(
                isinstance(((c.get("arguments") or {}).get("url")), str)
                and "84.247.180.192" in str((c.get("arguments") or {}).get("url") or "")
                for c in (act_resp.get("tool_calls") or [])
                if isinstance(c, dict) and str(c.get("name") or "").strip().lower() == "browser.navigate"
            ):
                _warn("Live /act response still contains remote host in navigate tool url")
    except Exception as exc:
        _fail(f"Live check failed for {url}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deployment checks for autoppia_operator")
    parser.add_argument(
        "--live-url",
        default="",
        help="Optional running base URL (e.g. http://127.0.0.1:8000) to run live HTTP checks.",
    )
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Fail if --live-url is not provided or server is not reachable.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))

    main_py = REPO_ROOT / "main.py"
    agent_py = REPO_ROOT / "agent.py"
    app_mod = _load_module(main_py, "main")
    agent_mod = _load_module(agent_py, "agent")

    app = getattr(app_mod, "app", None)
    if app is None:
        _fail("main.py does not expose `app`")
    _ok("main.py exposes FastAPI app")

    if not _find_route(app, "/act", "POST"):
        _fail("POST /act route not found")
    _ok("POST /act route found")

    if not _find_route(app, "/health", "GET"):
        _fail("GET /health route not found")
    _ok("GET /health route found")
    if _find_route(app, "/capabilities", "GET"):
        _ok("GET /capabilities route found")
    else:
        _warn("GET /capabilities route not found")

    _check_url_normalizer(agent_mod)
    _check_task_payload_task_from_payload(agent_mod)
    _check_handshake_fields()
    _check_act_response(app)
    _check_capabilities_response(app)

    if args.live_url:
        _check_http_live(args.live_url.rstrip("/"))
    elif args.require_live:
        _fail("--require-live set but --live-url not provided")

    print("\ndeploy_check complete")


if __name__ == "__main__":
    main()
