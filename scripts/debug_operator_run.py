#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import aiohttp
import asyncio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "debug_runs"
DEFAULT_TASK_CACHE = ROOT / "data" / "task_cache" / "tasks_5_projects.json"


def _safe_slug(value: str) -> str:
    text = str(value or "").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {" ", "/", "."}:
            out.append("-")
    return "".join(out).strip("-") or "run"


def _pick_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


async def _wait_health(url: str, timeout_s: float = 15.0) -> None:
    deadline = time.time() + float(timeout_s)
    base = url.rstrip("/")
    async with aiohttp.ClientSession() as session:
        last_error = None
        while time.time() < deadline:
            try:
                async with session.get(f"{base}/health") as resp:
                    if int(resp.status) == 200:
                        return
                    last_error = f"status={resp.status}"
            except Exception as exc:
                last_error = str(exc)
            await asyncio.sleep(0.15)
    raise RuntimeError(f"debugger_healthcheck_failed: {last_error or 'timeout'}")


def _launch_debugger(trace_dir: Path, *, task_cache: Path, host: str, port: int, open_browser: bool) -> int:
    env = os.environ.copy()
    env["OPERATOR_DEBUG_TRACE_DIR"] = str(trace_dir.resolve())
    env["OPERATOR_DEBUG_TASK_CACHE"] = str(task_cache.resolve())
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "debugger_app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    url = f"http://{host}:{port}/"
    try:
        asyncio.run(_wait_health(url))
        if open_browser:
            webbrowser.open(url)
        print(f"Debugger running at {url}")
        print(f"Trace dir: {trace_dir}")
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _run_eval(args: argparse.Namespace, *, debug_root: Path, trace_dir: Path, eval_out: Path) -> None:
    env = os.environ.copy()
    env["EVALUATOR_HEADLESS"] = "0" if bool(args.headed) else "1"
    if bool(args.capture_screenshot):
        env["EVAL_CAPTURE_SCREENSHOT"] = "1"
    cmd = [
        sys.executable,
        str(ROOT / "eval.py"),
        "--provider",
        str(args.provider),
        "--model",
        str(args.model),
        "--task-cache",
        str(Path(args.task_cache).resolve()),
        "--num-tasks",
        "1",
        "--task-concurrency",
        "1",
        "--agent-workers",
        "1",
        "--max-steps",
        str(int(args.max_steps)),
        "--save-act-traces",
        "--trace-dir",
        str(trace_dir),
        "--trace-full-payloads",
        "--out",
        str(eval_out),
        "--failure-judge",
        "--use-site-knowledge",
        "--include-reasoning",
    ]
    if args.web_project_id:
        cmd += ["--web-project-id", str(args.web_project_id)]
    if args.task_id:
        cmd += ["--task-id", str(args.task_id)]
    elif args.use_case:
        cmd += ["--use-case", str(args.use_case)]
    else:
        raise SystemExit("Need --task-id or --use-case for a fresh debug run.")
    print("Running debug eval:")
    print(" ".join(cmd))
    debug_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one operator debug episode and open a local trace viewer.")
    ap.add_argument("--reuse-trace", default=None, help="Existing trace directory to inspect without running eval.")
    ap.add_argument("--web-project-id", default=None)
    ap.add_argument("--use-case", default=None)
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"))
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.2"))
    ap.add_argument("--task-cache", default=str(DEFAULT_TASK_CACHE))
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--headed", action="store_true", help="Run the eval episode with a visible browser.")
    ap.add_argument("--capture-screenshot", action="store_true", help="Capture screenshots into the trace bundle when available.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="Do not auto-open the debugger URL in the system browser.")
    args = ap.parse_args()

    task_cache = Path(args.task_cache).resolve()
    if not task_cache.exists():
        raise SystemExit(f"task cache not found: {task_cache}")

    if args.reuse_trace:
        trace_dir = Path(args.reuse_trace).resolve()
        if not (trace_dir / "trace_index.json").exists():
            raise SystemExit(f"trace_index.json not found in: {trace_dir}")
    else:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        ident = _safe_slug(args.task_id or args.use_case or "task")
        project = _safe_slug(args.web_project_id or "project")
        debug_root = (DATA_DIR / f"{stamp}_{project}_{ident}").resolve()
        trace_dir = (debug_root / "trace").resolve()
        eval_out = (debug_root / "eval.json").resolve()
        _run_eval(args, debug_root=debug_root, trace_dir=trace_dir, eval_out=eval_out)

    port = _pick_port(int(args.port))
    return _launch_debugger(trace_dir, task_cache=task_cache, host=str(args.host), port=port, open_browser=not bool(args.no_open))


if __name__ == "__main__":
    raise SystemExit(main())
