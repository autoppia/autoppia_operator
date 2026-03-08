#!/usr/bin/env python3
"""
Simple evaluation script for the autoppia_operator /act endpoint.

Uses AsyncStatefulEvaluator from autoppia_iwa and calls the local agent
HTTP API directly (no autoppia_rl dependencies).
"""

import asyncio
import base64
import hashlib
import json
import os
from datetime import datetime
import random
import subprocess
import socket
import sys
import time
from typing import Any
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import aiohttp

# ── Ensure the operator repo is on sys.path ─────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OPERATOR_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(OPERATOR_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# ── Load .env from autoppia_operator ────────────────────────────
from dotenv import load_dotenv

operator_env = SCRIPT_DIR / ".env"
if operator_env.exists():
    load_dotenv(operator_env, override=True)

# ── Imports ──────────────────────────────────────────────────────
from loguru import logger

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.evaluation.stateful_evaluator import AsyncStatefulEvaluator
from autoppia_iwa.src.execution.actions.base import BaseAction
from pricing import estimate_cost_usd
import autoppia_iwa.src.execution.actions.actions  # noqa: F401

# Default task cache path
TASK_CACHE = OPERATOR_ROOT / "autoppia_rl" / "data" / "task_cache" / "autoppia_cinema_tasks.json"

random.seed(time.time())


# ── Task loading ─────────────────────────────────────────────────


def _load_raw_tasks(cache_path: Path) -> list[dict[str, Any]]:
    with open(cache_path) as f:
        data = json.load(f)
    raw_tasks = data["tasks"] if isinstance(data, dict) and "tasks" in data else data
    out: list[dict[str, Any]] = []
    for item in raw_tasks if isinstance(raw_tasks, list) else []:
        if isinstance(item, dict):
            out.append(item)
    return out


def _extract_use_case_name(task_payload: dict[str, Any]) -> str:
    uc = task_payload.get("use_case", {})
    if isinstance(uc, dict):
        return str(uc.get("name") or "").strip()
    return ""


def build_task_catalog(cache_path: Path) -> dict[str, dict[str, Any]]:
    raw_tasks = _load_raw_tasks(cache_path)
    catalog: dict[str, dict[str, Any]] = {}
    for td in raw_tasks:
        project_id = str(td.get("web_project_id", "") or "").strip() or "__missing__"
        uc_name = _extract_use_case_name(td) or "__missing__"
        entry = catalog.setdefault(project_id, {"count": 0, "use_cases": defaultdict(int)})
        entry["count"] += 1
        entry["use_cases"][uc_name] += 1
    # Normalize default dict for stable JSON/printing behavior.
    out: dict[str, dict[str, Any]] = {}
    for project_id, payload in catalog.items():
        out[project_id] = {
            "count": int(payload["count"]),
            "use_cases": dict(sorted(payload["use_cases"].items(), key=lambda it: it[0])),
        }
    return dict(sorted(out.items(), key=lambda it: it[0]))


def print_task_catalog(catalog: dict[str, dict[str, Any]], *, only_project: str | None = None) -> None:
    print("\nTask catalog")
    print("=" * 60)
    if not catalog:
        print("No tasks found in cache.")
        print("=" * 60)
        return
    items = catalog.items()
    if only_project:
        items = [(only_project, catalog[only_project])] if only_project in catalog else []
    if not items:
        print(f"No tasks found for project: {only_project}")
        print("=" * 60)
        return
    for project_id, payload in items:
        total = int(payload.get("count") or 0)
        print(f"- {project_id}: {total} task(s)")
        use_cases = payload.get("use_cases") or {}
        for uc_name, uc_count in sorted(use_cases.items(), key=lambda it: it[0]):
            label = uc_name if uc_name != "__missing__" else "<missing>"
            print(f"    {label:30s} {int(uc_count)}")
    print("=" * 60)


def select_all_use_case_tasks(
    raw_tasks: list[dict[str, Any]],
    *,
    web_project_id: str,
    tasks_per_use_case: int = 1,
    seed: int = 0,
) -> list[Task]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for td in raw_tasks:
        if str(td.get("web_project_id", "")) != str(web_project_id):
            continue
        uc_name = _extract_use_case_name(td) or "__missing__"
        grouped[uc_name].append(td)
    selected: list[Task] = []
    rnd = random.Random(int(seed))
    for uc_name in sorted(grouped.keys()):
        bucket = grouped[uc_name]
        rnd.shuffle(bucket)
        for td in bucket[: max(1, int(tasks_per_use_case))]:
            try:
                selected.append(Task(**td))
            except Exception as e:
                logger.debug(f"Skipping task {td.get('id', '?')} from use_case={uc_name}: {e}")
    return selected

def load_tasks(
    cache_path: Path = TASK_CACHE,
    use_case: str | None = None,
    web_project_id: str | None = None,
    task_id: str | None = None,
    limit: int = 20,
) -> list[Task]:
    """Load tasks from the JSON cache, optionally filtered."""
    raw_tasks = _load_raw_tasks(cache_path)

    tasks: list[Task] = []
    for td in raw_tasks:
        # Optional task id filter (exact match)
        if task_id:
            if str(td.get("id", "")) != str(task_id):
                continue

        # Optional use-case filter
        if use_case:
            uc = td.get("use_case", {})
            uc_name = uc.get("name", "") if isinstance(uc, dict) else ""
            if use_case.upper() not in str(uc_name).upper():
                continue

        # Optional web project filter
        if web_project_id is not None:
            if str(td.get("web_project_id", "")) != str(web_project_id):
                continue

        try:
            task = Task(**td)
            tasks.append(task)
        except Exception as e:
            logger.debug(f"Skipping task {td.get('id', '?')}: {e}")

        if len(tasks) >= limit:
            break

    return tasks


def _serialize_screenshot(raw: Any | None) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw or None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(raw)).decode("ascii")
    return None


def _safe_slug(value: str) -> str:
    txt = str(value or "").strip().lower()
    out = []
    for ch in txt:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {" ", ".", "/"}:
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "run"


def _json_dump_path(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _compute_results_summary(results: dict[str, Any], elapsed: float) -> None:
    total = int(results.get("num_tasks") or 0)
    total_steps = sum(int(ep.get("steps") or 0) for ep in results["episodes"])
    avg_task_seconds = (
        sum(float(ep.get("task_seconds", 0.0) or 0.0) for ep in results["episodes"]) / total
        if total > 0
        else 0.0
    )
    avg_step_seconds = (
        sum(float(ep.get("task_seconds", 0.0) or 0.0) for ep in results["episodes"]) / total_steps
        if total_steps > 0
        else 0.0
    )
    results["timing"]["total_seconds"] = round(elapsed, 4)
    results["timing"]["avg_task_seconds"] = round(avg_task_seconds, 4)
    results["timing"]["avg_step_seconds"] = round(avg_step_seconds, 4)


def _save_partial_results(
    out_path: Path,
    results: dict[str, Any],
    *,
    provider: str,
    model: str,
    t_start: float,
) -> None:
    elapsed = time.time() - t_start
    total = int(results.get("num_tasks") or 0)
    succ = int(results.get("successes") or 0)
    results["provider"] = provider
    results["model"] = model
    results["partial"] = True
    results["success_rate"] = round((succ / total) if total > 0 else 0.0, 6)
    results["avg_score"] = round(
        (sum(float(ep.get("score") or 0.0) for ep in results["episodes"]) / total) if total > 0 else 0.0,
        6,
    )
    results["avg_steps"] = round(
        (sum(int(ep.get("steps") or 0) for ep in results["episodes"]) / total) if total > 0 else 0.0,
        6,
    )
    _compute_results_summary(results, elapsed)
    _json_dump_path(out_path, results)


def inject_seed(task: Task, seed: int | None = None) -> tuple[Task, int]:
    """Inject a seed into the task URL for variation (or use a provided seed)."""
    t = deepcopy(task)
    seed_i = int(seed) if seed is not None else random.randint(1, 100_000)
    base_url = t.url.split("?")[0] if "?" in t.url else t.url
    t.url = f"{base_url}?seed={seed_i}"
    return t, seed_i


def _normalize_model_name(name: str) -> str:
    return str(name or "").strip().lower()


def _model_names_compatible(requested: str, effective: str) -> bool:
    req = _normalize_model_name(requested)
    eff = _normalize_model_name(effective)
    if not req or not eff:
        return False
    if req == eff:
        return True
    return eff.startswith(f"{req}-")


# ── Main evaluation loop ────────────────────────────────────────

async def run_evaluation(
    provider: str = "openai",
    model: str = "gpt-5-mini",
    num_tasks: int = 20,
    max_steps: int = 15,
    use_case: str | None = None,
    web_project_id: str | None = None,
    task_id: str | None = None,
    seed: int | None = None,
    repeat: int = 1,
    temperature: float = 0.2,
    distinct_use_cases: bool = False,
    out_path: str | None = None,
    task_cache: str | None = None,
    strict_model: bool = True,
    all_use_cases: bool = False,
    tasks_per_use_case: int = 1,
    task_concurrency: int = 1,
    save_act_traces: bool = False,
    trace_dir: str | None = None,
    trace_full_payloads: bool = True,
    include_reasoning: bool = False,
):
    # Re-load .env here as a guard: some imported modules may mutate env vars.
    try:
        from dotenv import load_dotenv
        operator_env = Path(__file__).resolve().parent / ".env"
        if operator_env.exists():
            load_dotenv(operator_env, override=True)
    except Exception:
        pass

    provider_s = str(provider or os.getenv('LLM_PROVIDER') or 'openai').strip().lower()

    cache_path = Path(task_cache).resolve() if task_cache else TASK_CACHE
    if provider_s == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        api_key_fpr = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:12] if api_key else "missing"
        logger.info(f"Eval env: ANTHROPIC_API_KEY={'set' if api_key else 'missing'} fpr={api_key_fpr}")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set. Check .env file.")
            sys.exit(1)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        api_key_fpr = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:12] if api_key else "missing"
        logger.info(f"Eval env: OPENAI_API_KEY={'set' if api_key else 'missing'} fpr={api_key_fpr}")
        # For sandbox-gateway routing, OPENAI_API_KEY may be intentionally absent.
        base_url = (os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')
        if not api_key and not base_url.startswith('http://sandbox-gateway') and not base_url.startswith('http://localhost') and not base_url.startswith('http://127.0.0.1'):
            logger.error("OPENAI_API_KEY not set. Check .env file.")
            sys.exit(1)
    logger.info("=" * 60)
    logger.info("  Autoppia Operator – LLM Agent Evaluation")
    logger.info(f"  Provider:   {provider_s}")
    logger.info(f"  Model:      {model}")
    logger.info(f"  Tasks:      {num_tasks}")
    logger.info(f"  Task cache: {cache_path}")
    logger.info(f"  Max steps:  {max_steps}")
    logger.info(f"  Use case:   {use_case or 'all'}")
    logger.info(f"  Web proj:   {web_project_id or 'all'}")
    logger.info(f"  Task id:    {task_id or 'auto'}")
    logger.info(f"  All UCs:    {bool(all_use_cases)}")
    logger.info(f"  UC samples: {int(tasks_per_use_case)}")
    logger.info(f"  Seed:       {seed if seed is not None else 'random'}")
    logger.info(f"  Repeat:     {int(repeat)}")
    logger.info(f"  Task conc:  {max(1, int(task_concurrency))}")
    logger.info(f"  Strict mdl: {bool(strict_model)}")
    logger.info(f"  Trace acts: {bool(save_act_traces)}")
    logger.info(f"  Reasoning:  {bool(include_reasoning)}")
    logger.info("=" * 60)

    trace_root: Path | None = None
    trace_index: dict[str, Any] = {}
    if bool(save_act_traces):
        if trace_dir:
            trace_root = Path(trace_dir).resolve()
        else:
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            model_slug = _safe_slug(model)
            project_slug = _safe_slug(web_project_id or "all-projects")
            trace_root = (SCRIPT_DIR / "data" / "act_traces" / f"{stamp}_{project_slug}_{model_slug}").resolve()
        trace_root.mkdir(parents=True, exist_ok=True)
        trace_index = {
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "provider": provider_s,
            "model": model,
            "requested_model": str(model),
            "web_project_id": web_project_id,
            "use_case": use_case,
            "all_use_cases": bool(all_use_cases),
            "tasks_per_use_case": int(tasks_per_use_case),
            "task_concurrency": max(1, int(task_concurrency)),
            "max_steps": int(max_steps),
            "trace_full_payloads": bool(trace_full_payloads),
            "episodes": [],
        }
        _json_dump_path(trace_root / "trace_index.json", trace_index)

    # Load tasks.
    if all_use_cases:
        if not web_project_id:
            logger.error("--all-use-cases requires --web-project-id")
            return
        raw_tasks = _load_raw_tasks(cache_path)
        tasks = select_all_use_case_tasks(
            raw_tasks,
            web_project_id=str(web_project_id),
            tasks_per_use_case=max(1, int(tasks_per_use_case)),
            seed=int(seed or 0),
        )
        logger.info(
            f"Selected {len(tasks)} task(s) across all use cases for project={web_project_id} "
            f"(tasks_per_use_case={max(1, int(tasks_per_use_case))})"
        )
    else:
        # If we want distinct use cases, load more upfront then filter down.
        load_limit = num_tasks
        if distinct_use_cases:
            load_limit = max(500, num_tasks * 20)
        tasks = load_tasks(cache_path=cache_path, use_case=use_case, web_project_id=web_project_id, task_id=task_id, limit=load_limit)
    logger.info(f"Loaded {len(tasks)} tasks")

    if not tasks:
        logger.error("No tasks found. Check task cache path and use_case filter.")
        return

    if distinct_use_cases:
        picked: list[Task] = []
        seen: set[str] = set()
        rest: list[Task] = []
        for t in tasks:
            uc_name = ""
            uc = getattr(t, "use_case", None)
            if isinstance(uc, dict):
                uc_name = str(uc.get("name") or "")
            elif uc is not None and hasattr(uc, "name"):
                uc_name = str(getattr(uc, "name") or "")
            if not uc_name:
                rest.append(t)
                continue
            if uc_name in seen:
                rest.append(t)
                continue
            seen.add(uc_name)
            picked.append(t)
            if len(picked) >= num_tasks:
                break
        if len(picked) < num_tasks:
            for t in rest:
                picked.append(t)
                if len(picked) >= num_tasks:
                    break
        tasks = picked[:num_tasks]
        logger.info(f"Selected {len(tasks)} tasks with distinct use cases")
    else:
        tasks = tasks[:num_tasks]

    # Agent endpoint config
    agent_base_url = os.getenv("AGENT_BASE_URL", "").strip().rstrip("/")
    start_server = os.getenv("START_AGENT_SERVER", "1") in {"1", "true", "yes"}
    base_web_agent_id = os.getenv("WEB_AGENT_ID", "1").strip() or "1"
    def _port_available(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False

    def _pick_port(preferred: int) -> int:
        if _port_available(preferred):
            return preferred
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    # If we're starting the server locally, choose a free port and point the client to it.
    if start_server:
        preferred_port = int(os.getenv("AGENT_PORT", "5000"))
        port = _pick_port(preferred_port)
        if not agent_base_url:
            agent_base_url = f"http://127.0.0.1:{port}"

        # Append logs per run so bind errors / tracebacks are preserved.
        out_dir = SCRIPT_DIR / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "agent_server.log"
        log_f = open(log_path, "a", encoding="utf-8")
        log_f.write(f"\n=== uvicorn main:app port={port} ===\n")
        log_f.flush()

        server_env = os.environ.copy()
        server_env["OPENAI_MODEL"] = str(model)
        server_env["LLM_PROVIDER"] = str(provider_s)
        server_env["OPENAI_TEMPERATURE"] = str(temperature)
        server_env["AGENT_RETURN_METRICS"] = "1"
        k = server_env.get("OPENAI_API_KEY") or ""
        k_fpr = hashlib.sha256(k.encode("utf-8")).hexdigest()[:12] if k else "missing"
        logger.info(f"Agent server env: OPENAI_MODEL={server_env.get('OPENAI_MODEL')} LLM_PROVIDER={server_env.get('LLM_PROVIDER')} START_AGENT_SERVER={os.getenv('START_AGENT_SERVER')} AGENT_BASE_URL={agent_base_url or 'local'} OPENAI_API_KEY={'set' if k else "missing"} fpr={k_fpr}")
        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ],
            cwd=str(SCRIPT_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=server_env,
        )
        async def _wait_for_server_health(url: str, *, timeout_s: float = 15.0) -> None:
            deadline = time.time() + float(timeout_s)
            last_error: str | None = None
            async with aiohttp.ClientSession() as session:
                while time.time() < deadline:
                    try:
                        async with session.get(f"{url}/health") as resp:
                            if int(resp.status) == 200:
                                return
                            last_error = f"status={resp.status}"
                    except Exception as exc:
                        last_error = str(exc)
                    await asyncio.sleep(0.15)
            raise RuntimeError(f"agent_server_healthcheck_failed url={url} err={last_error or 'timeout'}")

        await _wait_for_server_health(agent_base_url)
    else:
        if not agent_base_url:
            agent_base_url = "http://127.0.0.1:5000"
        server_proc = None




    async def call_agent_act(
        session: aiohttp.ClientSession,
        prepared_task: Task,
        episode_task_id: str,
        snapshot_html: str,
        url: str,
        step_index: int,
        history: list[dict],
        requested_model: str,
        state_in: dict[str, Any],
        screenshot: str | None = None,
    ) -> tuple[list[BaseAction], dict, bool, str | None, str | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
        payload = {
            "task_id": str(episode_task_id),
            "prompt": prepared_task.prompt,
            "url": url,
            "snapshot_html": snapshot_html,
            "screenshot": screenshot,
            "step_index": int(step_index),
            "web_project_id": prepared_task.web_project_id,
            "history": history,
            "model": str(requested_model),
            "include_reasoning": bool(include_reasoning),
            "state_in": state_in if isinstance(state_in, dict) else {},
            "allowed_tools": BaseAction.all_function_definitions(),
        }
        request_payload = dict(payload)
        data: dict[str, Any] | list[Any] | None = None
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                async with session.post(f"{agent_base_url}/act", json=payload) as resp:
                    if int(resp.status) >= 500 and attempt < 3:
                        await asyncio.sleep(0.35 * attempt)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    break
            except (
                aiohttp.ClientConnectionError,
                aiohttp.ClientOSError,
                aiohttp.ServerDisconnectedError,
                asyncio.TimeoutError,
            ) as exc:
                last_exc = exc
                if server_proc is not None and server_proc.poll() is not None:
                    raise RuntimeError(
                        f"agent_server_exited code={server_proc.poll()} task={episode_task_id} step={step_index}"
                    ) from exc
                if attempt >= 3:
                    raise
                logger.warning(
                    f"/act transient failure attempt={attempt}/3 task={episode_task_id} step={step_index}: {type(exc).__name__}: {exc}"
                )
                with contextlib.suppress(Exception):
                    await _wait_for_server_health(agent_base_url, timeout_s=5.0)
                await asyncio.sleep(min(0.5 * attempt, 1.5))
        if data is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"agent_act_no_response task={episode_task_id} step={step_index}")

        metrics = data.get("metrics") if isinstance(data, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        done = bool(data.get("done")) if isinstance(data, dict) else False
        content = data.get("content") if isinstance(data, dict) else None
        reasoning = data.get("reasoning") if isinstance(data, dict) else None
        state_out = data.get("state_out") if isinstance(data, dict) else {}
        if not isinstance(state_out, dict):
            state_out = {}

        tool_calls_payload = data.get("tool_calls") if isinstance(data, dict) else None
        actions_alias_payload = data.get("actions") if isinstance(data, dict) else None
        actions_payload: list[Any] = []
        if isinstance(tool_calls_payload, list):
            actions_payload = tool_calls_payload
        elif isinstance(actions_alias_payload, list):
            actions_payload = actions_alias_payload
        if not isinstance(actions_payload, list):
            return (
                [],
                metrics,
                done,
                content if isinstance(content, str) else None,
                reasoning if isinstance(reasoning, str) else None,
                state_out,
                request_payload,
                data if isinstance(data, dict) else {},
            )
        actions: list[BaseAction] = []
        for raw in actions_payload:
            if not isinstance(raw, dict):
                continue
            payload = dict(raw)
            # Canonical tool call shape.
            if isinstance(payload.get("name"), str):
                name = str(payload.get("name") or "").strip().lower()
                args = payload.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                if name.startswith("browser."):
                    action_type = name.split(".", 1)[1].strip()
                    if action_type:
                        normalized = dict(args)
                        normalized["type"] = str(normalized.get("type") or action_type)
                        payload = normalized
                    else:
                        payload = {"name": name, "arguments": args}
                elif name == "user.request_input":
                    normalized = dict(args)
                    normalized["type"] = str(normalized.get("type") or "request_user_input")
                    payload = normalized
                else:
                    payload = {"name": name, "arguments": args}
            # Legacy action shape with browser.* encoded under "type".
            elif isinstance(payload.get("type"), str) and str(payload.get("type")).startswith("browser."):
                action_type = str(payload.get("type")).split(".", 1)[1]
                args = {k: v for k, v in payload.items() if k != "type"}
                payload = dict(args)
                payload["type"] = action_type
            try:
                act = BaseAction.create_action(payload)
                if act is not None:
                    actions.append(act)
            except Exception:
                continue
        return (
            actions,
            metrics,
            done,
            content if isinstance(content, str) else None,
            reasoning if isinstance(reasoning, str) else None,
            state_out,
            request_payload,
            data if isinstance(data, dict) else {},
        )

    # Results tracking
    requested_model = str(model)
    results = {
        "provider": provider_s,
        "model": model,
        "requested_model": requested_model,
        "num_tasks": 0,
        "successes": 0,
        "failures": 0,
        "errors": 0,
        "model_mismatch_errors": 0,
        "timing": {
            "total_seconds": 0.0,
            "avg_task_seconds": 0.0,
            "avg_step_seconds": 0.0,
        },
        "episodes": [],
    }

    t_start = time.time()
    reps = max(1, int(repeat))
    concurrency = max(1, int(task_concurrency))
    episode_specs: list[dict[str, Any]] = []
    for i, base_task in enumerate(tasks):
        for r in range(reps):
            seed_i = (int(seed) + int(r)) if seed is not None else None
            task, seed_used = inject_seed(base_task, seed=seed_i)
            uc_name = ""
            if hasattr(task, "use_case") and task.use_case:
                uc = task.use_case
                if isinstance(uc, dict):
                    uc_name = uc.get("name", "unknown")
                elif hasattr(uc, "name"):
                    uc_name = uc.name
                else:
                    uc_name = str(uc)
            episode_index = len(episode_specs)
            episode_web_agent_id = f"{base_web_agent_id}-{episode_index + 1}"
            episode_task_id = f"{task.id}-{seed_used}-{r}-{episode_web_agent_id}"
            episode_specs.append(
                {
                    "episode_index": episode_index,
                    "task_index": i,
                    "task_total": len(tasks),
                    "prepared_task": task,
                    "seed_used": seed_used,
                    "repeat_index": r,
                    "use_case": uc_name,
                    "web_agent_id": episode_web_agent_id,
                    "episode_task_id": episode_task_id,
                }
            )

    async def run_episode(
        spec: dict[str, Any],
        *,
        agent_session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            prepared_task: Task = spec["prepared_task"]
            seed_used = int(spec["seed_used"])
            r = int(spec["repeat_index"])
            uc_name = str(spec["use_case"] or "")
            episode_task_id = str(spec["episode_task_id"])
            episode_web_agent_id = str(spec["web_agent_id"])
            rep_label = f" r={r + 1}/{reps}" if reps > 1 else ""
            logger.info(
                f"[{int(spec['task_index']) + 1}/{int(spec['task_total'])}]{rep_label} seed={seed_used} agent={episode_web_agent_id} | {uc_name} | {prepared_task.prompt[:50]}..."
            )

            task_start = time.time()
            episode_llm_calls = 0
            episode_prompt_tokens = 0
            episode_completion_tokens = 0
            episode_total_tokens = 0
            episode_cost_usd = 0.0
            episode_model = str(model)
            final_content: str | None = None
            episode_trace_steps: list[dict[str, Any]] = []
            trace_index_item: dict[str, Any] | None = None
            evaluator = None

            try:
                evaluator = AsyncStatefulEvaluator(
                    task=prepared_task,
                    web_agent_id=episode_web_agent_id,
                    capture_screenshot=True,
                )
                step_result = await evaluator.reset()

                history: list[dict] = []
                agent_state_in: dict[str, Any] = {}
                final_score = 0.0
                final_success = False
                total_steps = 0

                for step_idx in range(max_steps):
                    pre_url = str(step_result.snapshot.url)
                    pre_score = float(step_result.score.raw_score)
                    pre_success = bool(step_result.score.success)
                    actions, metrics, done, content, reasoning, state_out, act_request_payload, act_raw_response = await call_agent_act(
                        agent_session,
                        prepared_task,
                        episode_task_id=episode_task_id,
                        snapshot_html=step_result.snapshot.html,
                        url=step_result.snapshot.url,
                        step_index=step_idx,
                        screenshot=_serialize_screenshot(getattr(step_result.snapshot, "screenshot", None)),
                        history=history,
                        requested_model=str(requested_model),
                        state_in=agent_state_in,
                    )
                    agent_state_in = state_out if isinstance(state_out, dict) else {}
                    if isinstance(content, str) and content.strip():
                        final_content = content.strip()

                    llm_meta = metrics.get("llm") if isinstance(metrics, dict) else None
                    if isinstance(llm_meta, dict):
                        usages = llm_meta.get("llm_usages")
                        model_name = llm_meta.get("model") or model
                        episode_model = str(model_name)
                        if bool(strict_model) and not _model_names_compatible(str(requested_model), str(episode_model)):
                            raise RuntimeError(
                                f"model_mismatch requested={requested_model} effective={episode_model} task={episode_task_id} step={step_idx}"
                            )
                        if isinstance(usages, list):
                            for u in usages:
                                if not isinstance(u, dict):
                                    continue
                                pt = int(u.get("prompt_tokens") or 0)
                                ct = int(u.get("completion_tokens") or 0)
                                tt = int(u.get("total_tokens") or (pt + ct))
                                episode_prompt_tokens += pt
                                episode_completion_tokens += ct
                                episode_total_tokens += tt
                                c, _ = estimate_cost_usd(str(model_name), u)
                                episode_cost_usd += float(c)
                            episode_llm_calls += int(llm_meta.get("llm_calls") or len(usages))

                    if actions:
                        action = actions[0]
                        step_result = await evaluator.step(action)
                    elif done:
                        action = None
                        final_score = step_result.score.raw_score
                        final_success = step_result.score.success
                        total_steps = step_idx + 1
                        history.append(
                            {
                                "step": step_idx,
                                "url": str(step_result.snapshot.url),
                                "action": "done",
                                "candidate_id": None,
                                "text": None,
                                "exec_ok": True,
                                "error": None,
                                "agent_decision": (metrics.get("decision") if isinstance(metrics, dict) else None),
                                "llm_calls": int((llm_meta.get("llm_calls") if isinstance(llm_meta, dict) else 0) or 0),
                                "prompt_tokens": int(sum(int(u.get("prompt_tokens") or 0) for u in (llm_meta.get("llm_usages") if isinstance(llm_meta, dict) and isinstance(llm_meta.get("llm_usages"), list) else []))),
                                "completion_tokens": int(sum(int(u.get("completion_tokens") or 0) for u in (llm_meta.get("llm_usages") if isinstance(llm_meta, dict) and isinstance(llm_meta.get("llm_usages"), list) else []))),
                                "done": True,
                                "content": final_content,
                                "reasoning": reasoning if isinstance(reasoning, str) else None,
                            }
                        )
                        episode_trace_steps.append(
                            {
                                "step_index": int(step_idx),
                                "before": {"url": pre_url, "score": pre_score, "success": pre_success},
                                "after": {
                                    "url": str(step_result.snapshot.url),
                                    "score": float(step_result.score.raw_score),
                                    "success": bool(step_result.score.success),
                                },
                                "agent": {
                                    "done": bool(done),
                                    "content": content if isinstance(content, str) else None,
                                    "reasoning": reasoning if isinstance(reasoning, str) else None,
                                    "metrics": metrics if isinstance(metrics, dict) else {},
                                    "state_in": act_request_payload.get("state_in") if isinstance(act_request_payload, dict) else {},
                                    "state_out": state_out if isinstance(state_out, dict) else {},
                                },
                                "action": None,
                                "execution": {"executed": False, "done_break": True},
                                "act_request": act_request_payload if bool(trace_full_payloads) else {
                                    "task_id": act_request_payload.get("task_id") if isinstance(act_request_payload, dict) else None,
                                    "url": act_request_payload.get("url") if isinstance(act_request_payload, dict) else None,
                                    "step_index": act_request_payload.get("step_index") if isinstance(act_request_payload, dict) else None,
                                    "history_count": len((act_request_payload.get("history") or [])) if isinstance(act_request_payload, dict) else 0,
                                },
                                "act_response": act_raw_response if bool(trace_full_payloads) else {
                                    "done": bool(done),
                                    "content": content if isinstance(content, str) else None,
                                    "reasoning": reasoning if isinstance(reasoning, str) else None,
                                    "tool_calls_count": len((act_raw_response or {}).get("tool_calls") or []) if isinstance(act_raw_response, dict) else 0,
                                },
                            }
                        )
                        break
                    else:
                        action = None
                        step_result = await evaluator.step(None)

                    if os.getenv("EVAL_SAVE_TRACES", "0").lower() in {"1", "true", "yes"}:
                        try:
                            trace_dir = (SCRIPT_DIR / "data" / "traces" / str(episode_task_id))
                            trace_dir.mkdir(parents=True, exist_ok=True)
                            (trace_dir / f"{step_idx:02d}.url.txt").write_text(str(step_result.snapshot.url), encoding="utf-8")
                            (trace_dir / f"{step_idx:02d}.html").write_text(str(step_result.snapshot.html), encoding="utf-8", errors="replace")
                        except Exception:
                            pass

                    final_score = step_result.score.raw_score
                    final_success = step_result.score.success
                    total_steps = step_idx + 1

                    cid = None
                    if isinstance(metrics, dict):
                        cid = metrics.get("candidate_id")
                    if isinstance(cid, str) and cid.isdigit():
                        cid = int(cid)

                    ar = step_result.action_result
                    exec_ok = True
                    exec_err = None
                    try:
                        if ar is not None:
                            exec_ok = bool(getattr(ar, "successfully_executed", True))
                            exec_err = getattr(ar, "error", None)
                    except Exception:
                        exec_ok = True
                        exec_err = None

                    history.append(
                        {
                            "step": step_idx,
                            "url": str(step_result.snapshot.url),
                            "action": action.type if action else "done",
                            "candidate_id": cid,
                            "text": getattr(action, "text", None) if action else None,
                            "exec_ok": exec_ok,
                            "error": exec_err,
                            "agent_decision": (metrics.get("decision") if isinstance(metrics, dict) else None),
                            "done": bool(done),
                            "content": content if isinstance(content, str) else None,
                            "reasoning": reasoning if isinstance(reasoning, str) else None,
                            "llm_calls": int((llm_meta.get("llm_calls") if isinstance(llm_meta, dict) else 0) or 0),
                            "prompt_tokens": int(sum(int(u.get("prompt_tokens") or 0) for u in (llm_meta.get("llm_usages") if isinstance(llm_meta, dict) and isinstance(llm_meta.get("llm_usages"), list) else []))),
                            "completion_tokens": int(sum(int(u.get("completion_tokens") or 0) for u in (llm_meta.get("llm_usages") if isinstance(llm_meta, dict) and isinstance(llm_meta.get("llm_usages"), list) else []))),
                        }
                    )

                    step_trace: dict[str, Any] = {
                        "step_index": int(step_idx),
                        "before": {"url": pre_url, "score": pre_score, "success": pre_success},
                        "after": {
                            "url": str(step_result.snapshot.url),
                            "score": float(step_result.score.raw_score),
                            "success": bool(step_result.score.success),
                        },
                        "agent": {
                            "done": bool(done),
                            "content": content if isinstance(content, str) else None,
                            "reasoning": reasoning if isinstance(reasoning, str) else None,
                            "metrics": metrics if isinstance(metrics, dict) else {},
                            "state_in": act_request_payload.get("state_in") if isinstance(act_request_payload, dict) else {},
                            "state_out": state_out if isinstance(state_out, dict) else {},
                        },
                        "action": {
                            "type": action.type if action else None,
                            "raw": action.model_dump(mode="json", exclude_none=False) if action else None,
                        },
                        "execution": {"executed": bool(action is not None), "exec_ok": bool(exec_ok), "error": exec_err},
                    }
                    if bool(trace_full_payloads):
                        step_trace["act_request"] = act_request_payload
                        step_trace["act_response"] = act_raw_response
                    else:
                        step_trace["act_request"] = {
                            "task_id": act_request_payload.get("task_id") if isinstance(act_request_payload, dict) else None,
                            "url": act_request_payload.get("url") if isinstance(act_request_payload, dict) else None,
                            "step_index": act_request_payload.get("step_index") if isinstance(act_request_payload, dict) else None,
                            "history_count": len((act_request_payload.get("history") or [])) if isinstance(act_request_payload, dict) else 0,
                        }
                        step_trace["act_response"] = {
                            "done": bool(done),
                            "content": content if isinstance(content, str) else None,
                            "reasoning": reasoning if isinstance(reasoning, str) else None,
                            "tool_calls_count": len((act_raw_response or {}).get("tool_calls") or []) if isinstance(act_raw_response, dict) else 0,
                        }
                    episode_trace_steps.append(step_trace)

                    if final_success:
                        break

                await evaluator.close()
                evaluator = None

                task_elapsed = time.time() - task_start
                if not final_success:
                    try:
                        out_dir = SCRIPT_DIR / "data"
                        fail_dir = out_dir / "failures"
                        fail_dir.mkdir(parents=True, exist_ok=True)
                        (fail_dir / f"{episode_task_id}.url.txt").write_text(str(step_result.snapshot.url), encoding="utf-8")
                        (fail_dir / f"{episode_task_id}.html").write_text(str(step_result.snapshot.html), encoding="utf-8", errors="replace")
                    except Exception:
                        pass

                steps_count = total_steps or 0
                avg_step_seconds = (task_elapsed / steps_count) if steps_count > 0 else 0.0
                ep_data = {
                    "task_id": str(prepared_task.id),
                    "episode_task_id": str(episode_task_id),
                    "model": str(episode_model),
                    "repeat_index": int(r),
                    "use_case": uc_name,
                    "seed": seed_used,
                    "success": bool(final_success),
                    "score": float(final_score),
                    "steps": steps_count,
                    "task_seconds": round(task_elapsed, 4),
                    "llm_calls": int(episode_llm_calls),
                    "prompt_tokens": int(episode_prompt_tokens),
                    "completion_tokens": int(episode_completion_tokens),
                    "total_tokens": int(episode_total_tokens),
                    "estimated_cost_usd": round(float(episode_cost_usd), 6),
                    "avg_step_seconds": round(avg_step_seconds, 4),
                    "final_content": final_content,
                    "web_agent_id": episode_web_agent_id,
                }
                if bool(save_act_traces) and trace_root is not None:
                    ep_dir = trace_root / "episodes"
                    episode_payload = {
                        "episode": ep_data,
                        "task_prompt": str(getattr(prepared_task, "prompt", "")),
                        "task_url": str(getattr(prepared_task, "url", "")),
                        "task_web_project_id": str(getattr(prepared_task, "web_project_id", "")),
                        "steps": episode_trace_steps,
                    }
                    _json_dump_path(ep_dir / f"{episode_task_id}.json", episode_payload)
                    trace_index_item = {
                        "episode_task_id": str(episode_task_id),
                        "task_id": str(prepared_task.id),
                        "use_case": uc_name,
                        "success": bool(final_success),
                        "score": float(final_score),
                        "steps": int(steps_count),
                        "file": f"episodes/{episode_task_id}.json",
                    }

                logger.info(f"  -> {'SUCCESS' if final_success else 'FAILED '} (score={final_score:.2f}, steps={steps_count})")
                return {
                    "episode_index": int(spec["episode_index"]),
                    "status": "success" if final_success else "failure",
                    "model_mismatch_error": False,
                    "episode": ep_data,
                    "trace_index_item": trace_index_item,
                }
            except Exception as e:
                try:
                    if evaluator is not None:
                        await evaluator.close()
                except Exception:
                    pass
                task_elapsed = time.time() - task_start
                ep_data = {
                    "task_id": str(prepared_task.id),
                    "episode_task_id": str(episode_task_id),
                    "repeat_index": int(r),
                    "use_case": uc_name,
                    "seed": seed_used,
                    "success": False,
                    "score": 0.0,
                    "steps": 0,
                    "task_seconds": round(task_elapsed, 4),
                    "llm_calls": int(episode_llm_calls),
                    "prompt_tokens": int(episode_prompt_tokens),
                    "completion_tokens": int(episode_completion_tokens),
                    "total_tokens": int(episode_total_tokens),
                    "estimated_cost_usd": round(float(episode_cost_usd), 6),
                    "avg_step_seconds": 0.0,
                    "error": str(e),
                    "web_agent_id": episode_web_agent_id,
                }
                logger.error(f"  -> ERROR: {e}")
                return {
                    "episode_index": int(spec["episode_index"]),
                    "status": "error",
                    "model_mismatch_error": "model_mismatch" in str(e),
                    "episode": ep_data,
                    "trace_index_item": trace_index_item,
                }

    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_connect=20, sock_read=300)
    connector_limit = max(32, concurrency * 8)
    connector = aiohttp.TCPConnector(limit=connector_limit, limit_per_host=connector_limit, enable_cleanup_closed=True)
    semaphore = asyncio.Semaphore(concurrency)
    out_dir = SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path).resolve() if out_path else (out_dir / 'eval_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as agent_session:
        pending = [asyncio.create_task(run_episode(spec, agent_session=agent_session, semaphore=semaphore)) for spec in episode_specs]
        for future in asyncio.as_completed(pending):
            item = await future
            results["num_tasks"] += 1
            results["episodes"].append(item["episode"])
            if item["status"] == "success":
                results["successes"] += 1
            elif item["status"] == "failure":
                results["failures"] += 1
            else:
                results["errors"] += 1
                if bool(item.get("model_mismatch_error")):
                    results["model_mismatch_errors"] += 1
            trace_item = item.get("trace_index_item")
            if isinstance(trace_item, dict) and bool(save_act_traces) and trace_root is not None:
                trace_index["episodes"].append(trace_item)
            _save_partial_results(
                out_path,
                results,
                provider=provider_s,
                model=model,
                t_start=t_start,
            )
    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────
    total = results["num_tasks"]
    succ = results["successes"]
    rate = succ / total if total > 0 else 0
    avg_score = (
        sum(ep["score"] for ep in results["episodes"]) / total if total > 0 else 0
    )
    avg_steps = (
        sum(ep["steps"] for ep in results["episodes"]) / total if total > 0 else 0
    )
    avg_task_seconds = (
        sum(ep.get("task_seconds", 0.0) for ep in results["episodes"]) / total if total > 0 else 0
    )
    total_steps = sum(ep["steps"] for ep in results["episodes"])
    avg_step_seconds = (
        sum(ep.get("task_seconds", 0.0) for ep in results["episodes"]) / total_steps
        if total_steps > 0
        else 0
    )

    _compute_results_summary(results, elapsed)
    results["partial"] = False

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Provider:       {provider_s}")
    print(f"  Model:          {model}")
    print(f"  Tasks run:      {total}")
    print(f"  Successes:      {succ}")
    print(f"  Failures:       {results['failures']}")
    print(f"  Errors:         {results['errors']}")
    print(f"  Model mismatch: {results.get('model_mismatch_errors', 0)}")
    print(f"  Success rate:   {rate:.1%}")
    print(f"  Avg score:      {avg_score:.3f}")
    print(f"  Avg steps:      {avg_steps:.1f}")
    print(f"  Avg task time:  {avg_task_seconds:.2f}s")
    total_cost = sum(float(ep.get("estimated_cost_usd", 0.0)) for ep in results["episodes"])
    avg_cost = (total_cost / total) if total > 0 else 0.0
    total_tokens_all = sum(int(ep.get("total_tokens", 0)) for ep in results["episodes"])
    print(f"  Avg step time:  {avg_step_seconds:.2f}s")
    print(f"  Est. cost:      ${total_cost:.4f} (avg ${avg_cost:.4f}/task)")
    print(f"  Total tokens:   {total_tokens_all}")
    print(f"  Total time:     {elapsed:.1f}s")
    print("=" * 60)

    # Per-use-case breakdown
    uc_stats: dict[str, dict] = {}
    for ep in results["episodes"]:
        uc = ep.get("use_case", "unknown")
        if uc not in uc_stats:
            uc_stats[uc] = {"total": 0, "success": 0}
        uc_stats[uc]["total"] += 1
        if ep["success"]:
            uc_stats[uc]["success"] += 1

    if len(uc_stats) > 1:
        print("\n  Per use-case breakdown:")
        for uc, st in sorted(uc_stats.items()):
            uc_rate = st["success"] / st["total"] if st["total"] > 0 else 0
            print(f"    {uc:30s}  {st['success']}/{st['total']}  ({uc_rate:.0%})")
        print()

    # Save results
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}\n")
    if bool(save_act_traces) and trace_root is not None:
        trace_index["summary"] = {
            "num_tasks": int(results.get("num_tasks") or 0),
            "successes": int(results.get("successes") or 0),
            "failures": int(results.get("failures") or 0),
            "errors": int(results.get("errors") or 0),
            "timing": results.get("timing") or {},
            "results_file": str(out_path),
        }
        _json_dump_path(trace_root / "trace_index.json", trace_index)
        print(f"  Act traces saved to: {trace_root}\n")

    if server_proc:
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass
        server_proc.terminate()
        server_proc.wait(timeout=5)

    return results


# ── CLI ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autoppia Operator - LLM Agent Evaluation")
    parser.add_argument('--provider', default='chutes', help='LLM provider: openai|chutes|anthropic')
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3-0324", help="Model name")
    parser.add_argument("--num-tasks", type=int, default=20, help="Number of tasks to evaluate")
    parser.add_argument("--max-steps", type=int, default=15, help="Max steps per episode")
    parser.add_argument("--use-case", default=None, help="Filter by use case (e.g. LOGIN)")
    parser.add_argument("--web-project-id", default=None, help="Filter by web_project_id (exact match)")
    parser.add_argument("--task-id", default=None, help="Run a specific task id (exact match)")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed (otherwise random)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each selected task N times")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument('--out', default=None, help='Output JSON path (default: data/eval_results.json)')
    parser.add_argument('--task-cache', default=None, help='Task cache JSON path')
    parser.add_argument("--strict-model", action=argparse.BooleanOptionalAction, default=True, help="Fail episodes when effective model != requested model")
    parser.add_argument("--distinct-use-cases", action="store_true", help="Pick tasks with distinct use cases")
    parser.add_argument("--all-use-cases", action="store_true", help="Select all use cases for the given --web-project-id")
    parser.add_argument("--tasks-per-use-case", type=int, default=1, help="When --all-use-cases is set, number of tasks sampled per use case")
    parser.add_argument("--task-concurrency", type=int, default=1, help="Number of episodes to evaluate concurrently")
    parser.add_argument("--list-web-projects", action="store_true", help="List web projects available in --task-cache and exit")
    parser.add_argument("--list-use-cases", action="store_true", help="List use cases (optionally filtered by --web-project-id) and exit")
    parser.add_argument("--save-act-traces", action="store_true", help="Persist /act request-response traces per step")
    parser.add_argument("--trace-dir", default=None, help="Custom trace directory")
    parser.add_argument("--trace-full-payloads", action=argparse.BooleanOptionalAction, default=True, help="Include full payloads (snapshot_html/screenshot/history)")
    parser.add_argument("--include-reasoning", action=argparse.BooleanOptionalAction, default=False, help="Request reasoning from /act and store it in traces")
    args = parser.parse_args()

    cache_path = Path(args.task_cache).resolve() if args.task_cache else TASK_CACHE
    if bool(args.list_web_projects) or bool(args.list_use_cases):
        catalog = build_task_catalog(cache_path)
        if bool(args.list_web_projects) and not bool(args.list_use_cases):
            print("\nWeb projects")
            print("=" * 60)
            for project_id, payload in catalog.items():
                print(f"- {project_id}: {int(payload.get('count') or 0)} task(s)")
            print("=" * 60)
        else:
            print_task_catalog(catalog, only_project=args.web_project_id if args.web_project_id else None)
        return

    asyncio.run(
        run_evaluation(
            provider=args.provider,
            model=args.model,
            num_tasks=args.num_tasks,
            max_steps=args.max_steps,
            use_case=args.use_case,
            web_project_id=args.web_project_id,
            task_id=args.task_id,
            seed=args.seed,
            repeat=args.repeat,
            temperature=args.temperature,
            distinct_use_cases=bool(args.distinct_use_cases),
            out_path=args.out,
            task_cache=args.task_cache,
            strict_model=bool(args.strict_model),
            all_use_cases=bool(args.all_use_cases),
            tasks_per_use_case=max(1, int(args.tasks_per_use_case)),
            task_concurrency=max(1, int(args.task_concurrency)),
            save_act_traces=bool(args.save_act_traces),
            trace_dir=args.trace_dir,
            trace_full_payloads=bool(args.trace_full_payloads),
            include_reasoning=bool(args.include_reasoning),
        )
    )


if __name__ == "__main__":
    main()
