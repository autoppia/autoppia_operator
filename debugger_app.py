from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Operator Debugger")

ROOT = Path(__file__).resolve().parent
DEFAULT_TRACE_DIR = os.getenv("OPERATOR_DEBUG_TRACE_DIR", "").strip()
DEFAULT_TASK_CACHE = os.getenv(
    "OPERATOR_DEBUG_TASK_CACHE",
    str((ROOT / "data" / "task_cache" / "tasks_5_projects.json").resolve()),
).strip()
TRACE_SCAN_ROOTS = [
    ROOT / "data" / "debug_runs",
    ROOT / "data" / "act_traces",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _resolve_trace_dir(raw: str | None = None) -> Path:
    value = str(raw or DEFAULT_TRACE_DIR or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="trace_dir_missing")
    path = Path(value).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"trace_dir_not_found:{path}")
    trace_index = path / "trace_index.json"
    if not trace_index.exists():
        raise HTTPException(status_code=404, detail=f"trace_index_missing:{trace_index}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail=f"invalid_json_object:{path}")
    return data


def _pretty_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, indent=2, ensure_ascii=False, sort_keys=True)


def _unified_diff(before: str, after: str, *, from_label: str, to_label: str, limit: int = 400) -> str:
    lines = list(
        difflib.unified_diff(
            str(before or "").splitlines(),
            str(after or "").splitlines(),
            fromfile=from_label,
            tofile=to_label,
            lineterm="",
            n=2,
        )
    )
    if len(lines) > limit:
        lines = lines[:limit] + [f"... diff truncated ({len(lines) - limit} more lines)"]
    return "\n".join(lines)


def _normalize_image_data(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    data = raw.strip()
    if not data:
        return None
    if data.startswith("data:image/"):
        return data
    return f"data:image/png;base64,{data}"


def _scan_trace_dirs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for root in TRACE_SCAN_ROOTS:
        if not root.exists():
            continue
        for trace_index_path in root.rglob("trace_index.json"):
            trace_dir = trace_index_path.parent.resolve()
            with contextlib.suppress(Exception):
                trace_index = _load_json(trace_index_path)
                episodes = trace_index.get("episodes") if isinstance(trace_index.get("episodes"), list) else []
                items.append(
                    {
                        "trace_dir": str(trace_dir),
                        "name": trace_dir.name,
                        "project": str(trace_index.get("web_project_id") or ""),
                        "model": str(trace_index.get("model") or ""),
                        "provider": str(trace_index.get("provider") or ""),
                        "created_at_utc": str(trace_index.get("created_at_utc") or ""),
                        "episodes": len(episodes),
                        "source_root": str(root.resolve()),
                    }
                )
    items.sort(
        key=lambda item: (
            item.get("created_at_utc") or "",
            item.get("trace_dir") or "",
        ),
        reverse=True,
    )
    return items


def _compact_step_summary(step: dict[str, Any]) -> dict[str, Any]:
    before = step.get("before") if isinstance(step.get("before"), dict) else {}
    after = step.get("after") if isinstance(step.get("after"), dict) else {}
    agent = step.get("agent") if isinstance(step.get("agent"), dict) else {}
    execution = step.get("execution") if isinstance(step.get("execution"), dict) else {}
    actions = step.get("actions") if isinstance(step.get("actions"), list) else []
    return {
        "step_index": int(step.get("step_index") or 0),
        "before_url": str(before.get("url") or ""),
        "after_url": str(after.get("url") or ""),
        "before_score": float(before.get("score") or 0.0),
        "after_score": float(after.get("score") or 0.0),
        "done": bool(agent.get("done")),
        "action_types": [str((a or {}).get("type") or "") for a in actions if isinstance(a, dict)],
        "exec_ok": bool(execution.get("exec_ok", True)),
        "error": str(execution.get("error") or ""),
        "reasoning": str(agent.get("reasoning") or ""),
    }


def _annotate_step(step: dict[str, Any]) -> dict[str, Any]:
    before = step.get("before") if isinstance(step.get("before"), dict) else {}
    after = step.get("after") if isinstance(step.get("after"), dict) else {}
    agent = step.get("agent") if isinstance(step.get("agent"), dict) else {}
    state_in = agent.get("state_in") if isinstance(agent.get("state_in"), dict) else {}
    state_out = agent.get("state_out") if isinstance(agent.get("state_out"), dict) else {}
    before_html = str(before.get("html") or "")
    after_html = str(after.get("html") or "")
    before_shot = _normalize_image_data(before.get("screenshot"))
    after_shot = _normalize_image_data(after.get("screenshot"))
    out = dict(step)
    out["before"] = dict(before)
    out["after"] = dict(after)
    out["before"]["screenshot"] = before_shot
    out["after"]["screenshot"] = after_shot
    out["diffs"] = {
        "state": _unified_diff(
            _pretty_json(state_in),
            _pretty_json(state_out),
            from_label="state_in",
            to_label="state_out",
        ),
        "html": _unified_diff(before_html, after_html, from_label="before.html", to_label="after.html"),
    }
    return out


def _load_trace_bundle(trace_dir: Path) -> dict[str, Any]:
    trace_index = _load_json(trace_dir / "trace_index.json")
    episodes_index = trace_index.get("episodes") if isinstance(trace_index.get("episodes"), list) else []
    episodes: list[dict[str, Any]] = []
    for item in episodes_index:
        if not isinstance(item, dict):
            continue
        ep_file = trace_dir / str(item.get("file") or "")
        episode = None
        if ep_file.exists():
            with contextlib.suppress(Exception):
                episode = _load_json(ep_file)
        summary = {
            "episode_task_id": str(item.get("episode_task_id") or ""),
            "task_id": str(item.get("task_id") or ""),
            "use_case": str(item.get("use_case") or ""),
            "success": bool(item.get("success")),
            "score": float(item.get("score") or 0.0),
            "steps": int(item.get("steps") or 0),
            "failure_category": str(item.get("failure_category") or ""),
            "file": str(item.get("file") or ""),
        }
        if isinstance(episode, dict):
            ep_meta = episode.get("episode") if isinstance(episode.get("episode"), dict) else {}
            summary.update(
                {
                    "task_seconds": float(ep_meta.get("task_seconds") or 0.0),
                    "operator_duration_ms": int(ep_meta.get("operator_duration_ms") or 0),
                    "llm_calls": int(ep_meta.get("llm_calls") or 0),
                    "estimated_cost_usd": float(ep_meta.get("estimated_cost_usd") or 0.0),
                }
            )
        episodes.append(summary)
    return {
        "trace_dir": str(trace_dir),
        "trace_index": trace_index,
        "episodes": episodes,
    }


def _load_episode(trace_dir: Path, episode_task_id: str) -> dict[str, Any]:
    bundle = _load_trace_bundle(trace_dir)
    for item in bundle["episodes"]:
        if str(item.get("episode_task_id") or "") != str(episode_task_id):
            continue
        path = trace_dir / str(item.get("file") or "")
        payload = _load_json(path)
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        annotated_steps = [_annotate_step(step) for step in steps if isinstance(step, dict)]
        payload["steps"] = annotated_steps
        payload["step_summaries"] = [_compact_step_summary(step) for step in annotated_steps]
        payload["trace_dir"] = str(trace_dir)
        payload["trace_file"] = str(path)
        return payload
    raise HTTPException(status_code=404, detail=f"episode_not_found:{episode_task_id}")


STATIC_DIR = ROOT / "static"
DEBUGGER_STATIC_DIR = STATIC_DIR / "debugger"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ReplayManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._evaluator = None
        self._paused = False
        self._step_tokens = 0
        self._status: dict[str, Any] = {
            "state": "idle",
            "episode_task_id": None,
            "current_step_index": -1,
            "current_action_index": -1,
            "steps_total": 0,
            "started_at": None,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "current_url": None,
            "error": None,
        }

    def status(self) -> dict[str, Any]:
        payload = dict(self._status)
        payload["paused"] = bool(self._paused)
        payload["step_tokens"] = int(self._step_tokens)
        payload["running"] = bool(self._task and not self._task.done())
        return payload

    async def start(
        self,
        *,
        trace_dir: Path,
        episode_payload: dict[str, Any],
        step_index: int | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._task and not self._task.done():
                raise HTTPException(status_code=409, detail="replay_already_running")
            self._paused = False
            self._step_tokens = 0
            self._status = {
                "state": "starting",
                "episode_task_id": str((episode_payload.get("episode") or {}).get("episode_task_id") or ""),
                "current_step_index": -1,
                "current_action_index": -1,
                "steps_total": len(episode_payload.get("steps") or []),
                "started_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "current_url": None,
                "error": None,
                "trace_dir": str(trace_dir),
                "stop_after_step": (int(step_index) if step_index is not None else None),
            }
            self._task = asyncio.create_task(
                self._run(
                    trace_dir=trace_dir,
                    episode_payload=episode_payload,
                    stop_after_step=step_index,
                )
            )
            return self.status()

    async def pause(self) -> dict[str, Any]:
        self._paused = True
        self._touch(state="paused")
        return self.status()

    async def resume(self) -> dict[str, Any]:
        self._paused = False
        self._touch(state="running")
        return self.status()

    async def step_once(self) -> dict[str, Any]:
        self._paused = True
        self._step_tokens += 1
        self._touch(state="paused")
        return self.status()

    async def reset(self) -> dict[str, Any]:
        async with self._lock:
            if self._task and not self._task.done():
                self._task.cancel()
                with contextlib.suppress(Exception):
                    await self._task
            self._task = None
            if self._evaluator is not None:
                with contextlib.suppress(Exception):
                    await self._evaluator.close()
            self._evaluator = None
            self._paused = False
            self._step_tokens = 0
            self._status = {
                "state": "idle",
                "episode_task_id": None,
                "current_step_index": -1,
                "current_action_index": -1,
                "steps_total": 0,
                "started_at": None,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "current_url": None,
                "error": None,
            }
            return self.status()

    async def _wait_turn(self) -> None:
        while self._paused:
            if self._step_tokens > 0:
                self._step_tokens -= 1
                return
            await asyncio.sleep(0.1)

    def _touch(self, **updates: Any) -> None:
        self._status.update(updates)
        self._status["updated_at"] = datetime.utcnow().isoformat() + "Z"

    async def _run(
        self,
        *,
        trace_dir: Path,
        episode_payload: dict[str, Any],
        stop_after_step: int | None,
    ) -> None:
        from autoppia_iwa.src.data_generation.tasks.classes import Task
        from autoppia_iwa.src.execution.actions.base import BaseAction

        import eval as eval_mod

        previous_headless = getattr(eval_mod, "EVALUATOR_HEADLESS", True)
        episode_meta = episode_payload.get("episode") if isinstance(episode_payload.get("episode"), dict) else {}
        episode_task_id = str(episode_meta.get("episode_task_id") or "")
        task_id = str(episode_meta.get("task_id") or "")
        web_agent_id = f"debug-replay-{int(time.time())}"
        validator_id = f"debug-validator-{int(time.time())}"
        try:
            raw_tasks = eval_mod._load_raw_tasks(Path(DEFAULT_TASK_CACHE))
            task_payload = next((td for td in raw_tasks if str(td.get("id") or "") == task_id), None)
            if not isinstance(task_payload, dict):
                raise RuntimeError(f"task_not_found_in_cache:{task_id}")
            task = Task(**task_payload)
            eval_mod.EVALUATOR_HEADLESS = False
            evaluator = eval_mod._ScopedAsyncStatefulEvaluator(
                task=task,
                web_agent_id=web_agent_id,
                validator_id=validator_id,
                enable_score_cheating=False,
                capture_screenshot=False,
            )
            self._evaluator = evaluator
            step_result = await evaluator.reset()
            self._touch(
                state="running",
                current_url=str(step_result.snapshot.url),
                episode_task_id=episode_task_id,
            )
            for idx, step in enumerate(episode_payload.get("steps") or []):
                if stop_after_step is not None and idx > int(stop_after_step):
                    break
                await self._wait_turn()
                actions = step.get("actions") if isinstance(step, dict) else []
                self._touch(
                    current_step_index=int(idx),
                    current_action_index=-1,
                    current_url=str(step_result.snapshot.url),
                    state="running",
                )
                for action_idx, action_item in enumerate(actions if isinstance(actions, list) else []):
                    await self._wait_turn()
                    raw_action = (action_item.get("raw") if isinstance(action_item, dict) else None) or action_item
                    action = BaseAction.create_action(raw_action) if isinstance(raw_action, dict) else None
                    if action is None:
                        raise RuntimeError(f"action_rehydrate_failed step={idx} action={action_idx}")
                    self._touch(
                        current_step_index=int(idx),
                        current_action_index=int(action_idx),
                        current_url=str(step_result.snapshot.url),
                        action_type=str(getattr(action, "type", "")),
                    )
                    step_result = await evaluator.step(action)
                    self._touch(current_url=str(step_result.snapshot.url))
                await asyncio.sleep(0.15)
            self._touch(state="completed", current_action_index=-1)
        except asyncio.CancelledError:
            self._touch(state="cancelled")
            raise
        except Exception as exc:
            self._touch(state="error", error=str(exc)[:400])
        finally:
            if self._evaluator is not None:
                with contextlib.suppress(Exception):
                    await self._evaluator.close()
            self._evaluator = None
            eval_mod.EVALUATOR_HEADLESS = previous_headless


REPLAY = ReplayManager()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "trace_dir": DEFAULT_TRACE_DIR}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(DEBUGGER_STATIC_DIR / "index.html"))


@app.get("/api/run")
def api_run(trace_dir: str | None = Query(default=None)) -> JSONResponse:
    path = _resolve_trace_dir(trace_dir)
    return JSONResponse(_jsonable(_load_trace_bundle(path)))


@app.get("/api/traces")
def api_traces() -> JSONResponse:
    return JSONResponse(_jsonable({"items": _scan_trace_dirs(), "default_trace_dir": DEFAULT_TRACE_DIR}))


@app.get("/api/episode/{episode_task_id}")
def api_episode(episode_task_id: str, trace_dir: str | None = Query(default=None)) -> JSONResponse:
    path = _resolve_trace_dir(trace_dir)
    return JSONResponse(_jsonable(_load_episode(path, episode_task_id)))


@app.get("/api/replay/status")
def api_replay_status() -> JSONResponse:
    return JSONResponse(_jsonable(REPLAY.status()))


@app.post("/api/replay/start")
async def api_replay_start(
    payload: Annotated[dict[str, Any], Body(default_factory=dict)],
    trace_dir: str | None = Query(default=None),
) -> JSONResponse:
    path = _resolve_trace_dir(trace_dir)
    episode_task_id = str(payload.get("episode_task_id") or "").strip()
    if not episode_task_id:
        raise HTTPException(status_code=400, detail="episode_task_id_required")
    episode_payload = _load_episode(path, episode_task_id)
    step_index = payload.get("step_index")
    status = await REPLAY.start(
        trace_dir=path,
        episode_payload=episode_payload,
        step_index=(int(step_index) if step_index is not None else None),
    )
    return JSONResponse(_jsonable(status))


@app.post("/api/replay/pause")
async def api_replay_pause() -> JSONResponse:
    return JSONResponse(_jsonable(await REPLAY.pause()))


@app.post("/api/replay/resume")
async def api_replay_resume() -> JSONResponse:
    return JSONResponse(_jsonable(await REPLAY.resume()))


@app.post("/api/replay/step")
async def api_replay_step() -> JSONResponse:
    return JSONResponse(_jsonable(await REPLAY.step_once()))


@app.post("/api/replay/reset")
async def api_replay_reset() -> JSONResponse:
    return JSONResponse(_jsonable(await REPLAY.reset()))
