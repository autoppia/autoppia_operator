#!/usr/bin/env python3
"""IWAP helper tools used for automated sn36 tuning.

This module is intentionally mock-first until IWAP endpoints are finalized.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_MOCK_DATA = {
    "current_season": 36,
    "rounds": [
        {
            "round_id": "sn36-r-0012",
            "season_id": 36,
            "status": "completed",
            "started_at": "2026-02-20T00:00:00Z",
            "finished_at": "2026-02-21T00:00:00Z",
            "runs": [
                {
                    "uid": 11,
                    "tasks": 10,
                    "success_rate": 0.72,
                    "avg_score": 0.61,
                    "total_cost_usd": 0.45,
                },
                {
                    "uid": 12,
                    "tasks": 10,
                    "success_rate": 0.84,
                    "avg_score": 0.77,
                    "total_cost_usd": 0.39,
                },
                {
                    "uid": 18,
                    "tasks": 10,
                    "success_rate": 0.55,
                    "avg_score": 0.41,
                    "total_cost_usd": 0.31,
                },
            ],
            "task_logs": [
                {"task_id": "ts-001", "uid": 11, "score": 0.63},
                {"task_id": "ts-002", "uid": 12, "score": 0.81},
            ],
        },
        {
            "round_id": "sn36-r-0011",
            "season_id": 36,
            "status": "completed",
            "started_at": "2026-02-19T00:00:00Z",
            "finished_at": "2026-02-20T00:00:00Z",
            "runs": [
                {
                    "uid": 11,
                    "tasks": 10,
                    "success_rate": 0.68,
                    "avg_score": 0.57,
                    "total_cost_usd": 0.54,
                },
                {
                    "uid": 12,
                    "tasks": 10,
                    "success_rate": 0.79,
                    "avg_score": 0.71,
                    "total_cost_usd": 0.33,
                },
                {
                    "uid": 18,
                    "tasks": 10,
                    "success_rate": 0.50,
                    "avg_score": 0.38,
                    "total_cost_usd": 0.28,
                },
            ],
            "task_logs": [
                {"task_id": "ts-003", "uid": 11, "score": 0.54},
                {"task_id": "ts-004", "uid": 12, "score": 0.72},
            ],
        },
    ],
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _to_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {}


def _load_mock_tasks(payload: dict[str, Any], *, season_id: int | None = None) -> list[dict[str, Any]]:
    rounds = list(payload.get("rounds") or [])
    if season_id is not None:
        rounds = [r for r in rounds if r.get("season_id") == season_id]

    all_tasks: list[dict[str, Any]] = []
    for rd in rounds:
        run_id = rd.get("round_id") or rd.get("id") or ""
        task_logs = rd.get("task_logs") or []
        for log in task_logs:
            if not isinstance(log, dict):
                continue
            task = dict(log)
            task.setdefault("round_id", run_id)
            task.setdefault("season_id", rd.get("season_id"))
            task.setdefault("round_status", rd.get("status", "unknown"))
            task.setdefault("run_id", run_id)
            all_tasks.append(task)
    return all_tasks


def _summarize_task_row(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "uid": task.get("uid"),
        "score": float(task.get("score", 0.0) or 0.0),
        "run_id": task.get("run_id") or task.get("round_id"),
        "season_id": task.get("season_id"),
        "round_status": task.get("round_status", "unknown"),
    }


def _load_mock(path: str | None) -> dict[str, Any]:
    if path:
        fp = Path(path).expanduser()
        if fp.exists():
            try:
                return _to_dict(json.loads(fp.read_text(encoding="utf-8")))
            except Exception:
                pass
    return dict(DEFAULT_MOCK_DATA)


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_last_round(args: argparse.Namespace) -> int:
    data = _load_mock(args.mock_data)
    rounds = list(data.get("rounds") or [])
    if not rounds:
        print(json.dumps({"ok": False, "error": "No mock IWAP rounds available."}, indent=2))
        return 1

    selected = rounds[0]
    payload = {
        "ok": True,
        "source": "mock",
        "as_of": _now_iso(),
        "round": selected,
    }
    _print(payload)
    return 0


def cmd_current_season(args: argparse.Namespace) -> int:
    data = _load_mock(args.mock_data)
    season_id = args.season_id or data.get("current_season")
    rounds = [r for r in (data.get("rounds") or []) if r.get("season_id") == season_id]
    if not rounds:
        payload = {
            "ok": False,
            "source": "mock",
            "season_id": season_id,
            "message": "No rounds found for this season in mock data.",
        }
        _print(payload)
        return 1

    by_uid: dict[int, dict[str, float | int]] = defaultdict(lambda: {"tasks": 0, "success": 0.0, "score": 0.0, "runs": 0})
    for run in rounds:
        for item in run.get("runs") or []:
            uid = int(item.get("uid", -1))
            if uid < 0:
                continue
            state = by_uid[uid]
            state["tasks"] = int(state["tasks"]) + int(item.get("tasks", 0))
            state["runs"] = int(state["runs"]) + 1
            state["success"] = float(state["success"]) + float(item.get("success_rate", 0.0))
            state["score"] = float(state["score"]) + float(item.get("avg_score", 0.0))

    summary_rows = []
    for uid, stats in by_uid.items():
        runs_count = int(stats["runs"])
        summary_rows.append(
            {
                "uid": uid,
                "round_count": runs_count,
                "tasks": int(stats["tasks"]),
                "avg_success_rate": float(stats["success"]) / max(1, runs_count),
                "avg_score": float(stats["score"]) / max(1, runs_count),
            }
        )
    summary_rows.sort(key=lambda item: item["avg_score"], reverse=True)

    payload = {
        "ok": True,
        "source": "mock",
        "season_id": season_id,
        "round_count": len(rounds),
        "uid_summary": summary_rows,
        "rounds": rounds,
    }
    _print(payload)
    return 0


def cmd_top_uids(args: argparse.Namespace) -> int:
    data = _load_mock(args.mock_data)
    rounds = list(data.get("rounds") or [])
    if not rounds:
        _print({"ok": False, "source": "mock", "error": "No mock data."})
        return 1

    bucket: dict[int, list[float]] = defaultdict(list)
    for run in rounds:
        for entry in run.get("runs") or []:
            uid = int(entry.get("uid", -1))
            if uid >= 0:
                bucket[uid].append(float(entry.get("avg_score", 0.0)))

    leaderboard = []
    for uid, scores in bucket.items():
        if not scores:
            continue
        leaderboard.append(
            {
                "uid": uid,
                "run_count": len(scores),
                "avg_score": sum(scores) / len(scores),
            }
        )

    leaderboard.sort(key=lambda item: item["avg_score"], reverse=True)
    payload = {
        "ok": True,
        "source": "mock",
        "top": leaderboard[: max(1, int(args.limit))],
    }
    _print(payload)
    return 0


def cmd_rounds(args: argparse.Namespace) -> int:
    data = _load_mock(args.mock_data)
    rounds = list(data.get("rounds") or [])
    rounds.reverse()
    if args.limit:
        rounds = rounds[: int(args.limit)]

    payload = {
        "ok": True,
        "source": "mock",
        "round_count": len(rounds),
        "rounds": rounds,
    }
    _print(payload)
    return 0


def _fetch_iwap_season_tasks(base_url: str, headers: dict[str, str], season_id: int | None) -> list[dict[str, Any]]:
    if not base_url:
        return []

    cleaned = base_url.rstrip("/")
    if not season_id:
        raise RuntimeError("season_id is required for live IWAP task fetch")

    candidates = [
        f"/api/v1/seasons/{season_id}/tasks",
        f"/api/v1/season/{season_id}/tasks",
        f"/api/v1/task-logs?season_id={season_id}",
    ]

    with httpx.Client(base_url=cleaned, headers=headers, timeout=20.0) as client:
        for path in candidates:
            try:
                response = client.get(path)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") if isinstance(payload, dict) else payload
                if isinstance(data, dict) and "tasks" in data:
                    tasks = data.get("tasks") or []
                elif isinstance(data, list):
                    tasks = data
                elif isinstance(payload, dict):
                    tasks = payload.get("tasks") or []
                else:
                    tasks = []
                if isinstance(tasks, list):
                    return [dict(t) for t in tasks if isinstance(t, dict)]
            except Exception:
                continue

    return []


def cmd_season_tasks(args: argparse.Namespace) -> int:
    data = _load_mock(args.mock_data)
    season_id = args.season_id or data.get("current_season")
    if not season_id:
        print(
            json.dumps(
                {"ok": False, "error": "No season id available."},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1

    base_url = (args.base_url or os.getenv("IWAP_BASE_URL", "")).strip()
    token = args.token or os.getenv("IWAP_API_TOKEN")
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    source = "mock"
    tasks: list[dict[str, Any]] = []
    if args.prefer_live and base_url:
        live_tasks = _fetch_iwap_season_tasks(base_url, headers, int(season_id))
        if live_tasks:
            source = "live"
            tasks = live_tasks
        else:
            print("[iwap-tools] live IWAP task endpoint unavailable, falling back to mock payload")

    if not tasks:
        tasks = _load_mock_tasks(data, season_id=int(season_id))

    if args.uid is not None:
        tasks = [t for t in tasks if int(t.get("uid") or -1) == int(args.uid)]
    if args.min_score is not None:
        tasks = [t for t in tasks if float(t.get("score", 0.0) or 0.0) >= float(args.min_score)]

    if args.max_score is not None:
        tasks = [t for t in tasks if float(t.get("score", 0.0) or 0.0) <= float(args.max_score)]

    if args.limit:
        tasks = tasks[: int(args.limit)]

    rows = [_summarize_task_row(task) for task in tasks]
    summary = {
        "ok": True,
        "source": source,
        "season_id": int(season_id),
        "count": len(rows),
        "limit": int(args.limit) if args.limit else 0,
        "tasks": rows,
    }
    _print(summary)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IWAP tooling for sn36 operator iteration (mock-first)")
    parser.add_argument(
        "--mock-data",
        default=os.getenv("IWAP_MOCK_DATA", ""),
        help="JSON file override for mock IWAP data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_last = subparsers.add_parser("last-round", help="get latest IWAP round summary")
    p_last.set_defaults(func=cmd_last_round)

    p_season = subparsers.add_parser("season-results", help="aggregate all rounds for one season")
    p_season.add_argument("--season-id", type=int, default=None)
    p_season.set_defaults(func=cmd_current_season)

    p_top = subparsers.add_parser("top-uids", help="top UIDs by average score in mock IWAP rounds")
    p_top.add_argument("--limit", type=int, default=10)
    p_top.set_defaults(func=cmd_top_uids)

    p_rounds = subparsers.add_parser("rounds", help="list recent mock IWAP rounds")
    p_rounds.add_argument("--limit", type=int, default=0, help="0 = all")
    p_rounds.set_defaults(func=cmd_rounds)

    p_season_tasks = subparsers.add_parser("season-tasks", help="get season-level task rows from IWAP (or mock fallback)")
    p_season_tasks.add_argument("--season-id", type=int, default=None)
    p_season_tasks.add_argument("--uid", type=int, default=None, help="Filter tasks for one UID")
    p_season_tasks.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Filter tasks with score >= min-score",
    )
    p_season_tasks.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="Filter tasks with score <= max-score",
    )
    p_season_tasks.add_argument("--limit", type=int, default=0, help="0 = all")
    p_season_tasks.add_argument("--base-url", default=os.getenv("IWAP_BASE_URL", ""), help="IWAP API base URL")
    p_season_tasks.add_argument("--token", default=os.getenv("IWAP_API_TOKEN", ""), help="Optional Bearer token")
    p_season_tasks.add_argument(
        "--prefer-live",
        action="store_true",
        help="Try live IWAP endpoint before mock fallback",
    )
    p_season_tasks.set_defaults(func=cmd_season_tasks)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        print(f"[iwap-tools] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
