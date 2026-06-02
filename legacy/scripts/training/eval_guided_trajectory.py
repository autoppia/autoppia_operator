#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["DEMO_WEBS_ENDPOINT"] = "http://84.247.180.192"
os.environ["DEMO_WEB_SERVICE_PORT"] = "8090"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.claude_guided_harvester import run_guided_brief
from training.demo_project_context import build_project_context
from training.harvester_support import semantic_event_validation


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("brief"), dict):
        return payload
    if isinstance(payload, dict):
        return {"brief": payload, "meta": {"source": str(path)}}
    raise ValueError(f"Invalid brief payload in {path}")


def _flatten_backend_events(report: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = report.get("episodes") if isinstance(report, dict) else None
    episode = episodes[0] if isinstance(episodes, list) and episodes else None
    guided = episode.get("guided_execution") if isinstance(episode, dict) else None
    flat: list[dict[str, Any]] = []
    if not isinstance(guided, list):
        return flat
    for item in guided:
        if not isinstance(item, dict):
            continue
        for event in item.get("backend_events") or []:
            if isinstance(event, dict):
                flat.append(event)
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick guided-eval loop for Claude Code harvest iteration")
    parser.add_argument("--web-project-id", required=True)
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-cache", required=True)
    parser.add_argument("--brief-file", required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()


    brief_path = Path(args.brief_file).resolve()
    out_path = Path(args.out).resolve()
    summary_path = out_path.with_suffix(".summary.json")
    payload = _load_payload(brief_path)
    report = run_guided_brief(
        web_project_id=str(args.web_project_id),
        use_case=str(args.use_case),
        seed=int(args.seed),
        brief_payload=payload,
        task_cache=Path(args.task_cache).resolve(),
        max_steps=int(args.max_steps),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    episodes = report.get("episodes") if isinstance(report, dict) else None
    episode = episodes[0] if isinstance(episodes, list) and episodes else {}
    backend_events = _flatten_backend_events(report)
    task_row = build_project_context(
        web_project_id=str(args.web_project_id),
        use_case=str(args.use_case),
        task_cache_path=Path(args.task_cache).resolve(),
        seed=int(args.seed),
    )["task_row"]
    semantic = semantic_event_validation(task_row=task_row, report=report)
    official_success = bool((episode or {}).get("success"))
    official_score = float((episode or {}).get("score") or 0.0)
    effective_success = official_success or bool(semantic.get("success"))
    effective_score = max(official_score, 1.0 if effective_success else 0.0)
    summary = {
        "success": effective_success,
        "score": effective_score,
        "official_success": official_success,
        "official_score": official_score,
        "semantic_success": bool(semantic.get("success")),
        "semantic_matched_tests": int(semantic.get("matched_tests") or 0),
        "semantic_total_tests": int(semantic.get("total_tests") or 0),
        "steps": int((episode or {}).get("steps") or 0),
        "final_url": str((episode or {}).get("final_url") or ""),
        "backend_events": backend_events[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
