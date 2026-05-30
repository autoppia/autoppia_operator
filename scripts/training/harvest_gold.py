#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Any
from collections import OrderedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["DEMO_WEBS_ENDPOINT"] = "http://84.247.180.192"
os.environ["DEMO_WEB_SERVICE_PORT"] = "8090"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.autoppia_operator.discovery import run_claude_code_harvest
from training.claude_guided_harvester import run_guided_brief
from training.layout import use_case_layout
from training.demo_project_context import build_project_context
from training.harvester_support import semantic_event_validation
from training.trajectory_contract import build_provenance_fields


def _parse_seeds(raw: str) -> list[int]:
    raw = str(raw).strip()
    if ".." in raw:
        start, end = raw.split("..", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _episode_row(*, web_project_id: str, use_case: str, seed: int, report: dict[str, Any], result_path: Path, trace_file: Path, task_cache_path: Path) -> dict[str, Any]:
    episodes = report.get("episodes") if isinstance(report, dict) else None
    episode = episodes[0] if isinstance(episodes, list) and episodes else {}
    row = {
        "web_project_id": web_project_id,
        "task_id": str((episode or {}).get("task_id") or ""),
        "episode_task_id": str((episode or {}).get("episode_task_id") or f"claude-guided-{use_case.lower()}-{seed}"),
        "use_case": use_case,
        "seed": int(seed),
        "success": bool((episode or {}).get("success")),
        "score": float((episode or {}).get("score") or 0.0),
        "steps": int((episode or {}).get("steps") or 0),
        "model": str(report.get("model") or "claude-code"),
        "result_path": str(result_path),
        "trace_dir": str(trace_file.parent.parent),
        "trace_root": str(trace_file.parent.parent),
        "trace_file": str(trace_file),
        "attempt_name": "claude_code",
        "harvest_mode": "claude_code_iterative",
        "final_url": str((episode or {}).get("final_url") or ""),
    }
    row.update(build_provenance_fields(task_cache_path=task_cache_path, prompt_override="", operator_version="guided_replay", policy_mode="claude_code_teacher"))
    return row


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _unique_gold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: "OrderedDict[int, dict[str, Any]]" = OrderedDict()
    for row in rows:
        try:
            seed = int(row.get("seed"))
        except Exception:
            continue
        if not bool(row.get("success")) or float(row.get("score") or 0.0) < 1.0:
            continue
        by_seed[seed] = row
    return [by_seed[key] for key in sorted(by_seed)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical Claude Code gold harvester")
    parser.add_argument("--web-project-id", required=True)
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--task-cache", required=True)
    parser.add_argument("--claude-model", default="claude-sonnet-4-5")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()


    web_project_id = str(args.web_project_id).strip()
    use_case = str(args.use_case).strip().upper()
    task_cache_path = Path(args.task_cache).resolve()
    layout = use_case_layout(repo_root=REPO_ROOT, web_project=web_project_id, use_case=use_case)
    all_rows: list[dict[str, Any]] = []

    for seed in _parse_seeds(args.seeds):
        work_dir = layout.gold_dir / "harvester" / "claude_code" / f"seed_{seed:04d}"
        brief_path = work_dir / "candidate_brief.json"
        eval_out = work_dir / "last_eval.json"
        eval_cmd = (
            f"DEMO_WEBS_ENDPOINT=http://84.247.180.192 python scripts/training/eval_guided_trajectory.py "
            f"--web-project-id {web_project_id} --use-case {use_case} --seed {seed} "
            f"--task-cache {task_cache_path} --brief-file {brief_path} --max-steps {int(args.max_steps)} --out {eval_out}"
        )
        try:
            payload = run_claude_code_harvest(
                web_project_id=web_project_id,
                use_case=use_case,
                seed=int(seed),
                model=str(args.claude_model),
                task_cache_path=task_cache_path,
                work_dir=work_dir,
                eval_command=eval_cmd,
                timeout_seconds=int(args.timeout_seconds),
                max_attempts=max(1, int(args.max_attempts)),
            )
            report = run_guided_brief(
                web_project_id=web_project_id,
                use_case=use_case,
                seed=int(seed),
                brief_payload=payload,
                task_cache=task_cache_path,
                max_steps=int(args.max_steps),
            )
            result_path = layout.runs_dir / f"seed_{seed:04d}_claude_code.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            trace_dir = layout.traces_dir / f"seed_{seed:04d}_claude_code" / "episodes"
            trace_dir.mkdir(parents=True, exist_ok=True)
            episodes = report.get("episodes") if isinstance(report, dict) else None
            episode = episodes[0] if isinstance(episodes, list) and episodes else {}
            trace_file = trace_dir / f"{str((episode or {}).get('episode_task_id') or f'claude-guided-{use_case.lower()}-{seed}')}.json"
            trace_payload = {"episode": episode, "report_model": report.get("model"), "seed": seed, "use_case": use_case, "brief": payload.get("brief")}
            trace_file.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            row = _episode_row(
                web_project_id=web_project_id,
                use_case=use_case,
                seed=int(seed),
                report=report,
                result_path=result_path,
                trace_file=trace_file,
                task_cache_path=task_cache_path,
            )
            task_row = build_project_context(
                web_project_id=web_project_id,
                use_case=use_case,
                task_cache_path=task_cache_path,
                seed=int(seed),
            )["task_row"]
            semantic = semantic_event_validation(task_row=task_row, report=report)
            if not bool(semantic.get("success")) and eval_out.exists():
                try:
                    eval_report = json.loads(eval_out.read_text(encoding="utf-8"))
                    semantic = semantic_event_validation(task_row=task_row, report=eval_report)
                except Exception:
                    pass
            if bool(semantic.get("success")) and not bool(row.get("success")):
                row["success"] = True
                row["score"] = max(float(row.get("score") or 0.0), 1.0)
                row["semantic_success"] = True
                row["semantic_matched_tests"] = int(semantic.get("matched_tests") or 0)
                row["semantic_total_tests"] = int(semantic.get("total_tests") or 0)
        except Exception as exc:
            row = {
                "web_project_id": web_project_id,
                "use_case": use_case,
                "seed": int(seed),
                "success": False,
                "score": 0.0,
                "steps": 0,
                "model": str(args.claude_model),
                "attempt_name": "claude_code",
                "harvest_mode": "claude_code_iterative",
                "error": str(exc),
            }
            row.update(build_provenance_fields(task_cache_path=task_cache_path, prompt_override="", operator_version="guided_replay", policy_mode="claude_code_teacher"))
        _append_jsonl(layout.attempts_path, row)
        if bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0:
            _append_jsonl(layout.episodes_path, row)
        all_rows.append(row)

    attempt_rows = _load_jsonl(layout.attempts_path)
    episode_rows = _load_jsonl(layout.episodes_path)
    gold_rows = _unique_gold_rows(episode_rows)
    summary = {
        "web_project_id": web_project_id,
        "use_case": use_case,
        "num_attempts": len(attempt_rows),
        "num_gold": len(gold_rows),
        "rows": gold_rows,
    }
    layout.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(layout.summary_path), "num_gold": summary["num_gold"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
