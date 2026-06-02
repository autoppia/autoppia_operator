from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.trajectory_candidate import TrajectoryCandidate


def replay_output_paths(*, output_root: Path, seed: int, attempt_name: str) -> tuple[Path, Path]:
    replay_root = output_root / "replays" / f"seed_{int(seed):04d}_{str(attempt_name).strip()}"
    replay_root.mkdir(parents=True, exist_ok=True)
    return replay_root / "result.json", replay_root


def guided_execution_to_candidate_actions(report: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = report.get("episodes") if isinstance(report, dict) else None
    episode = episodes[0] if isinstance(episodes, list) and episodes else None
    execution_log = episode.get("guided_execution") if isinstance(episode, dict) else None
    if not isinstance(execution_log, list):
        return []
    actions: list[dict[str, Any]] = []
    for item in execution_log:
        if not isinstance(item, dict):
            continue
        planned_action = item.get("planned_action")
        if isinstance(planned_action, dict):
            actions.append(dict(planned_action))
    return actions


def candidate_row_from_replay(
    *,
    candidate: TrajectoryCandidate,
    report: dict[str, Any],
    result_path: Path,
) -> dict[str, Any] | None:
    episodes = report.get("episodes") if isinstance(report, dict) else None
    episode = episodes[0] if isinstance(episodes, list) and episodes else None
    if not isinstance(episode, dict):
        return None
    return {
        "web_project_id": "autocinema",
        "task_id": str(episode.get("task_id") or ""),
        "episode_task_id": str(episode.get("episode_task_id") or ""),
        "use_case": candidate.use_case,
        "seed": int(candidate.seed),
        "success": bool(episode.get("success")),
        "score": float(episode.get("score") or 0.0),
        "steps": int(episode.get("steps") or 0),
        "model": str(report.get("model") or candidate.teacher_model or ""),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "usage_breakdown": {},
        "result_path": str(result_path),
        "trace_dir": "",
        "trace_root": "",
        "trace_file": "",
        "trace_ref": str(episode.get("episode_task_id") or ""),
        "attempt_name": candidate.attempt_name,
        "harvest_mode": "candidate_replay",
        "teacher_model": candidate.teacher_model,
        "teacher_brief_path": candidate.brief_path,
        "notes": f"{candidate.use_case} seed={candidate.seed} attempt={candidate.attempt_name}",
        "final_url": str(episode.get("final_url") or ""),
    }


def write_replay_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
