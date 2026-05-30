from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPERATOR_ARTIFACT_VERSION = "autoppia-operator.v1"
MIN_VERIFIED_SCORE = 1.0


def _slug(value: str) -> str:
    out = []
    for ch in str(value or "").strip().lower():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {" ", "/", "."}:
            out.append("-")
    return "".join(out).strip("-") or "unknown"


@dataclass(frozen=True)
class OperatorArtifactPaths:
    root: Path
    task_context: Path
    attempts_jsonl: Path
    final_trajectory: Path
    final_report: Path
    iwa_score: Path
    backend_events: Path
    screenshots_dir: Path

    @classmethod
    def for_run(cls, root: Path) -> "OperatorArtifactPaths":
        root = Path(root)
        return cls(
            root=root,
            task_context=root / "task_context.json",
            attempts_jsonl=root / "attempts.jsonl",
            final_trajectory=root / "trajectory.json",
            final_report=root / "final_report.json",
            iwa_score=root / "iwa_score.json",
            backend_events=root / "backend_events.json",
            screenshots_dir=root / "screenshots",
        )


def operator_run_root(*, base_dir: Path, web_project_id: str, use_case: str, seed: int) -> Path:
    return (
        Path(base_dir)
        / "operator_runs"
        / _slug(web_project_id)
        / _slug(use_case)
        / f"seed_{int(seed):04d}"
    )


def is_verified_success(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    success = bool(report.get("success"))
    score = report.get("score")
    if not success:
        episodes = report.get("episodes")
        if isinstance(episodes, list) and episodes:
            first = episodes[0] if isinstance(episodes[0], dict) else {}
            success = bool(first.get("success"))
            score = first.get("score", score)
    try:
        score_f = float(score or 0.0)
    except Exception:
        score_f = 0.0
    return success and score_f >= MIN_VERIFIED_SCORE


def build_operator_manifest(
    *,
    web_project_id: str,
    use_case: str,
    seed: int,
    paths: OperatorArtifactPaths,
    final_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_version": OPERATOR_ARTIFACT_VERSION,
        "role": "trajectory_discovery_operator",
        "web_project_id": str(web_project_id),
        "use_case": str(use_case).strip().upper(),
        "seed": int(seed),
        "verified_success": is_verified_success(final_report),
        "paths": {
            "root": str(paths.root),
            "task_context": str(paths.task_context),
            "attempts_jsonl": str(paths.attempts_jsonl),
            "final_trajectory": str(paths.final_trajectory),
            "final_report": str(paths.final_report),
            "iwa_score": str(paths.iwa_score),
            "backend_events": str(paths.backend_events),
            "screenshots_dir": str(paths.screenshots_dir),
        },
        "distillation_ready": is_verified_success(final_report),
    }
