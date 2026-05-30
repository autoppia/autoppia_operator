from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from training.autoppia_operator.briefs import REPO_ROOT, generate_claude_brief


def run_claude_code_harvest(
    *,
    web_project_id: str,
    use_case: str,
    seed: int,
    model: str,
    task_cache_path: Path,
    work_dir: Path,
    eval_command: str = "",
    timeout_seconds: int = 600,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Iterate Claude brief generation against evaluator feedback."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    previous_attempts: list[dict[str, Any]] = []
    best_payload: dict[str, Any] = {}
    attempts = max(1, int(max_attempts))
    for attempt_idx in range(1, attempts + 1):
        payload = generate_claude_brief(
            use_case=use_case,
            seed=int(seed),
            model=model,
            previous_attempts=previous_attempts,
            timeout_seconds=max(1, int(timeout_seconds)),
            task_cache_path=Path(task_cache_path).resolve(),
            web_project_id=web_project_id,
        )
        best_payload = payload
        candidate_path = work_dir / "candidate_brief.json"
        attempt_path = work_dir / f"attempt_{attempt_idx:02d}_brief.json"
        candidate_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        attempt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not str(eval_command or "").strip():
            break
        proc = subprocess.run(
            eval_command,
            cwd=str(REPO_ROOT),
            shell=True,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
        eval_record = {
            "attempt": attempt_idx,
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout or "")[-4000:],
            "stderr": str(proc.stderr or "")[-4000:],
        }
        (work_dir / f"attempt_{attempt_idx:02d}_eval_command.json").write_text(
            json.dumps(eval_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report_path = work_dir / "last_eval.json"
        report: dict[str, Any] = {}
        if report_path.exists():
            try:
                loaded = json.loads(report_path.read_text(encoding="utf-8"))
                report = loaded if isinstance(loaded, dict) else {}
            except Exception:
                report = {}
        if proc.returncode == 0 and _report_success(report):
            break
        previous_attempts.append(
            {
                "attempt_name": f"attempt_{attempt_idx:02d}",
                "report": report,
                "command": eval_record,
            }
        )
    final_path = work_dir / "final_brief.json"
    final_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return best_payload


def _report_success(report: dict[str, Any]) -> bool:
    if bool(report.get("success")) and float(report.get("score") or 0.0) >= 1.0:
        return True
    episodes = report.get("episodes")
    if isinstance(episodes, list) and episodes:
        first = episodes[0] if isinstance(episodes[0], dict) else {}
        try:
            return bool(first.get("success")) and float(first.get("score") or 0.0) >= 1.0
        except Exception:
            return False
    return False


__all__ = ["run_claude_code_harvest"]
