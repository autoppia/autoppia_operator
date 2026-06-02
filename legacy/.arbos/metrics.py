from __future__ import annotations

import json
from pathlib import Path


def compute_metrics(*, project_dir: Path, repo_dir: Path, run_dir: Path) -> dict:
    summary_path = repo_dir / "data" / "autocinema_trajectory_harvest" / "summary.json"
    if not summary_path.exists():
        return {
            "name": "harvest",
            "label": "Harvest",
            "summary": "waiting for autocinema harvest summary",
            "details": ["No summary.json yet under data/autocinema_trajectory_harvest."],
        }

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    per_use_case = data.get("per_use_case") if isinstance(data.get("per_use_case"), dict) else {}
    target = int(data.get("success_target_per_use_case") or 10)
    required_use_cases = data.get("required_use_cases") if isinstance(data.get("required_use_cases"), list) else sorted(per_use_case)
    successes_total = int(data.get("successes_total") or 0)
    failures_total = int(data.get("failures_total") or 0)
    target_hits = 0
    zero_success = 0
    min_successes = None
    weakest: list[str] = []

    for use_case in required_use_cases:
        stats = per_use_case.get(use_case) if isinstance(per_use_case.get(use_case), dict) else {}
        successes = int(stats.get("successes") or 0)
        shortfall = int(stats.get("success_shortfall") or max(target - successes, 0))
        if successes >= target:
            target_hits += 1
        if successes == 0:
            zero_success += 1
        if min_successes is None or successes < min_successes:
            min_successes = successes
        if shortfall > 0 and len(weakest) < 3:
            weakest.append(f"{use_case}: {successes}/{target}")

    total_use_cases = len(required_use_cases)
    score = (target_hits * 1000) + successes_total - (zero_success * 100)
    summary = (
        f"{successes_total} successes · {target_hits}/{total_use_cases} use cases at target"
        f" · {zero_success} at zero · {failures_total} failures kept"
    )
    details = []
    if weakest:
        details.append("Weakest: " + ", ".join(weakest))
    if min_successes is not None:
        details.append(f"Minimum successes in any use case: {min_successes}/{target}")

    return {
        "name": "harvest",
        "label": "Harvest",
        "score": score,
        "summary": summary,
        "details": details,
        "values": {
            "successes_total": successes_total,
            "failures_total": failures_total,
            "use_cases_at_target": target_hits,
            "total_use_cases": total_use_cases,
            "zero_success_use_cases": zero_success,
            "min_successes_per_use_case": min_successes,
            "success_target_per_use_case": target,
        },
    }
