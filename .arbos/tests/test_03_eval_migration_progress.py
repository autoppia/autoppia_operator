#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


EXPECTED_USE_CASES = {
    "ADD_COMMENT",
    "ADD_FILM",
    "ADD_TO_WATCHLIST",
    "CONTACT",
    "DELETE_FILM",
    "EDIT_FILM",
    "EDIT_USER",
    "FILM_DETAIL",
    "FILTER_FILM",
    "LOGIN",
    "LOGOUT",
    "REGISTRATION",
    "REMOVE_FROM_WATCHLIST",
    "SEARCH_FILM",
    "SHARE_MOVIE",
    "WATCH_TRAILER",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    repo = Path(os.environ["ARBOS_TARGET_REPO"]).resolve()
    harvest_dir = repo / "data" / "autocinema_trajectory_harvest"
    summary_path = harvest_dir / "summary.json"
    episodes_path = harvest_dir / "episodes.jsonl"
    manifest_path = harvest_dir / "collection_manifest.json"
    golden_path = harvest_dir / "golden_seeds.json"

    assert summary_path.exists(), "Missing data/autocinema_trajectory_harvest/summary.json"
    assert episodes_path.exists(), "Missing data/autocinema_trajectory_harvest/episodes.jsonl"
    assert manifest_path.exists(), "Missing data/autocinema_trajectory_harvest/collection_manifest.json"
    assert golden_path.exists(), "Missing data/autocinema_trajectory_harvest/golden_seeds.json"

    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)
    golden = _load_json(golden_path)

    assert summary.get("project_id") == "autocinema"
    assert summary.get("branch") == "arbos", f"Expected branch=arbos, got {summary.get('branch')!r}"
    assert summary.get("require_trace_files") is True

    got_use_cases = set(summary.get("use_cases") or [])
    assert got_use_cases == EXPECTED_USE_CASES, (
        "The harvest summary must list every Autocinema use case.\n"
        f"missing={sorted(EXPECTED_USE_CASES - got_use_cases)} extra={sorted(got_use_cases - EXPECTED_USE_CASES)}"
    )

    episodes_total = int(summary.get("episodes_total", 0))
    successes_total = int(summary.get("successes_total", 0))
    failures_total = int(summary.get("failures_total", 0))
    replayable_total = int(summary.get("replayable_episodes_total", 0))
    assert successes_total >= len(EXPECTED_USE_CASES) * 10, (
        "Need at least 10 successful trajectories per use case in the committed harvest."
    )
    assert failures_total >= len(EXPECTED_USE_CASES), "The committed harvest must retain failures too."
    assert replayable_total == episodes_total, "Every committed episode must be replayable."

    per_use_case = summary.get("per_use_case")
    assert isinstance(per_use_case, dict), "summary.json must include per_use_case object."
    assert set(per_use_case) == EXPECTED_USE_CASES, "per_use_case must cover every Autocinema use case."
    for use_case in sorted(EXPECTED_USE_CASES):
        row = per_use_case[use_case]
        attempted = int(row.get("attempted", 0))
        successes = int(row.get("successes", 0))
        failures = int(row.get("failures", 0))
        distinct_seeds = row.get("distinct_seeds") or []
        successful_seeds = row.get("successful_seeds") or []
        assert attempted >= 10, f"{use_case} needs at least 10 attempts."
        assert successes >= 10, f"{use_case} needs at least 10 successful trajectories."
        assert failures >= 1, f"{use_case} should retain at least one failure or near-miss."
        assert len(distinct_seeds) >= 10, f"{use_case} needs at least 10 distinct attempted seeds."
        assert len(successful_seeds) >= 10, f"{use_case} needs at least 10 distinct successful seeds."
        assert len(set(successful_seeds)) >= 10, f"{use_case} successful seeds must be distinct."
        assert int(row.get("golden_seed_count", 0)) >= 10, f"{use_case} needs 10 observed golden seeds."

    lines = [line for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= episodes_total
    records = [json.loads(line) for line in lines]
    for key in ("web_project_id", "use_case", "seed", "success", "episode_task_id", "trace_file", "trace_root"):
        assert key in records[0], f"episodes.jsonl record missing required key: {key}"

    success_records = [row for row in records if bool(row.get("success"))]
    assert len(success_records) >= successes_total
    assert any("correction" in row or "advice" in row or "harvest_mode" in row for row in records), (
        "episodes.jsonl should record correction/advice/harvest metadata when building a DAgger-capable dataset."
    )

    command_sources = manifest.get("command_sources") or []
    assert any(item.get("type") == "fresh_eval" for item in command_sources), (
        "The committed dataset must come from fresh harvest runs."
    )
    assert any(item.get("type") in {"dagger", "advice_loop", "correction_eval"} for item in command_sources), (
        "The manifest must record at least one correction/advice-assisted harvest source."
    )
    assert manifest.get("result_files"), "collection_manifest.json must record result files."
    assert manifest.get("trace_roots"), "collection_manifest.json must record trace roots."

    golden_by_use_case = golden.get("golden_by_use_case") or {}
    assert set(golden_by_use_case) == EXPECTED_USE_CASES, "golden_seeds.json must cover every use case."
    for use_case in sorted(EXPECTED_USE_CASES):
        seeds = golden_by_use_case.get(use_case) or []
        assert len(seeds) >= 10, f"{use_case} needs at least 10 golden seeds recorded."
        assert len(set(seeds)) >= 10, f"{use_case} golden seeds must be distinct."

    print("PASS: Autocinema harvest reached strong replayable coverage with 10 successful distinct-seed trajectories per use case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
