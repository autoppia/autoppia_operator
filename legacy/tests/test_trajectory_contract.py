from __future__ import annotations

from pathlib import Path

from training.trajectory_contract import canonical_paths_for_row, repair_episode_row_paths, validate_episode_row


def test_canonical_paths_for_row_uses_gold_layout(tmp_path: Path) -> None:
    gold_root = tmp_path / "gold"
    row = {
        "seed": 12,
        "attempt_name": "baseline",
        "episode_task_id": "ep-12",
    }
    paths = canonical_paths_for_row(gold_root=gold_root, row=row)
    assert paths.result_path == gold_root / "runs" / "seed_0012_baseline.json"
    assert paths.trace_dir == gold_root / "traces" / "seed_0012_baseline"
    assert paths.trace_file == gold_root / "traces" / "seed_0012_baseline" / "episodes" / "ep-12.json"


def test_repair_episode_row_paths_rewrites_legacy_locations(tmp_path: Path) -> None:
    gold_root = tmp_path / "gold"
    row = {
        "seed": 5,
        "attempt_name": "prompt_correction",
        "episode_task_id": "ep-5",
        "result_path": "/old/result.json",
        "trace_dir": "/old/traces",
        "trace_file": "/old/trace.json",
    }
    repaired = repair_episode_row_paths(gold_root=gold_root, row=row)
    assert repaired["result_path"].endswith("gold/runs/seed_0005_prompt_correction.json")
    assert repaired["trace_dir"].endswith("gold/traces/seed_0005_prompt_correction")
    assert repaired["trace_file"].endswith("gold/traces/seed_0005_prompt_correction/episodes/ep-5.json")


def test_validate_episode_row_requires_paths_and_provenance(tmp_path: Path) -> None:
    result_path = tmp_path / "gold" / "runs" / "seed_0001_baseline.json"
    trace_dir = tmp_path / "gold" / "traces" / "seed_0001_baseline"
    trace_file = trace_dir / "episodes" / "ep-1.json"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("{}", encoding="utf-8")
    trace_file.write_text("{}", encoding="utf-8")
    row = {
        "episode_task_id": "ep-1",
        "seed": 1,
        "use_case": "LOGIN",
        "success": True,
        "score": 1.0,
        "result_path": str(result_path),
        "trace_dir": str(trace_dir),
        "trace_file": str(trace_file),
        "attempt_name": "baseline",
        "operator_version": "step_engine",
        "policy_mode": "direct",
        "task_cache_hash": "abc",
        "prompt_hash": "def",
    }
    assert validate_episode_row(row) == []
