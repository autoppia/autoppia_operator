from __future__ import annotations

import json
from pathlib import Path

from training.dataset_validation import validate_focus_dataset


def test_validate_focus_dataset_passes_for_minimal_valid_layout(tmp_path: Path) -> None:
    root = tmp_path / "focus"
    gold = root / "gold"
    sft = root / "sft"
    trace_dir = gold / "traces" / "seed_0001_baseline"
    trace_file = trace_dir / "episodes" / "ep-1.json"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text("{}", encoding="utf-8")

    episodes = [
        {
            "episode_task_id": "ep-1",
            "success": True,
            "score": 1.0,
            "seed": 1,
            "use_case": "LOGIN",
            "attempt_name": "baseline",
            "result_path": str(gold / "runs" / "seed_0001_baseline.json"),
            "trace_dir": str(trace_dir),
            "trace_file": str(trace_file),
            "operator_version": "step_engine",
            "policy_mode": "direct",
            "task_cache_hash": "abc",
            "prompt_hash": "def",
        }
    ]
    gold.mkdir(parents=True, exist_ok=True)
    (gold / "runs").mkdir(parents=True, exist_ok=True)
    Path(episodes[0]["result_path"]).write_text("{}", encoding="utf-8")
    (gold / "episodes.jsonl").write_text(json.dumps(episodes[0]) + "\n", encoding="utf-8")
    (gold / "attempts.jsonl").write_text(json.dumps(episodes[0]) + "\n", encoding="utf-8")
    (gold / "summary.json").write_text(json.dumps({"gold_episodes_total": 1}) + "\n", encoding="utf-8")

    sft.mkdir(parents=True, exist_ok=True)
    (sft / "train.jsonl").write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")
    (sft / "val.jsonl").write_text("", encoding="utf-8")
    (sft / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": "autocinema.browser_use.sft.v2",
                "base_model": "browser-use/bu-30b-a3b-preview",
                "train_examples": 1,
                "val_examples": 0,
                "successful_episodes_used": 1,
                "source_episodes_path": str(gold / "episodes.jsonl"),
                "source_summary_path": str(gold / "summary.json"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_focus_dataset(root)
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_focus_dataset_reports_missing_trace_and_manifest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "focus"
    gold = root / "gold"
    sft = root / "sft"
    gold.mkdir(parents=True, exist_ok=True)
    sft.mkdir(parents=True, exist_ok=True)

    episode = {
        "episode_task_id": "ep-1",
        "success": True,
        "score": 1.0,
        "seed": 1,
        "use_case": "LOGIN",
        "attempt_name": "baseline",
        "result_path": str(gold / "runs" / "seed_0001_baseline.json"),
        "trace_dir": str(gold / "traces" / "seed_0001_baseline"),
        "trace_file": str(gold / "traces" / "episodes" / "missing.json"),
        "operator_version": "",
        "policy_mode": "",
        "task_cache_hash": "",
        "prompt_hash": "",
    }
    (gold / "episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
    (gold / "attempts.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
    (gold / "summary.json").write_text(json.dumps({"gold_episodes_total": 2}) + "\n", encoding="utf-8")
    (sft / "train.jsonl").write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")
    (sft / "val.jsonl").write_text("", encoding="utf-8")
    (sft / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": "",
                "base_model": "",
                "train_examples": 2,
                "val_examples": 1,
                "successful_episodes_used": 0,
                "source_episodes_path": "",
                "source_summary_path": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_focus_dataset(root)
    assert result["ok"] is False
    issues = set(result["issues"])
    assert "trace_missing:ep-1" in issues
    assert "result_path_missing:ep-1" in issues
    assert "summary_gold_mismatch:2!=1" in issues
    assert "train_examples_mismatch:2!=1" in issues
    assert "val_examples_mismatch:1!=0" in issues
    assert "successful_episodes_used_empty" in issues
    assert "base_model_missing" in issues
    assert "format_version_missing" in issues
    assert "source_episodes_path_missing" in issues
    assert "source_summary_path_missing" in issues
    assert "provenance_missing:operator_version" in issues
