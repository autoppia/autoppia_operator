from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.trajectory_contract import row_identity, validate_episode_row


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def validate_gold_dataset(
    root: Path,
    *,
    require_trace_files: bool = True,
    require_result_paths: bool = True,
) -> dict[str, Any]:
    gold_dir = root / "gold"
    episodes_path = gold_dir / "episodes.jsonl"
    attempts_path = gold_dir / "attempts.jsonl"
    summary_path = gold_dir / "summary.json"
    issues: list[str] = []

    for path in (episodes_path, attempts_path, summary_path):
        if not path.exists():
            issues.append(f"missing:{path.name}")

    if issues:
        return {"ok": False, "root": str(root), "issues": issues}

    episodes = _load_jsonl(episodes_path)
    attempts = _load_jsonl(attempts_path)
    summary = _load_json(summary_path)
    if not episodes:
        issues.append("episodes_empty")
    if not attempts:
        issues.append("attempts_empty")

    attempt_keys = {row_identity(row) for row in attempts if isinstance(row, dict)}
    seen_episode_ids: set[str] = set()
    for row in episodes:
        episode_id = str(row.get("episode_task_id") or "")
        if episode_id in seen_episode_ids:
            issues.append(f"episode_duplicate_id:{episode_id}")
        seen_episode_ids.add(episode_id)
        issues.extend(
            validate_episode_row(
                row,
                require_trace_files=require_trace_files,
                require_result_paths=require_result_paths,
            )
        )
        if row_identity(row) not in attempt_keys:
            issues.append(f"episode_missing_in_attempts:{episode_id or 'unknown'}")

    gold_total = int(summary.get("gold_episodes_total") or 0)
    if gold_total != len(episodes):
        issues.append(f"summary_gold_mismatch:{gold_total}!={len(episodes)}")

    return {
        "ok": not issues,
        "root": str(root),
        "gold_episode_rows": len(episodes),
        "attempt_rows": len(attempts),
        "summary_gold_episodes_total": gold_total,
        "issues": sorted(set(issues)),
    }


def validate_sft_dataset(root: Path, *, sft_dir_name: str = "sft") -> dict[str, Any]:
    sft_dir = root / sft_dir_name
    manifest_path = sft_dir / "manifest.json"
    train_path = sft_dir / "train.jsonl"
    val_path = sft_dir / "val.jsonl"
    issues: list[str] = []

    for path in (manifest_path, train_path, val_path):
        if not path.exists():
            issues.append(f"missing:{path.name}")

    if issues:
        return {"ok": False, "root": str(root), "sft_dir": str(sft_dir), "issues": issues}

    manifest = _load_json(manifest_path)
    train_rows = _load_jsonl(train_path)
    val_rows = _load_jsonl(val_path)

    if manifest.get("train_examples") != len(train_rows):
        issues.append(f"train_examples_mismatch:{manifest.get('train_examples')}!={len(train_rows)}")
    if manifest.get("val_examples") != len(val_rows):
        issues.append(f"val_examples_mismatch:{manifest.get('val_examples')}!={len(val_rows)}")
    if not train_rows:
        issues.append("train_empty")
    if int(manifest.get("successful_episodes_used") or 0) <= 0:
        issues.append("successful_episodes_used_empty")
    if not str(manifest.get("base_model") or "").strip():
        issues.append("base_model_missing")
    if not str(manifest.get("format_version") or "").strip():
        issues.append("format_version_missing")
    if not str(manifest.get("source_episodes_path") or "").strip():
        issues.append("source_episodes_path_missing")
    if not str(manifest.get("source_summary_path") or "").strip():
        issues.append("source_summary_path_missing")

    return {
        "ok": not issues,
        "root": str(root),
        "sft_dir": str(sft_dir),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "issues": sorted(set(issues)),
    }


def validate_focus_dataset(
    root: Path,
    *,
    sft_dir_name: str = "sft",
    require_trace_files: bool = True,
    require_result_paths: bool = True,
) -> dict[str, Any]:
    gold = validate_gold_dataset(
        root,
        require_trace_files=require_trace_files,
        require_result_paths=require_result_paths,
    )
    sft = validate_sft_dataset(root, sft_dir_name=sft_dir_name)
    issues = list(gold.get("issues") or []) + list(sft.get("issues") or [])
    return {
        "ok": not issues,
        "root": str(root),
        "gold": gold,
        "sft": sft,
        "issues": sorted(set(issues)),
    }

