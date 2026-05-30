#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.trajectory_contract import build_provenance_fields, repair_episode_row_paths

LOGIN_ROOT = REPO_ROOT / "data" / "autocinema" / "login"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _repair_sft_manifest(*, root: Path, episodes_path: Path, summary_path: Path) -> None:
    manifest_path = root / "sft" / "manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    payload["source_episodes_path"] = str(episodes_path)
    payload["source_summary_path"] = str(summary_path)
    payload["migration_note"] = "Manifest provenance repaired to canonical login gold layout."
    _write_json(manifest_path, payload)


def migrate_login_dataset(*, drop_unresolved: bool = True) -> dict[str, object]:
    gold_root = LOGIN_ROOT / "gold"
    attempts_path = gold_root / "attempts.jsonl"
    episodes_path = gold_root / "episodes.jsonl"
    summary_path = gold_root / "summary.json"

    attempts = _load_jsonl(attempts_path)
    episodes = _load_jsonl(episodes_path)
    migrated_attempts: list[dict] = []
    migrated_episodes: list[dict] = []
    repaired_rows = 0
    dropped_rows = 0

    for source_rows, out_rows in ((attempts, migrated_attempts), (episodes, migrated_episodes)):
        for row in source_rows:
            repaired = repair_episode_row_paths(gold_root=gold_root, row=row)
            repaired.update(build_provenance_fields(operator_version="step_engine", policy_mode="direct"))
            result_ok = Path(str(repaired.get("result_path") or "")).exists()
            trace_ok = Path(str(repaired.get("trace_file") or "")).exists()
            if result_ok and trace_ok:
                repaired_rows += 1
                out_rows.append(repaired)
                continue
            if drop_unresolved:
                dropped_rows += 1
                continue
            out_rows.append(repaired)

    attempt_keys = {(int(row.get("seed") or 0), str(row.get("attempt_name") or "")) for row in migrated_attempts}
    migrated_episodes = [row for row in migrated_episodes if (int(row.get("seed") or 0), str(row.get("attempt_name") or "")) in attempt_keys]
    migrated_episodes = [row for row in migrated_episodes if bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0]

    summary = {
        "use_case": "LOGIN",
        "gold_episodes_total": len(migrated_episodes),
        "attempts_total": len(migrated_attempts),
        "migrated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "migration_note": "Canonicalized LOGIN dataset paths under data/autocinema/login/gold and backfilled provenance fields.",
    }

    _write_jsonl(attempts_path, migrated_attempts)
    _write_jsonl(episodes_path, migrated_episodes)
    _write_json(summary_path, summary)
    _repair_sft_manifest(root=LOGIN_ROOT, episodes_path=episodes_path, summary_path=summary_path)

    return {
        "attempts_path": str(attempts_path),
        "episodes_path": str(episodes_path),
        "summary_path": str(summary_path),
        "attempt_rows": len(migrated_attempts),
        "episode_rows": len(migrated_episodes),
        "repaired_rows": repaired_rows,
        "dropped_rows": dropped_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair LOGIN dataset paths and provenance to canonical gold layout")
    parser.add_argument("--keep-unresolved", action="store_true", help="Keep rows even if canonical result/trace files are missing")
    args = parser.parse_args()
    result = migrate_login_dataset(drop_unresolved=not bool(args.keep_unresolved))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
