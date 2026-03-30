from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import split_train_val, write_jsonl
from .models import TrajectoryRecord
from .pipeline import build_ppo_bootstrap_rows


@dataclass
class ExportStats:
    total_trajectories: int
    sft_rows: int = 0
    sft_train_rows: int = 0
    sft_val_rows: int = 0
    ppo_rows: int = 0


def load_cleaned_trajectories(path: Path) -> list[TrajectoryRecord]:
    rows: list[TrajectoryRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(TrajectoryRecord.from_dict(payload))
    return rows


def export_sft(
    *,
    records: list[TrajectoryRecord],
    out_dir: Path,
    val_ratio: float,
    seed: int,
    system_prompt: str | None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = [r.to_sft_record(system_prompt=system_prompt) for r in records]
    train_rows, val_rows = split_train_val(all_rows, val_ratio=val_ratio, seed=seed)

    all_path = out_dir / "sft_all.jsonl"
    train_path = out_dir / "sft_train.jsonl"
    val_path = out_dir / "sft_val.jsonl"

    write_jsonl(all_path, all_rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    return {
        "sft_all": str(all_path),
        "sft_train": str(train_path),
        "sft_val": str(val_path),
        "stats": {
            "total_trajectories": len(records),
            "sft_rows": len(all_rows),
            "sft_train_rows": len(train_rows),
            "sft_val_rows": len(val_rows),
        },
    }


def export_ppo_bootstrap(
    *,
    records: list[TrajectoryRecord],
    out_path: Path,
    terminal_bonus: float = 1.0,
) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_ppo_bootstrap_rows(records, terminal_bonus=terminal_bonus)
    write_jsonl(out_path, rows)

    success_terms = sum(1 for r in rows if bool(r.get("done")) and bool(r.get("success")))
    avg_reward = (sum(float(r.get("reward") or 0.0) for r in rows) / len(rows)) if rows else 0.0

    return {
        "ppo_bootstrap": str(out_path),
        "stats": {
            "total_trajectories": len(records),
            "ppo_rows": len(rows),
            "terminal_successes": success_terms,
            "avg_step_reward": avg_reward,
        },
    }


__all__ = [
    "ExportStats",
    "export_ppo_bootstrap",
    "export_sft",
    "load_cleaned_trajectories",
]
