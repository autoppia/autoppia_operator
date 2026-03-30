from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_val(rows: list[dict[str, Any]], *, val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    ratio = min(max(float(val_ratio), 0.0), 0.5)
    if ratio <= 0.0:
        return list(rows), []

    items = list(rows)
    rng = random.Random(int(seed))
    rng.shuffle(items)

    val_count = int(round(len(items) * ratio))
    val_count = min(max(val_count, 1), len(items) - 1) if len(items) > 1 else len(items)
    val = items[:val_count]
    train = items[val_count:]
    return train, val


__all__ = ["split_train_val", "write_jsonl"]
