from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def build_focus_cost_report(root: Path) -> dict[str, Any]:
    attempts_path = root / "gold" / "attempts.jsonl"
    rows = _load_jsonl(attempts_path)
    by_seed: dict[int, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}

    for row in rows:
        seed = int(row.get("seed") or 0)
        model = str(row.get("model") or "").strip() or "unknown"
        prompt_tokens = int(row.get("prompt_tokens") or 0)
        completion_tokens = int(row.get("completion_tokens") or 0)
        total_tokens = int(row.get("total_tokens") or 0)
        cost = float(row.get("estimated_cost_usd") or 0.0)

        seed_entry = by_seed.setdefault(
            seed,
            {
                "seed": seed,
                "attempts": 0,
                "successes": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "models": {},
            },
        )
        seed_entry["attempts"] += 1
        seed_entry["successes"] += 1 if bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0 else 0
        seed_entry["prompt_tokens"] += prompt_tokens
        seed_entry["completion_tokens"] += completion_tokens
        seed_entry["total_tokens"] += total_tokens
        seed_entry["estimated_cost_usd"] = round(float(seed_entry["estimated_cost_usd"]) + cost, 6)
        seed_entry["models"][model] = int(seed_entry["models"].get(model) or 0) + 1

        model_entry = by_model.setdefault(
            model,
            {
                "model": model,
                "attempts": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        model_entry["attempts"] += 1
        model_entry["prompt_tokens"] += prompt_tokens
        model_entry["completion_tokens"] += completion_tokens
        model_entry["total_tokens"] += total_tokens
        model_entry["estimated_cost_usd"] = round(float(model_entry["estimated_cost_usd"]) + cost, 6)

    totals = {
        "attempts": len(rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        "estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows), 6),
    }
    return {
        "root": str(root),
        "attempts_path": str(attempts_path),
        "totals": totals,
        "by_seed": [by_seed[key] for key in sorted(by_seed)],
        "by_model": [by_model[key] for key in sorted(by_model)],
    }
