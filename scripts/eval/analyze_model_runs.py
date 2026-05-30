#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(obj: dict[str, Any]) -> dict[str, Any]:
    episodes = obj.get("episodes") if isinstance(obj.get("episodes"), list) else []
    n = len(episodes)
    succ = int(obj.get("successes") or sum(1 for e in episodes if e.get("success")))
    total_steps = sum(int(e.get("steps") or 0) for e in episodes)
    total_task_s = sum(float(e.get("task_seconds") or 0.0) for e in episodes)
    total_tokens = sum(int(e.get("total_tokens") or 0) for e in episodes)
    total_cost = sum(float(e.get("estimated_cost_usd") or 0.0) for e in episodes)
    return {
        "provider": str(obj.get("provider") or ""),
        "model": str(obj.get("model") or ""),
        "episodes": n,
        "successes": succ,
        "success_rate": (succ / n) if n > 0 else 0.0,
        "avg_steps_per_task": (total_steps / n) if n > 0 else 0.0,
        "avg_time_per_task_s": (total_task_s / n) if n > 0 else 0.0,
        "avg_time_per_step_s": (total_task_s / total_steps) if total_steps > 0 else 0.0,
        "avg_tokens_per_task": (total_tokens / n) if n > 0 else 0.0,
        "avg_cost_per_task_usd": (total_cost / n) if n > 0 else 0.0,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze one or more eval JSON files and output comparable metrics.")
    ap.add_argument("--runs", nargs="+", required=True, help="Paths to eval JSON files.")
    ap.add_argument("--out", default=None, help="Optional output JSON path.")
    args = ap.parse_args()

    rows = []
    for p in args.runs:
        path = Path(p).resolve()
        obj = _load(path)
        m = _metrics(obj)
        m["path"] = str(path)
        rows.append(m)

    rows = sorted(
        rows,
        key=lambda x: (
            -float(x["success_rate"]),
            float(x["avg_cost_per_task_usd"]),
            float(x["avg_time_per_task_s"]),
        ),
    )
    payload = {"runs": rows}

    for r in rows:
        print(
            f"{r['model'][:56]:56s} | sr={r['success_rate']:.1%} | ep={int(r['episodes']):3d} | avg_steps={r['avg_steps_per_task']:.2f} | task_s={r['avg_time_per_task_s']:.2f} | step_s={r['avg_time_per_step_s']:.2f} | cost/task=${r['avg_cost_per_task_usd']:.5f} | tok/task={r['avg_tokens_per_task']:.1f}"
        )

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
