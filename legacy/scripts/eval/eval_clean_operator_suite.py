#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_seed_spec(raw: str) -> list[int]:
    text = str(raw or "").strip()
    if not text:
        return [1]
    out: list[int] = []
    for part in text.split(","):
        token = str(part).strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            for value in range(start, end + step, step):
                if value not in out:
                    out.append(value)
            continue
        value = int(token)
        if value not in out:
            out.append(value)
    return out or [1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the clean structured operator across multiple seeds")
    parser.add_argument("--project-id", default="autocinema")
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--endpoint", default="http://84.247.180.192")
    parser.add_argument("--out", default="tmp/clean_operator_suite.json")
    args, extra = parser.parse_known_args()

    seeds = parse_seed_spec(args.seeds)
    out_path = (ROOT / str(args.out)).resolve() if not str(args.out).startswith("/") else Path(str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out_path.parent / f"{out_path.stem}_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["WEB_AGENT_RUNTIME"] = "structured"
    env["DEMO_WEBS_ENDPOINT"] = str(args.endpoint)

    rows: list[dict[str, object]] = []
    successes = 0
    score_sum = 0.0
    for seed in seeds:
        seed_out = run_dir / f"seed_{int(seed):04d}.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "eval" / "eval_structured_operator.py"),
            "--project-id", str(args.project_id),
            "--use-case", str(args.use_case),
            "--num-tasks", "1",
            "--repeat", "1",
            "--seed", str(int(seed)),
            "--max-steps", str(int(args.max_steps)),
            "--task-concurrency", str(int(args.task_concurrency)),
            "--provider", str(args.provider),
            "--model", str(args.model),
            "--out", str(seed_out),
        ]
        cmd.extend(extra)
        completed = subprocess.run(cmd, cwd=str(ROOT), env=env)
        payload: dict[str, object] = {}
        if seed_out.exists():
            payload = json.loads(seed_out.read_text(encoding="utf-8"))
        row = {
            "seed": int(seed),
            "returncode": int(completed.returncode),
            "success_rate": float(payload.get("success_rate") or 0.0),
            "avg_score": float(payload.get("avg_score") or 0.0),
            "artifact_path": str(seed_out),
        }
        rows.append(row)
        if row["success_rate"] >= 1.0:
            successes += 1
        score_sum += float(row["avg_score"])

    summary = {
        "project_id": str(args.project_id),
        "use_case": str(args.use_case).upper(),
        "endpoint": str(args.endpoint),
        "provider": str(args.provider),
        "model": str(args.model),
        "seeds": seeds,
        "total": len(rows),
        "successes": successes,
        "success_rate": (float(successes) / float(len(rows))) if rows else 0.0,
        "avg_score": (score_sum / float(len(rows))) if rows else 0.0,
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
