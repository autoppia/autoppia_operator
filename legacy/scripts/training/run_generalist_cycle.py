#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generalist cycle: prepare campaign, optionally train, then clean eval")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--use-cases", default="all")
    parser.add_argument("--existing-pod-id", default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--all-use-cases-eval-tasks", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-now", action="store_true")
    parser.add_argument("--eval-clean-use-case", default="")
    parser.add_argument("--eval-clean-seeds", default="1-10")
    parser.add_argument("--endpoint", default="http://84.247.180.192")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    prepare_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "training" / "run_autocinema_training.py"),
        "--run-name", str(args.run_name),
        "--use-cases", str(args.use_cases),
        "--existing-pod-id", str(args.existing_pod_id),
        "--epochs", str(int(args.epochs)),
        "--lora-rank", str(int(args.lora_rank)),
        "--all-use-cases-eval-tasks", str(int(args.all_use_cases_eval_tasks)),
    ]
    subprocess.run(prepare_cmd, cwd=str(ROOT), check=True)

    run_root = ROOT / "data" / "autocinema_multi" / str(args.run_name)
    plan_path = run_root / "campaign_plan.json"
    plan = _read_json(plan_path)
    result: dict[str, object] = {
        "run_name": str(args.run_name),
        "campaign_plan_path": str(plan_path),
        "campaign_plan": plan,
    }

    if bool(args.train_now):
        train_cmd = [str(part) for part in list(plan.get("train_command") or [])]
        if train_cmd:
            subprocess.run(train_cmd, cwd=str(ROOT), check=True, env=os.environ.copy())
            result["train_executed"] = True
        else:
            result["train_executed"] = False

    if str(args.eval_clean_use_case).strip():
        eval_out = run_root / f"clean_eval_{str(args.eval_clean_use_case).strip().lower()}.json"
        eval_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "eval" / "eval_clean_operator_suite.py"),
            "--project-id", "autocinema",
            "--use-case", str(args.eval_clean_use_case),
            "--seeds", str(args.eval_clean_seeds),
            "--max-steps", str(int(args.max_steps)),
            "--provider", str(args.provider),
            "--model", str(args.model),
            "--endpoint", str(args.endpoint),
            "--out", str(eval_out),
        ]
        subprocess.run(eval_cmd, cwd=str(ROOT), check=True)
        result["clean_eval_path"] = str(eval_out)
        result["clean_eval"] = _read_json(eval_out)

    if bool(args.prepare_only):
        result["prepare_only"] = True

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
