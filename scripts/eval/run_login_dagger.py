#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.dagger import merge_dagger_episodes, run_dagger_round


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DAgger corrections from a failed-summary split")
    parser.add_argument("--project-id", default="autocinema")
    parser.add_argument("--use-case", default="LOGIN")
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--teacher-provider", default="openai")
    parser.add_argument("--teacher-model", default="gpt-5.4")
    parser.add_argument("--merge-base-episodes", default="")
    parser.add_argument("--merge-output", default="")
    args = parser.parse_args(argv)

    use_case = str(args.use_case).strip().upper()
    env_overrides = {"DEMO_WEBS_ENDPOINT": "http://84.247.180.192"} if use_case == "LOGIN" else {}
    summary = run_dagger_round(
        use_case=use_case,
        source_summary_path=Path(args.source_summary).resolve(),
        split_name=str(args.split_name),
        teacher_provider=str(args.teacher_provider),
        teacher_model=str(args.teacher_model),
        env_overrides=env_overrides,
        project_id=str(args.project_id),
    )
    print(json.dumps(summary, indent=2))
    if args.merge_base_episodes and args.merge_output:
        merged = merge_dagger_episodes(
            base_episodes_path=Path(args.merge_base_episodes).resolve(),
            dagger_teacher_path=REPO_ROOT / "data" / str(args.project_id).strip() / str(use_case).lower() / "dagger" / str(args.split_name) / "teacher_gold_episodes.jsonl",
            output_path=Path(args.merge_output).resolve(),
        )
        print(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
