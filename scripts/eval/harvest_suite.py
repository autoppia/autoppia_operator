#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.focus_use_case import _parse_model_ladder, _parse_seed_spec
from training.focus_pipeline import default_task_cache_for_project
from training.harvest_suite import HarvestSuiteConfig, collect_suite, parse_use_case_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the unified harvester across multiple use cases")
    parser.add_argument("--project-id", default="autocinema", help="Web project ID (e.g. autocinema, autohealth)")
    parser.add_argument("--use-cases", default="all", help="Comma-separated use cases or 'all'")
    parser.add_argument("--seeds", default="1..10", help="Seed spec like 1..10 or 1,2,3")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--attempt-models", default="", help="Optional retry ladder, e.g. gpt-5.4-mini,gpt-5.4")
    parser.add_argument("--strategy", choices=["baseline", "code-aware"], default="baseline")
    parser.add_argument("--task-cache", default="")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--agent-workers", type=int, default=1)
    parser.add_argument("--collect-workers", type=int, default=1)
    parser.add_argument("--brief-dir", default="")
    parser.add_argument("--brief-model", default="gpt-5.4-mini")
    parser.add_argument("--execution-mode", choices=["direct", "operator"], default="operator")
    parser.add_argument("--max-claude-attempts", type=int, default=3)
    parser.add_argument("--max-seeds-per-use-case", type=int, default=0)
    parser.add_argument("--target-gold-per-use-case", type=int, default=0)
    parser.add_argument("--deterministic-only", action="store_true", help="Zero-AI mode: use IWA trajectory plans only, no LLM calls")
    args = parser.parse_args(argv)

    project_id = str(args.project_id or "autocinema").strip() or "autocinema"
    task_cache_arg = str(Path(args.task_cache).resolve()) if str(args.task_cache).strip() else str(default_task_cache_for_project(project_id))
    use_cases = parse_use_case_spec(args.use_cases, project_id=project_id, task_cache_path=Path(task_cache_arg))
    seeds = _parse_seed_spec(args.seeds)
    attempt_models = tuple(_parse_model_ladder(args.attempt_models, args.model)) if str(args.attempt_models).strip() else ()
    config = HarvestSuiteConfig(
        use_cases=tuple(use_cases),
        seeds=tuple(seeds),
        provider=str(args.provider),
        model=str(args.model),
        project_id=project_id,
        task_cache_arg=task_cache_arg,
        attempt_models=attempt_models,
        strategy=str(args.strategy),
        max_steps=int(args.max_steps),
        task_concurrency=int(args.task_concurrency),
        agent_workers=int(args.agent_workers),
        collect_workers=int(args.collect_workers),
        brief_dir=(Path(args.brief_dir).resolve() if str(args.brief_dir).strip() else None),
        brief_model=str(args.brief_model),
        execution_mode=str(args.execution_mode),
        max_claude_attempts=int(args.max_claude_attempts),

        max_seeds_per_use_case=int(args.max_seeds_per_use_case),
        target_gold_per_use_case=int(args.target_gold_per_use_case),
        deterministic_only=bool(args.deterministic_only),
    )
    result = collect_suite(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
