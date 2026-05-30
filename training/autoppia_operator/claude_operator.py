from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.autoppia_operator.artifacts import (
    OperatorArtifactPaths,
    build_operator_manifest,
    operator_run_root,
)
from training.autoppia_operator.discovery import run_claude_code_harvest
from training.claude_guided_harvester import run_guided_brief


@dataclass(frozen=True)
class OperatorRunConfig:
    web_project_id: str
    use_case: str
    seed: int
    task_cache: Path
    output_dir: Path
    claude_model: str = "claude-sonnet-4-5"
    max_steps: int = 12
    timeout_seconds: int = 600
    max_attempts: int = 3


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_autoppia_operator(config: OperatorRunConfig) -> dict[str, Any]:
    """Run the Claude/IWA trajectory-discovery operator for one task seed.

    This is intentionally not a `/act` runtime. It is an offline discovery loop
    that can use Claude Code, IWA, evaluator feedback, and replay until it finds
    a verified trajectory suitable for later distillation.
    """
    root = operator_run_root(
        base_dir=config.output_dir,
        web_project_id=config.web_project_id,
        use_case=config.use_case,
        seed=config.seed,
    )
    paths = OperatorArtifactPaths.for_run(root)
    paths.screenshots_dir.mkdir(parents=True, exist_ok=True)

    brief_path = paths.root / "candidate_brief.json"
    eval_out = paths.root / "last_eval.json"
    eval_command = (
        "DEMO_WEBS_ENDPOINT=http://84.247.180.192 "
        "python scripts/training/eval_guided_trajectory.py "
        f"--web-project-id {config.web_project_id} "
        f"--use-case {config.use_case.upper()} "
        f"--seed {int(config.seed)} "
        f"--task-cache {Path(config.task_cache).resolve()} "
        f"--brief-file {brief_path} "
        f"--max-steps {int(config.max_steps)} "
        f"--out {eval_out}"
    )
    brief_payload = run_claude_code_harvest(
        web_project_id=config.web_project_id,
        use_case=config.use_case.upper(),
        seed=int(config.seed),
        model=config.claude_model,
        task_cache_path=Path(config.task_cache).resolve(),
        work_dir=paths.root,
        eval_command=eval_command,
        timeout_seconds=int(config.timeout_seconds),
        max_attempts=max(1, int(config.max_attempts)),
    )
    report = run_guided_brief(
        web_project_id=config.web_project_id,
        use_case=config.use_case.upper(),
        seed=int(config.seed),
        brief_payload=brief_payload,
        task_cache=Path(config.task_cache).resolve(),
        max_steps=int(config.max_steps),
    )
    _write_json(paths.final_report, report if isinstance(report, dict) else {"report": report})
    _write_json(paths.final_trajectory, {"brief": brief_payload.get("brief"), "report": report})
    manifest = build_operator_manifest(
        web_project_id=config.web_project_id,
        use_case=config.use_case,
        seed=config.seed,
        paths=paths,
        final_report=report if isinstance(report, dict) else {},
    )
    _write_json(paths.root / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Claude/IWA Autoppia Operator trajectory-discovery loop")
    parser.add_argument("--web-project-id", required=True)
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--claude-model", default="claude-sonnet-4-5")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    manifest = run_autoppia_operator(
        OperatorRunConfig(
            web_project_id=args.web_project_id,
            use_case=args.use_case,
            seed=args.seed,
            task_cache=args.task_cache,
            output_dir=args.output_dir,
            claude_model=args.claude_model,
            max_steps=args.max_steps,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
        )
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
