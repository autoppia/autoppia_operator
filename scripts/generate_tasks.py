#!/usr/bin/env python3
"""Generate and cache benchmark tasks using autoppia_iwa's TaskGenerationPipeline.

Why this exists:
- Running ad-hoc python does NOT load this repo's .env by default.
- autoppia_iwa may load its own env or rely on process env vars.
- We want an explicit way to load autoppia_operator/.env (override=True) so
  OPENAI_API_KEY is always the one we expect.

Default output matches what eval.py consumes:
  ../autoppia_rl/data/task_cache/autoppia_cinema_tasks.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def _load_operator_env(operator_dir: Path) -> None:
    # Load dotenv before importing autoppia_iwa so our env wins.
    try:
        from dotenv import load_dotenv
    except Exception as e:  # pragma: no cover
        raise RuntimeError("python-dotenv is required to load .env for task generation") from e

    env_path = operator_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


async def _generate(project_id: str, prompts_per_use_case: int, dynamic: bool) -> dict:
    from autoppia_iwa.entrypoints.benchmark.utils.task_generation import (
        generate_tasks_for_project,
        get_projects_by_ids,
    )
    from autoppia_iwa.src.demo_webs.config import demo_web_projects

    [project] = get_projects_by_ids(demo_web_projects, [project_id])
    tasks = await generate_tasks_for_project(
        project,
        prompts_per_use_case=prompts_per_use_case,
        use_cases=None,
        dynamic=dynamic,
    )

    return {
        "project_id": project.id,
        "project_name": project.name,
        "timestamp": datetime.now().isoformat(),
        "tasks": [t.serialize() for t in tasks],
    }


def main() -> None:
    operator_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Generate and cache tasks via autoppia_iwa.")
    parser.add_argument("--project-id", default="autocinema", help="Web project id (e.g. autocinema)")
    parser.add_argument("--prompts-per-use-case", type=int, default=1, help="Prompt variants per use case")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic task generation")
    parser.add_argument(
        "--out",
        default=str(operator_dir.parent / "autoppia_rl" / "data" / "task_cache" / "autoppia_cinema_tasks.json"),
        help="Output cache JSON path",
    )
    args = parser.parse_args()

    _load_operator_env(operator_dir)

    k = os.getenv("OPENAI_API_KEY") or ""
    k_fpr = hashlib.sha256(k.encode("utf-8")).hexdigest()[:12] if k else "missing"
    if not k:
        raise SystemExit("OPENAI_API_KEY missing in environment (check autoppia_operator/.env).")
    print(f"OPENAI_API_KEY=set fpr={k_fpr}")

    payload = asyncio.run(_generate(args.project_id, args.prompts_per_use_case, bool(args.dynamic)))
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload['tasks'])} tasks -> {out_path}")

    if not payload["tasks"]:
        raise SystemExit("Generated 0 tasks (check OPENAI key/model access and generation logs).")


if __name__ == "__main__":
    main()
