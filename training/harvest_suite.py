from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.focus_pipeline import DEFAULT_TASK_CACHE, default_task_cache_for_project, focus_root as _focus_root
from training.harvester import HarvestConfig, collect_rows_for_seeds, write_harvest_artifacts
from training.use_case_registry import all_use_case_specs, get_use_case_spec


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _use_cases_from_task_cache(task_cache_path: Path, project_id: str) -> list[str]:
    try:
        payload = json.loads(task_cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    tasks = payload.get("tasks", payload if isinstance(payload, list) else [])
    seen: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        row_project = str(task.get("web_project_id") or task.get("project_id") or "").strip().lower()
        if row_project and row_project != project_id.lower():
            continue
        uc = task.get("use_case", {})
        name = str(uc.get("name", "") if isinstance(uc, dict) else uc).strip().upper()
        if name and name not in seen:
            seen.append(name)
    return seen


def parse_use_case_spec(value: str, *, project_id: str = "autocinema", task_cache_path: Path | None = None) -> list[str]:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "all":
        if project_id.lower() != "autocinema" and task_cache_path is not None:
            names = _use_cases_from_task_cache(task_cache_path, project_id)
            if names:
                return names
        return [spec.name for spec in all_use_case_specs()]
    names: list[str] = []
    for part in raw.split(","):
        name = str(part).strip().upper()
        if not name:
            continue
        if name not in names:
            names.append(name)
    return names


@dataclass(frozen=True)
class HarvestSuiteConfig:
    use_cases: tuple[str, ...]
    seeds: tuple[int, ...]
    provider: str
    model: str
    project_id: str = "autocinema"
    task_cache_arg: str = str(DEFAULT_TASK_CACHE)
    attempt_models: tuple[str, ...] = ()
    strategy: str = "baseline"
    max_steps: int = 12
    task_concurrency: int = 1
    agent_workers: int = 1
    collect_workers: int = 1
    brief_model: str = "gpt-5.4-mini"
    execution_mode: str = "operator"
    max_claude_attempts: int = 3
    brief_dir: Path | None = None
    max_seeds_per_use_case: int = 0
    target_gold_per_use_case: int = 0
    deterministic_only: bool = False


def collect_suite(config: HarvestSuiteConfig) -> dict[str, Any]:
    use_case_summaries: list[dict[str, Any]] = []
    total_attempt_rows = 0
    total_gold_rows = 0
    project_id = str(config.project_id or "autocinema").strip() or "autocinema"

    for use_case in config.use_cases:
        output_root = _focus_root(use_case=use_case, project_id=project_id)
        existing_gold_rows = _read_jsonl(output_root / "gold" / "episodes.jsonl")
        existing_gold_count = len(existing_gold_rows)
        if config.target_gold_per_use_case > 0 and existing_gold_count >= int(config.target_gold_per_use_case):
            use_case_summaries.append(
                {
                    "use_case": use_case,
                    "status": "target_already_met",
                    "existing_gold_count": existing_gold_count,
                    "target_gold_per_use_case": int(config.target_gold_per_use_case),
                    "generated_at": _now_utc(),
                }
            )
            total_gold_rows += existing_gold_count
            continue

        seeds = list(config.seeds)
        if config.max_seeds_per_use_case > 0:
            seeds = seeds[: max(0, int(config.max_seeds_per_use_case))]
        attempt_models = tuple(config.attempt_models) or tuple(get_use_case_spec(use_case).harvester_model_ladder) or (config.model,)
        max_claude_attempts = 0 if config.deterministic_only else config.max_claude_attempts
        effective_strategy = "code-aware" if config.deterministic_only else config.strategy
        # Deterministic replay executes every planned action — don't cut trajectories short.
        effective_max_steps = max(config.max_steps, 30) if config.deterministic_only else config.max_steps
        harvest_config = HarvestConfig(
            use_case=use_case,
            output_root=output_root,
            provider=config.provider,
            model=attempt_models[0],
            task_cache_arg=config.task_cache_arg,
            web_project_id=project_id,
            max_steps=effective_max_steps,
            task_concurrency=config.task_concurrency,
            agent_workers=config.agent_workers,
            attempt_models=attempt_models,
            brief_dir=config.brief_dir,
            brief_model=config.brief_model,
            execution_mode=config.execution_mode,
            max_claude_attempts=max_claude_attempts,
            deterministic_only=config.deterministic_only,
        )
        rows = collect_rows_for_seeds(
            config=harvest_config,
            seeds=seeds,
            strategy=effective_strategy,
            collect_workers=config.collect_workers,
        )
        episodes_path, summary_path, summary = write_harvest_artifacts(
            output_root=output_root,
            use_case=use_case,
            target_seeds=seeds,
            rows=rows,
        )
        attempt_count = len(rows)
        gold_count = sum(1 for row in rows if bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0)
        total_attempt_rows += attempt_count
        total_gold_rows += int(summary.get("gold_episodes_total") or gold_count)
        use_case_summaries.append(
            {
                "use_case": use_case,
                "status": "collected",
                "episodes_path": str(episodes_path),
                "summary_path": str(summary_path),
                "attempt_rows_added": attempt_count,
                "gold_rows_added": gold_count,
                "existing_gold_count": existing_gold_count,
                "canonical_gold_count": int(summary.get("gold_episodes_total") or 0),
                "attempts_total": int(summary.get("attempts_total") or 0),
                "generated_at": _now_utc(),
            }
        )

    suite_summary = {
        "project_id": project_id,
        "use_cases": list(config.use_cases),
        "strategy": "code-aware" if config.deterministic_only else str(config.strategy),
        "deterministic_only": config.deterministic_only,
        "provider": str(config.provider),
        "model": str(config.model),
        "seeds": [int(seed) for seed in config.seeds],
        "use_case_count": len(config.use_cases),
        "attempt_rows_added_total": total_attempt_rows,
        "canonical_gold_total": total_gold_rows,
        "per_use_case": use_case_summaries,
        "generated_at": _now_utc(),
    }
    return suite_summary
