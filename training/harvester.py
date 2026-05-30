from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.claude_code_harvester import generate_claude_brief
from training.claude_guided_harvester import _guided_actions_from_brief, run_guided_brief
from training.deterministic_harvester import build_deterministic_plan, load_task_objective, load_task_row
from training.deterministic_harvester.normalizer import extract_seed_from_task_url
from training.deterministic_harvester.projects import normalized_origin, resolve_project_id
from training.focus_pipeline import (
    build_prompt_override,
    build_task_cache_override,
    run_eval_attempt,
)
from training.harvester_support import brief_prompt_lines, summarize_attempt_for_claude
from training.trajectory_candidate import TrajectoryCandidate, candidate_path, load_candidate, write_candidate
from training.trajectory_contract import build_provenance_fields
from training.trajectory_replay import candidate_row_from_replay, replay_output_paths, write_replay_report


@dataclass(frozen=True)
class HarvestConfig:
    use_case: str
    output_root: Path
    provider: str
    model: str
    task_cache_arg: str
    web_project_id: str | None = None
    max_steps: int = 12
    task_concurrency: int = 1
    agent_workers: int = 1
    attempt_models: tuple[str, ...] = ()
    brief_dir: Path | None = None
    brief_model: str = "gpt-5.4-mini"
    execution_mode: str = "operator"
    max_claude_attempts: int = 3
    claude_workers: int = 1
    replay_workers: int = 1
    claude_timeout_seconds: int = 120
    deterministic_only: bool = False
    headed: bool = False


def _brief_path(*, output_root: Path, seed: int, attempt_idx: int | None = None) -> Path:
    base = output_root / "harvester" / "claude_runs" / f"seed_{int(seed):04d}"
    base.mkdir(parents=True, exist_ok=True)
    if attempt_idx is None:
        return base / "brief.json"
    return base / f"attempt_{attempt_idx:02d}_brief.json"


def _feedback_path(*, output_root: Path, seed: int, attempt_idx: int) -> Path:
    base = output_root / "harvester" / "claude_runs" / f"seed_{int(seed):04d}"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"attempt_{attempt_idx:02d}_feedback.json"


def _build_task_cache_for_seed(
    *,
    config: HarvestConfig,
    seed: int,
    prompt_override: str,
    suffix: str = "",
) -> Path:
    task_cache_path = config.output_root / "task_cache" / f"{config.use_case.lower()}_seed_{seed:04d}{suffix}.json"
    build_task_cache_override(
        source_task_cache=Path(config.task_cache_arg).resolve(),
        use_case=config.use_case,
        prompt_override=prompt_override,
        out_path=task_cache_path,
        project_id=str(config.web_project_id or "").strip() or None,
        seed=int(seed),
    )
    return task_cache_path


def _apply_row_provenance(
    row: dict[str, Any] | None,
    *,
    task_cache_path: Path | None,
    prompt_override: str,
    policy_mode: str,
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return row
    row.update(
        build_provenance_fields(
            task_cache_path=task_cache_path,
            prompt_override=prompt_override,
            operator_version="step_engine",
            policy_mode=policy_mode,
        )
    )
    return row


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_guided_row(
    *,
    use_case: str,
    seed: int,
    attempt_name: str,
    report: dict[str, Any],
    out_path: Path,
    web_project_id: str = "autocinema",
    teacher_model: str = "",
    teacher_brief_path: str = "",
) -> dict[str, Any] | None:
    episodes = report.get("episodes") if isinstance(report, dict) else None
    episode = episodes[0] if isinstance(episodes, list) and episodes else None
    if not isinstance(episode, dict):
        return None
    effective_seed = int(episode.get("seed") or 0) or extract_seed_from_task_url(str(episode.get("final_url") or "")) or int(seed)
    trace_root = out_path.parent.parent / "traces" / f"seed_{int(effective_seed):04d}_{attempt_name}"
    trace_episodes_dir = trace_root / "episodes"
    trace_episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_task_id = str(episode.get("episode_task_id") or "")
    trace_file = trace_episodes_dir / f"{episode_task_id}.json"
    trace_payload = {
        "episode": episode,
        "report_model": report.get("model"),
        "attempt_name": attempt_name,
        "use_case": use_case,
        "seed": int(effective_seed),
    }
    trace_file.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    row = {
        "web_project_id": str(web_project_id or "autocinema"),
        "task_id": str(episode.get("task_id") or ""),
        "episode_task_id": episode_task_id,
        "use_case": use_case,
        "seed": int(effective_seed),
        "success": bool(episode.get("success")),
        "score": float(episode.get("score") or 0.0),
        "steps": int(episode.get("steps") or 0),
        "model": str(report.get("model") or ""),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "usage_breakdown": {},
        "result_path": str(out_path),
        "trace_dir": str(trace_root),
        "trace_root": str(trace_root),
        "trace_file": str(trace_file),
        "trace_ref": episode_task_id,
        "attempt_name": attempt_name,
        "harvest_mode": "claude_code_direct",
        "teacher_model": str(teacher_model or ""),
        "teacher_brief_path": str(teacher_brief_path or ""),
        "notes": f"{use_case} seed={effective_seed} attempt={attempt_name}",
        "final_url": str(episode.get("final_url") or ""),
    }
    return row


def build_candidate_from_brief(
    *,
    use_case: str,
    seed: int,
    attempt_name: str,
    brief_payload: dict[str, Any],
    brief_path: Path,
    task_url: str,
    web_project_id: str = "autocinema",
) -> TrajectoryCandidate:
    brief = brief_payload.get("brief") if isinstance(brief_payload, dict) else {}
    actions = _guided_actions_from_brief(
        task_url=task_url,
        brief=brief if isinstance(brief, dict) else {},
        web_project_id=web_project_id,
    )
    return TrajectoryCandidate(
        use_case=str(use_case).strip().upper(),
        seed=int(seed),
        attempt_name=str(attempt_name).strip(),
        teacher_model=str((brief_payload.get("meta") or {}).get("model") or ""),
        generation_mode="claude_code_brief",
        brief_path=str(brief_path),
        prompt_lines=tuple(brief_prompt_lines(brief_payload)),
        actions=tuple(dict(action) for action in actions if isinstance(action, dict)),
        metadata={
            "brief_meta": dict(brief_payload.get("meta") or {}) if isinstance(brief_payload.get("meta"), dict) else {},
            "web_files": list(brief_payload.get("web_files") or []) if isinstance(brief_payload.get("web_files"), list) else [],
            "web_project_id": str(web_project_id or "autocinema"),
        },
    )


def _deterministic_plan_path(*, output_root: Path, seed: int, attempt_name: str) -> Path:
    base = output_root / "harvester" / "deterministic_runs" / f"seed_{int(seed):04d}"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{attempt_name}.json"


def build_candidate_from_plan(
    *,
    use_case: str,
    seed: int,
    attempt_name: str,
    prompt_lines: list[str] | tuple[str, ...],
    actions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    metadata: dict[str, Any],
    plan_path: Path,
) -> TrajectoryCandidate:
    return TrajectoryCandidate(
        use_case=str(use_case).strip().upper(),
        seed=int(seed),
        attempt_name=str(attempt_name).strip(),
        teacher_model="",
        generation_mode="deterministic_plan",
        brief_path=str(plan_path),
        prompt_lines=tuple(str(item).strip() for item in prompt_lines if str(item).strip()),
        actions=tuple(dict(action) for action in actions if isinstance(action, dict)),
        metadata=dict(metadata or {}),
    )


def _generate_deterministic_candidate_attempt(
    *,
    config: HarvestConfig,
    seed: int,
    attempt_idx: int,
) -> dict[str, Any]:
    objective = load_task_objective(
        cache_path=Path(config.task_cache_arg).resolve(),
        use_case=config.use_case,
        seed=seed,
        web_project_id=config.web_project_id,
    )
    plan = build_deterministic_plan(objective)
    attempt_name = f"deterministic_{attempt_idx:02d}"
    plan_path = _deterministic_plan_path(output_root=config.output_root, seed=seed, attempt_name=attempt_name)
    plan_payload = {
        "use_case": objective.use_case,
        "web_project_id": objective.web_project_id,
        "seed": int(seed),
        "task_url": objective.task_url,
        "prompt": objective.prompt,
        "route_target": objective.route_target,
        "field_values": dict(objective.field_values),
        "entity_filters": dict(objective.entity_filters),
        "auth_required": bool(objective.auth_required),
        "success_expectations": dict(objective.success_expectations),
        "constraints": [
            {
                "field": hint.field,
                "operator": hint.operator,
                "value": hint.value,
                "source": hint.source,
            }
            for hint in objective.constraints
        ],
        "actions": [dict(action) for action in plan.actions],
        "metadata": dict(plan.metadata),
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prompt_override = build_prompt_override(use_case=config.use_case, extra_lines=list(plan.prompt_lines))
    task_cache_path = _build_task_cache_for_seed(
        config=config,
        seed=seed,
        prompt_override=prompt_override,
        suffix=f"_deterministic_{attempt_idx:02d}",
    )
    candidate = build_candidate_from_plan(
        use_case=config.use_case,
        seed=seed,
        attempt_name=attempt_name,
        prompt_lines=list(plan.prompt_lines),
        actions=list(plan.actions),
        metadata={
            "web_project_id": objective.web_project_id,
            "plan": plan.metadata,
            "objective": {k: v for k, v in plan_payload.items() if k != "actions"},
        },
        plan_path=plan_path,
    )
    stored_candidate_path = candidate_path(output_root=config.output_root, seed=seed, attempt_name=attempt_name)
    write_candidate(stored_candidate_path, candidate)
    return {
        "seed": int(seed),
        "attempt_idx": int(attempt_idx),
        "attempt_name": attempt_name,
        "brief_payload": {"brief": {}, "meta": {"model": ""}},
        "brief_path": plan_path,
        "extra_lines": list(plan.prompt_lines),
        "prompt_override": prompt_override,
        "task_cache_path": task_cache_path,
        "candidate": candidate,
        "stored_candidate_path": stored_candidate_path,
    }


def _generate_candidate_attempt(
    *,
    config: HarvestConfig,
    seed: int,
    attempt_idx: int,
    prior_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    if int(attempt_idx) == 1:
        if bool(config.deterministic_only):
            return _generate_deterministic_candidate_attempt(
                config=config,
                seed=seed,
                attempt_idx=attempt_idx,
            )
        try:
            return _generate_deterministic_candidate_attempt(
                config=config,
                seed=seed,
                attempt_idx=attempt_idx,
            )
        except Exception:
            pass
    if bool(config.deterministic_only):
        raise RuntimeError(f"deterministic-only mode: no teacher fallback for seed={int(seed)}")
    brief_payload = generate_claude_brief(
        use_case=config.use_case,
        seed=seed,
        model=config.brief_model,
        previous_attempts=prior_attempts,
        timeout_seconds=config.claude_timeout_seconds,
        task_cache_path=Path(config.task_cache_arg).resolve(),
        web_project_id=config.web_project_id,
    )
    brief_path = _brief_path(output_root=config.output_root, seed=seed, attempt_idx=attempt_idx)
    brief_path.write_text(json.dumps(brief_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    extra_lines = brief_prompt_lines(brief_payload)
    prompt_override = build_prompt_override(use_case=config.use_case, extra_lines=extra_lines)
    task_cache_path = _build_task_cache_for_seed(
        config=config,
        seed=seed,
        prompt_override=prompt_override,
        suffix=f"_claude_{attempt_idx:02d}",
    )
    attempt_name = f"claude_{attempt_idx:02d}"
    try:
        task_objective = load_task_objective(
            cache_path=Path(config.task_cache_arg).resolve(),
            use_case=config.use_case,
            seed=seed,
            web_project_id=config.web_project_id,
        )
        task_url = task_objective.task_url
        resolved_project_id = task_objective.web_project_id
    except Exception:
        resolved_project_id = resolve_project_id(explicit_project_id=config.web_project_id)
        task_url = f"{normalized_origin(project_id=resolved_project_id)}/?seed={int(seed)}"
    candidate = build_candidate_from_brief(
        use_case=config.use_case,
        seed=seed,
        attempt_name=attempt_name,
        brief_payload=brief_payload,
        brief_path=brief_path,
        task_url=task_url,
        web_project_id=resolved_project_id,
    )
    stored_candidate_path = candidate_path(output_root=config.output_root, seed=seed, attempt_name=attempt_name)
    write_candidate(stored_candidate_path, candidate)
    return {
        "seed": int(seed),
        "attempt_idx": int(attempt_idx),
        "attempt_name": attempt_name,
        "brief_payload": brief_payload,
        "brief_path": brief_path,
        "extra_lines": extra_lines,
        "prompt_override": prompt_override,
        "task_cache_path": task_cache_path,
        "candidate": candidate,
        "stored_candidate_path": stored_candidate_path,
    }


def _execute_candidate_attempt(
    *,
    config: HarvestConfig,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    seed = int(bundle["seed"])
    attempt_name = str(bundle["attempt_name"])
    candidate = bundle["candidate"]
    brief_path = bundle["brief_path"]
    task_cache_path = bundle["task_cache_path"]
    generation_mode = str(getattr(candidate, "generation_mode", "") or "")
    if str(config.execution_mode).strip().lower() == "direct":
        report = run_guided_brief(
            use_case=config.use_case,
            seed=seed,
            brief_payload=bundle["brief_payload"],
            task_cache=task_cache_path,
            web_project_id=str(candidate.metadata.get("web_project_id") or config.web_project_id or "autocinema"),
            max_steps=config.max_steps,
            planned_actions_override=[dict(action) for action in candidate.actions],
            headless=not bool(config.headed),
        )
        runs_dir = config.output_root / "gold" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        out_path = runs_dir / f"seed_{seed:04d}_{attempt_name}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        row = build_guided_row(
            use_case=config.use_case,
            seed=seed,
            attempt_name=attempt_name,
            report=report,
            out_path=out_path,
            web_project_id=str(candidate.metadata.get("web_project_id") or config.web_project_id or "autocinema"),
            teacher_model=config.brief_model,
            teacher_brief_path=str(brief_path),
        )
        report_payload = report
    else:
        report_payload, row, _ = replay_candidate(
            candidate=candidate,
            output_root=config.output_root,
            task_cache=task_cache_path,
            max_steps=config.max_steps,
            headed=bool(config.headed),
        )
    is_gold = bool(row and bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0)
    if row:
        if generation_mode == "deterministic_plan":
            row["harvest_mode"] = "deterministic_direct" if str(config.execution_mode).strip().lower() == "direct" else "deterministic_replay"
            row["teacher_model"] = str(candidate.teacher_model or "")
        else:
            row["harvest_mode"] = "claude_code_direct" if str(config.execution_mode).strip().lower() == "direct" else "claude_code_replay"
            row["teacher_model"] = config.brief_model
        row["teacher_brief_path"] = str(brief_path)
        row["candidate_path"] = str(bundle["stored_candidate_path"])
        row = (
            _apply_row_provenance(
                row,
                task_cache_path=task_cache_path,
                prompt_override=str(bundle["prompt_override"]),
                policy_mode="guided" if str(config.execution_mode).strip().lower() == "direct" else "direct",
            )
            or row
        )
    feedback = summarize_attempt_for_claude(attempt_name=attempt_name, report=report_payload, row=row)
    feedback["brief_path"] = str(brief_path)
    feedback["prompt_lines"] = list(bundle["extra_lines"])
    feedback_path = _feedback_path(output_root=config.output_root, seed=seed, attempt_idx=int(bundle["attempt_idx"]))
    feedback_path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "seed": seed,
        "row": row,
        "report_payload": report_payload,
        "feedback": feedback,
        "is_gold": is_gold,
    }


def _generation_failure_feedback(*, seed: int, attempt_idx: int, error: Exception) -> dict[str, Any]:
    return {
        "attempt_name": f"claude_{attempt_idx:02d}",
        "seed": int(seed),
        "success": False,
        "score": 0.0,
        "steps": 0,
        "failure_hints": ["claude_generation_failed"],
        "error": str(error),
    }


def _candidate_attempt_sort_key(candidate: TrajectoryCandidate) -> tuple[int, str]:
    return int(candidate.seed), str(candidate.attempt_name)


def list_candidates(*, output_root: Path, seeds: list[int] | None = None) -> list[Path]:
    candidates_dir = output_root / "candidates"
    if not candidates_dir.exists():
        return []
    allowed_seeds = {int(seed) for seed in (seeds or [])}
    paths: list[Path] = []
    for path in sorted(candidates_dir.glob("seed_*.json")):
        try:
            candidate = load_candidate(path)
        except Exception:
            continue
        if allowed_seeds and int(candidate.seed) not in allowed_seeds:
            continue
        paths.append(path)
    return paths


def generate_candidates_for_seeds(*, config: HarvestConfig, seeds: list[int]) -> list[Path]:
    pending_seeds = [int(seed) for seed in seeds]
    prior_attempts_by_seed: dict[int, list[dict[str, Any]]] = {int(seed): [] for seed in pending_seeds}
    generated_paths: list[Path] = []
    claude_workers = max(1, int(config.claude_workers or 1))

    max_attempts = 1 if bool(config.deterministic_only) else max(1, int(config.max_claude_attempts))
    for attempt_idx in range(1, max_attempts + 1):
        if not pending_seeds:
            break
        generation_failures: list[dict[str, Any]] = []
        generated_bundles: list[dict[str, Any]] = []
        if claude_workers <= 1 or len(pending_seeds) <= 1:
            for seed in pending_seeds:
                try:
                    generated_bundles.append(
                        _generate_candidate_attempt(
                            config=config,
                            seed=seed,
                            attempt_idx=attempt_idx,
                            prior_attempts=prior_attempts_by_seed.get(seed, []),
                        )
                    )
                except Exception as exc:
                    generation_failures.append({"seed": int(seed), "feedback": _generation_failure_feedback(seed=seed, attempt_idx=attempt_idx, error=exc)})
        else:
            with ThreadPoolExecutor(max_workers=claude_workers) as pool:
                futures = {
                    pool.submit(
                        _generate_candidate_attempt,
                        config=config,
                        seed=seed,
                        attempt_idx=attempt_idx,
                        prior_attempts=list(prior_attempts_by_seed.get(seed, [])),
                    ): seed
                    for seed in pending_seeds
                }
                for future in as_completed(futures):
                    seed = int(futures[future])
                    try:
                        generated_bundles.append(future.result())
                    except Exception as exc:
                        generation_failures.append({"seed": seed, "feedback": _generation_failure_feedback(seed=seed, attempt_idx=attempt_idx, error=exc)})
        next_pending: list[int] = []
        for bundle in generated_bundles:
            generated_paths.append(Path(bundle["stored_candidate_path"]))
            next_pending.append(int(bundle["seed"]))
        for failure in generation_failures:
            seed = int(failure["seed"])
            prior_attempts_by_seed.setdefault(seed, []).append(dict(failure["feedback"]))
            next_pending.append(seed)
        pending_seeds = sorted(set(next_pending))
    return sorted(generated_paths)


def replay_candidates(
    *,
    config: HarvestConfig,
    candidate_paths: list[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replay_workers = max(1, int(config.replay_workers or 1))

    def load_bundle(path: Path) -> dict[str, Any]:
        candidate = load_candidate(path)
        task_cache_path = config.output_root / "task_cache" / f"{config.use_case.lower()}_seed_{int(candidate.seed):04d}_{candidate.attempt_name}.json"
        if not task_cache_path.exists():
            prompt_override = "\n".join(candidate.prompt_lines)
            task_cache_path = _build_task_cache_for_seed(
                config=config,
                seed=int(candidate.seed),
                prompt_override=prompt_override,
                suffix=f"_{candidate.attempt_name}",
            )
        brief_path = Path(candidate.brief_path).resolve() if str(candidate.brief_path).strip() else Path(path)
        return {
            "seed": int(candidate.seed),
            "attempt_idx": 0,
            "attempt_name": candidate.attempt_name,
            "brief_payload": {"brief": {}, "meta": {"model": candidate.teacher_model}},
            "brief_path": brief_path,
            "extra_lines": list(candidate.prompt_lines),
            "prompt_override": "\n".join(candidate.prompt_lines),
            "task_cache_path": task_cache_path,
            "candidate": candidate,
            "stored_candidate_path": path,
        }

    bundles = [load_bundle(Path(path).resolve()) for path in candidate_paths]
    if replay_workers <= 1 or len(bundles) <= 1:
        for bundle in bundles:
            result = _execute_candidate_attempt(config=config, bundle=bundle)
            row = result.get("row")
            if isinstance(row, dict):
                rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=replay_workers) as pool:
        futures = {pool.submit(_execute_candidate_attempt, config=config, bundle=bundle): bundle for bundle in bundles}
        for future in as_completed(futures):
            result = future.result()
            row = result.get("row")
            if isinstance(row, dict):
                rows.append(row)
    return rows


def replay_candidate(
    *,
    candidate: TrajectoryCandidate,
    output_root: Path,
    task_cache: Path,
    max_steps: int = 12,
    headed: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None, Path]:
    result_path, _ = replay_output_paths(output_root=output_root, seed=candidate.seed, attempt_name=candidate.attempt_name)
    report = run_guided_brief(
        use_case=candidate.use_case,
        seed=int(candidate.seed),
        brief_payload={"brief": {}, "meta": {"model": candidate.teacher_model}},
        task_cache=task_cache,
        web_project_id=str(candidate.metadata.get("web_project_id") or "autocinema"),
        max_steps=int(max_steps),
        planned_actions_override=[dict(action) for action in candidate.actions],
        headless=not bool(headed),
    )
    write_replay_report(result_path, report)
    row = candidate_row_from_replay(candidate=candidate, report=report, result_path=result_path)
    if isinstance(row, dict):
        row["candidate_path"] = str(candidate_path(output_root=output_root, seed=candidate.seed, attempt_name=candidate.attempt_name))
    return report, row, result_path


def build_harvest_summary(*, use_case: str, target_seeds: list[int], rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_rows = [row for row in rows if bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0]
    all_attempts = len(rows)
    gold_seeds = sorted({int(row.get("seed") or 0) for row in gold_rows})
    failed_seeds = sorted(set(target_seeds) - set(gold_seeds))
    total_prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    total_completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    total_estimated_cost_usd = round(sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows), 6)
    per_model_attempts: dict[str, int] = {}
    deterministic_attempts_total = 0
    ai_assisted_attempts_total = 0
    for row in rows:
        model = str(row.get("model") or "").strip()
        if not model:
            model = "unknown"
        per_model_attempts[model] = int(per_model_attempts.get(model) or 0) + 1
        harvest_mode = str(row.get("harvest_mode") or "").strip().lower()
        if harvest_mode.startswith("deterministic_"):
            deterministic_attempts_total += 1
        elif harvest_mode.startswith("claude_") or "claude" in harvest_mode:
            ai_assisted_attempts_total += 1
    return {
        "use_case": use_case,
        "seeds_targeted": [int(seed) for seed in target_seeds],
        "target_seed_count": len(target_seeds),
        "attempts_total": all_attempts,
        "gold_episodes_total": len(gold_rows),
        "gold_seeds": gold_seeds,
        "failed_seeds": failed_seeds,
        "prompt_tokens_total": total_prompt_tokens,
        "completion_tokens_total": total_completion_tokens,
        "tokens_total": total_tokens,
        "estimated_cost_usd_total": total_estimated_cost_usd,
        "avg_tokens_per_attempt": round((total_tokens / all_attempts) if all_attempts else 0.0, 6),
        "avg_cost_usd_per_attempt": round((total_estimated_cost_usd / all_attempts) if all_attempts else 0.0, 8),
        "attempts_by_model": per_model_attempts,
        "deterministic_attempts_total": deterministic_attempts_total,
        "ai_assisted_attempts_total": ai_assisted_attempts_total,
        "zero_ai_verified": ai_assisted_attempts_total == 0,
        "passed_target": len(gold_seeds) >= len(target_seeds),
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def write_harvest_artifacts(
    *,
    output_root: Path,
    use_case: str,
    target_seeds: list[int],
    rows: list[dict[str, Any]],
    merge_existing: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    gold_root = output_root / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)
    attempts_path = gold_root / "attempts.jsonl"
    episodes_path = gold_root / "episodes.jsonl"
    summary_path = gold_root / "summary.json"
    merged_rows = list(rows)
    if merge_existing:
        existing_attempts = _read_jsonl(attempts_path)
        seen_attempt_keys = {(int(row.get("seed") or 0), str(row.get("attempt_name") or "")) for row in rows if isinstance(row, dict)}
        for row in existing_attempts:
            key = (int(row.get("seed") or 0), str(row.get("attempt_name") or ""))
            if key not in seen_attempt_keys:
                merged_rows.append(row)
    merged_rows.sort(key=lambda row: (int(row.get("seed") or 0), str(row.get("attempt_name") or "")))
    gold_rows = [row for row in merged_rows if bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0]
    _write_jsonl(attempts_path, merged_rows)
    _write_jsonl(episodes_path, gold_rows)
    summary = build_harvest_summary(use_case=use_case, target_seeds=target_seeds, rows=merged_rows)
    _write_json(summary_path, summary)
    return episodes_path, summary_path, summary


def export_harvest_sft(
    *,
    episodes_path: Path,
    summary_path: Path,
    output_dir: Path,
    split_seed: int = 36,
    val_ratio: float = 0.2,
    train_seeds: list[int] | None = None,
    val_seeds: list[int] | None = None,
    trace_only: bool = True,
    runtime_aligned: bool = False,
) -> dict[str, Any]:
    from training.format_for_sft import RUNTIME_ALIGNED_SYSTEM_PROMPT, SYSTEM_PROMPT, export_harvest_to_sft

    return export_harvest_to_sft(
        input_path=str(episodes_path),
        summary_path=str(summary_path),
        output_dir=str(output_dir),
        seed=int(split_seed),
        val_ratio=float(val_ratio),
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        trace_only=bool(trace_only),
        runtime_aligned=bool(runtime_aligned),
        system_prompt=RUNTIME_ALIGNED_SYSTEM_PROMPT if bool(runtime_aligned) else SYSTEM_PROMPT,
    )


def _attempt_name_from_run_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("seed_"):
        return ""
    parts = stem.split("_", 2)
    return parts[2] if len(parts) >= 3 else ""


def _seed_from_run_path(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("seed_"):
        return 0
    parts = stem.split("_", 2)
    try:
        return int(parts[1]) if len(parts) >= 2 else 0
    except Exception:
        return 0


def _rows_from_run_reports(*, gold_root: Path, use_case: str) -> list[dict[str, Any]]:
    runs_dir = gold_root / "runs"
    if not runs_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for run_path in sorted(runs_dir.glob("seed_*.json")):
        try:
            report = _read_json(run_path)
        except Exception:
            continue
        seed = _seed_from_run_path(run_path)
        attempt_name = _attempt_name_from_run_path(run_path)
        if not attempt_name:
            continue
        row = build_guided_row(
            use_case=use_case,
            seed=seed,
            attempt_name=attempt_name,
            report=report,
            out_path=run_path,
            web_project_id="autocinema",
        )
        if isinstance(row, dict):
            rows.append(row)
    return rows


def consolidate_harvest_gold(*, output_root: Path, use_case: str) -> tuple[Path, Path, dict[str, Any]]:
    gold_root = output_root / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)
    canonical_attempts_path = gold_root / "attempts.jsonl"
    canonical_episodes_path = gold_root / "episodes.jsonl"
    canonical_summary_path = gold_root / "summary.json"
    legacy_attempts_path = output_root / "attempts.jsonl"
    legacy_episodes_path = output_root / "episodes.jsonl"
    legacy_summary_path = output_root / "summary.json"

    canonical_attempts = _read_jsonl(canonical_attempts_path)
    legacy_attempts = _read_jsonl(legacy_attempts_path)
    merged_attempts: list[dict[str, Any]] = []
    seen_attempt_keys: set[tuple[int, str]] = set()
    for row in canonical_attempts + legacy_attempts:
        if not isinstance(row, dict):
            continue
        key = (int(row.get("seed") or 0), str(row.get("attempt_name") or ""))
        if key in seen_attempt_keys:
            continue
        seen_attempt_keys.add(key)
        merged_attempts.append(row)
    if not merged_attempts:
        for row in _rows_from_run_reports(gold_root=gold_root, use_case=use_case):
            key = (int(row.get("seed") or 0), str(row.get("attempt_name") or ""))
            if key in seen_attempt_keys:
                continue
            seen_attempt_keys.add(key)
            merged_attempts.append(row)
    merged_attempts.sort(key=lambda row: (int(row.get("seed") or 0), str(row.get("attempt_name") or "")))

    canonical_episodes = _read_jsonl(canonical_episodes_path)
    legacy_episodes = _read_jsonl(legacy_episodes_path)
    merged_episodes: list[dict[str, Any]] = []
    seen_episode_keys: set[tuple[int, str]] = set()
    for row in canonical_episodes + legacy_episodes:
        if not isinstance(row, dict):
            continue
        key = (int(row.get("seed") or 0), str(row.get("episode_task_id") or ""))
        if key in seen_episode_keys:
            continue
        seen_episode_keys.add(key)
        merged_episodes.append(row)
    if not merged_episodes:
        for row in merged_attempts:
            if not (bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0):
                continue
            key = (int(row.get("seed") or 0), str(row.get("episode_task_id") or ""))
            if key in seen_episode_keys:
                continue
            seen_episode_keys.add(key)
            merged_episodes.append(row)
    merged_episodes.sort(key=lambda row: (int(row.get("seed") or 0), str(row.get("attempt_name") or "")))

    target_seeds: list[int] = []
    for summary_path in (legacy_summary_path, canonical_summary_path):
        if not summary_path.exists():
            continue
        payload = _read_json(summary_path)
        target_seeds.extend(int(seed) for seed in payload.get("seeds_targeted") or [] if str(seed).strip())
    target_seeds = sorted(set(target_seeds)) if target_seeds else sorted({int(row.get("seed") or 0) for row in merged_attempts or merged_episodes})

    summary = build_harvest_summary(use_case=use_case, target_seeds=target_seeds, rows=merged_attempts)
    summary.update(
        {
            "episodes_total": len(merged_episodes),
            "successes_total": len(merged_episodes),
            "consolidated_from": [str(legacy_episodes_path), str(canonical_episodes_path), str(gold_root / "runs")],
        }
    )
    _write_jsonl(canonical_attempts_path, merged_attempts)
    _write_jsonl(canonical_episodes_path, merged_episodes)
    _write_json(canonical_summary_path, summary)
    return canonical_episodes_path, canonical_summary_path, summary


def collect_seed_rows(*, config: HarvestConfig, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_lines: list[str] = []
    if config.brief_dir is not None:
        brief_path = Path(config.brief_dir).resolve() / f"seed_{int(seed):04d}.json"
        if brief_path.exists():
            try:
                payload = json.loads(brief_path.read_text(encoding="utf-8"))
                extra_lines.extend(brief_prompt_lines(payload if isinstance(payload, dict) else None))
            except Exception:
                pass
    prompt_override = build_prompt_override(use_case=config.use_case, extra_lines=extra_lines)
    task_cache_path = Path(config.task_cache_arg).resolve()
    if prompt_override:
        task_cache_path = _build_task_cache_for_seed(config=config, seed=seed, prompt_override=prompt_override)

    ladder = list(config.attempt_models or (config.model,))
    attempt_names = ["baseline", "prompt_correction", "prompt_correction_2", "prompt_correction_3"]
    baseline = run_eval_attempt(
        use_case=config.use_case,
        seed=seed,
        attempt_name=attempt_names[0],
        output_root=config.output_root,
        provider=ladder[0] and config.provider,
        model=ladder[0],
        max_steps=config.max_steps,
        task_concurrency=config.task_concurrency,
        agent_workers=config.agent_workers,
        task_cache=task_cache_path,
        web_project_id=str(config.web_project_id or "autocinema"),
        env_overrides={},
        headed=bool(config.headed),
    )
    if baseline.row:
        rows.append(_apply_row_provenance(baseline.row, task_cache_path=task_cache_path, prompt_override=prompt_override, policy_mode="direct") or baseline.row)
    if baseline.is_gold or not prompt_override:
        return rows

    for idx, attempt_model in enumerate(ladder[1:], start=1):
        corrected = run_eval_attempt(
            use_case=config.use_case,
            seed=seed,
            attempt_name=attempt_names[min(idx, len(attempt_names) - 1)],
            output_root=config.output_root,
            provider=config.provider,
            model=attempt_model,
            max_steps=config.max_steps,
            task_concurrency=config.task_concurrency,
            agent_workers=config.agent_workers,
            task_cache=task_cache_path,
            web_project_id=str(config.web_project_id or "autocinema"),
            env_overrides={},
            headed=bool(config.headed),
        )
        if corrected.row:
            rows.append(_apply_row_provenance(corrected.row, task_cache_path=task_cache_path, prompt_override=prompt_override, policy_mode="direct") or corrected.row)
        if corrected.is_gold:
            break
    return rows


def collect_seed_rows_code_aware(*, config: HarvestConfig, seed: int) -> list[dict[str, Any]]:
    return _collect_rows_for_seeds_code_aware(config=config, seeds=[seed])


def _collect_seed_rows_deterministic_first(*, config: HarvestConfig, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prior_attempts: list[dict[str, Any]] = []

    deterministic_bundle = _generate_candidate_attempt(
        config=config,
        seed=seed,
        attempt_idx=1,
        prior_attempts=prior_attempts,
    )
    deterministic_result = _execute_candidate_attempt(config=config, bundle=deterministic_bundle)
    deterministic_row = deterministic_result.get("row")
    if isinstance(deterministic_row, dict):
        rows.append(deterministic_row)
    deterministic_feedback = deterministic_result.get("feedback")
    if isinstance(deterministic_feedback, dict):
        prior_attempts.append(dict(deterministic_feedback))
    if deterministic_result.get("is_gold"):
        return rows
    if bool(config.deterministic_only):
        return rows

    for attempt_idx in range(1, max(1, int(config.max_claude_attempts)) + 1):
        bundle = _generate_candidate_attempt(
            config=config,
            seed=seed,
            attempt_idx=attempt_idx + 1,
            prior_attempts=list(prior_attempts),
        )
        result = _execute_candidate_attempt(config=config, bundle=bundle)
        row = result.get("row")
        if isinstance(row, dict):
            rows.append(row)
        feedback = result.get("feedback")
        if isinstance(feedback, dict):
            prior_attempts.append(dict(feedback))
        if result.get("is_gold"):
            break
    return rows


def _collect_rows_for_seeds_code_aware(*, config: HarvestConfig, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_workers = max(1, int(config.replay_workers or config.claude_workers or 1))
    if max_workers <= 1 or len(seeds) <= 1:
        for seed in seeds:
            rows.extend(_collect_seed_rows_deterministic_first(config=config, seed=int(seed)))
        return rows
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_collect_seed_rows_deterministic_first, config=config, seed=int(seed)): int(seed) for seed in seeds}
        for future in as_completed(futures):
            rows.extend(future.result())
    return rows


def collect_rows_from_guided_brief(
    *,
    use_case: str,
    seeds: list[int],
    brief_payload: dict[str, Any],
    task_cache: Path,
    output_root: Path,
    attempt_name: str = "guided",
    max_steps: int = 12,
    collect_workers: int = 1,
    success_texts: list[str] | None = None,
    teacher_brief_path: str = "",
    web_project_id: str | None = None,
) -> list[dict[str, Any]]:
    guided_project_id = resolve_project_id(explicit_project_id=web_project_id)
    try:
        if Path(task_cache).exists():
            guided_project_id = resolve_project_id(
                load_task_row(cache_path=task_cache, use_case=use_case, web_project_id=guided_project_id),
                explicit_project_id=guided_project_id,
            )
    except Exception:
        guided_project_id = resolve_project_id(explicit_project_id=web_project_id)
    if success_texts:
        brief = brief_payload.get("brief") if isinstance(brief_payload, dict) else None
        if isinstance(brief, dict):
            signals = brief.setdefault("success_signals", {"texts": [], "ids": [], "url_contains": []})
            if isinstance(signals, dict):
                texts = signals.setdefault("texts", [])
                if isinstance(texts, list):
                    for value in success_texts:
                        text = str(value).strip()
                        if text and text not in texts:
                            texts.append(text)

    runs_dir = output_root / "gold" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    teacher_model = str((brief_payload.get("meta") or {}).get("model") or "")

    def run_one(seed: int) -> dict[str, Any] | None:
        report = run_guided_brief(
            use_case=use_case,
            seed=seed,
            brief_payload=brief_payload,
            task_cache=task_cache,
            web_project_id=guided_project_id,
            max_steps=int(max_steps),
        )
        out_path = runs_dir / f"seed_{seed:04d}_{attempt_name}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return build_guided_row(
            use_case=use_case,
            seed=seed,
            attempt_name=attempt_name,
            report=report,
            out_path=out_path,
            web_project_id=guided_project_id,
            teacher_model=teacher_model,
            teacher_brief_path=teacher_brief_path,
        )

    rows: list[dict[str, Any]] = []
    max_workers = max(1, int(collect_workers))
    if max_workers <= 1 or len(seeds) <= 1:
        for seed in seeds:
            row = run_one(seed)
            if isinstance(row, dict):
                rows.append(row)
        return rows
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, seed): seed for seed in seeds}
        for future in as_completed(futures):
            row = future.result()
            if isinstance(row, dict):
                rows.append(row)
    return rows


def collect_rows_for_seeds(
    *,
    config: HarvestConfig,
    seeds: list[int],
    strategy: str = "baseline",
    collect_workers: int = 1,
) -> list[dict[str, Any]]:
    normalized_strategy = str(strategy).strip().lower()
    if normalized_strategy in {"code-aware", "claude", "guided"}:
        return _collect_rows_for_seeds_code_aware(config=config, seeds=seeds)
    worker = collect_seed_rows
    rows: list[dict[str, Any]] = []
    max_workers = max(1, int(collect_workers))
    if max_workers <= 1 or len(seeds) <= 1:
        for seed in seeds:
            rows.extend(worker(config=config, seed=seed))
        return rows

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, config=config, seed=seed): seed for seed in seeds}
        for future in as_completed(futures):
            rows.extend(future.result())
    return rows
