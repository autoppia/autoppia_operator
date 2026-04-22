#!/usr/bin/env python3
"""Focused single-use-case data pipeline for Autocinema.

Typical flow:
1. collect gold trajectories for one use case and a fixed seed set
2. export SFT from that gold set only
3. train a LoRA on that use case
4. evaluate only that use case
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_operator_env(operator_dir: Path) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    env_path = operator_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


_load_operator_env(REPO_ROOT)

from training.claude_code_harvester import generate_claude_brief, save_claude_brief
from training.deterministic_harvester.normalizer import task_seeds_for_use_case
from training.focus_cost_analytics import build_focus_cost_report
from training.focus_dataset_validation import validate_focus_dataset
from training.focus_pipeline import (
    build_focus_eval_command,
    build_prompt_override,
    build_runpod_job_command,
    build_task_cache_override,
    default_task_cache_for_project,
    focus_root,
)
from training.harvester import (
    HarvestConfig,
    collect_rows_for_seeds,
    collect_rows_from_guided_brief,
    consolidate_harvest_gold,
    export_harvest_sft,
    generate_candidates_for_seeds,
    list_candidates,
    replay_candidates,
    write_harvest_artifacts,
)
from training.use_case_registry import get_use_case_spec


def _parse_seed_spec(value: str) -> list[int]:
    value = str(value).strip()
    if not value:
        return []
    if ".." in value:
        left, right = value.split("..", 1)
        start = int(left)
        end = int(right)
        step = 1 if end >= start else -1
        return list(range(start, end + step, step))
    return [int(part) for part in value.split(",") if part.strip()]


def _resolve_seed_list(*, use_case: str, seed_spec: str, task_cache: str, project_id: str | None = None) -> list[int]:
    explicit = _parse_seed_spec(seed_spec)
    if explicit:
        return explicit
    seeds = task_seeds_for_use_case(
        cache_path=Path(task_cache).resolve(),
        use_case=use_case,
        web_project_id=str(project_id or "").strip() or None,
    )
    if seeds:
        return seeds
    project_suffix = f" project_id={project_id}" if str(project_id or "").strip() else ""
    raise ValueError(f"No URL seeds found for use_case={use_case}{project_suffix} in {Path(task_cache).resolve()}")


def _resolve_single_seed(*, use_case: str, seed: int | None, task_cache: str, project_id: str | None = None) -> int:
    if seed is not None:
        return int(seed)
    seeds = _resolve_seed_list(use_case=use_case, seed_spec="", task_cache=task_cache, project_id=project_id)
    if len(seeds) != 1:
        raise ValueError(f"Multiple URL seeds found for use_case={use_case}; pass --seed explicitly")
    return int(seeds[0])


def _resolve_task_cache(*, task_cache_arg: str, project_id: str) -> str:
    value = str(task_cache_arg or "").strip()
    if value:
        return str(Path(value).resolve())
    return str(default_task_cache_for_project(project_id).resolve())


def _parse_model_ladder(value: str, default_model: str) -> list[str]:
    parts = [part.strip() for part in str(value or "").split(",") if part.strip()]
    return parts or [str(default_model).strip()]


def _rows_estimated_cost_usd(rows: list[dict[str, object]]) -> float:
    return float(sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows if isinstance(row, dict)))


def cmd_collect(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    task_cache = _resolve_task_cache(task_cache_arg=args.task_cache, project_id=project_id)
    output_root = focus_root(use_case=use_case, project_id=project_id)
    seeds = _resolve_seed_list(use_case=use_case, seed_spec=args.seeds, task_cache=task_cache, project_id=project_id)
    spec = get_use_case_spec(use_case)
    if str(args.attempt_models).strip():
        attempt_models = _parse_model_ladder(args.attempt_models, str(args.model).strip() or "gpt-5.4")
    elif spec.harvester_model_ladder:
        attempt_models = list(spec.harvester_model_ladder)
    else:
        attempt_models = [str(args.model).strip() or "gpt-5.4"]
    config = HarvestConfig(
        use_case=use_case,
        output_root=output_root,
        provider=args.provider,
        model=attempt_models[0],
        task_cache_arg=task_cache,
        web_project_id=project_id,
        max_steps=args.max_steps,
        task_concurrency=args.task_concurrency,
        agent_workers=args.agent_workers,
        attempt_models=tuple(attempt_models),
        brief_dir=(Path(args.brief_dir).resolve() if str(args.brief_dir).strip() else None),
        headed=bool(getattr(args, "headed", False)),
    )
    max_usd = float(args.max_usd) if args.max_usd is not None else 0.0
    max_attempts = int(args.max_attempts) if args.max_attempts is not None else 0
    collect_workers = max(1, int(args.collect_workers))
    rows: list[dict[str, object]] = []
    if collect_workers == 1:
        for seed in seeds:
            seed_rows = collect_rows_for_seeds(config=config, seeds=[seed], strategy="baseline", collect_workers=1)
            rows.extend(seed_rows)
            if max_attempts > 0 and len(rows) >= max_attempts:
                break
            if max_usd > 0 and _rows_estimated_cost_usd(rows) >= max_usd:
                break
    else:
        rows.extend(collect_rows_for_seeds(config=config, seeds=seeds, strategy="baseline", collect_workers=collect_workers))

    episodes_path, summary_path, summary = write_harvest_artifacts(
        output_root=output_root,
        use_case=use_case,
        target_seeds=seeds,
        rows=rows,
        merge_existing=not args.no_merge_existing,
    )
    print(json.dumps({"episodes_path": str(episodes_path), "summary_path": str(summary_path), "summary": summary}, indent=2))
    return 0


def cmd_export_sft(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    output_root = focus_root(use_case=use_case, project_id=project_id)
    train_seeds = _parse_seed_spec(args.train_seeds) if args.train_seeds else None
    val_seeds = _parse_seed_spec(args.val_seeds) if args.val_seeds else None
    sft_output_dir = output_root / (str(args.output_dir_name).strip() or "sft")
    manifest = export_harvest_sft(
        episodes_path=output_root / "gold" / "episodes.jsonl",
        summary_path=output_root / "gold" / "summary.json",
        output_dir=sft_output_dir,
        split_seed=int(args.split_seed),
        val_ratio=float(args.val_ratio),
        train_seeds=train_seeds,
        val_seeds=val_seeds,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_consolidate_gold(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    output_root = focus_root(use_case=use_case, project_id=project_id)
    episodes_path, summary_path, summary = consolidate_harvest_gold(output_root=output_root, use_case=use_case)
    print(json.dumps({"episodes_path": str(episodes_path), "summary_path": str(summary_path), "summary": summary}, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    output_root = focus_root(use_case=use_case, project_id=project_id)
    cmd = build_runpod_job_command(
        sft_dir=output_root / (str(args.sft_dir_name).strip() or "sft"),
        output_dir=Path(args.output_dir).resolve(),
        existing_pod_id=args.existing_pod_id,
        epochs=int(args.epochs),
        lora_rank=int(args.lora_rank),
    )
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2])
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    task_cache = _resolve_task_cache(task_cache_arg=args.task_cache, project_id=project_id)
    output_root = focus_root(use_case=use_case, project_id=project_id)
    task_cache_path = None
    prompt_override = build_prompt_override(use_case=use_case)
    if prompt_override:
        task_cache_path = output_root / "task_cache" / f"{use_case.lower()}_eval.json"
        build_task_cache_override(
            source_task_cache=Path(task_cache).resolve(),
            use_case=use_case,
            prompt_override=prompt_override,
            out_path=task_cache_path,
            project_id=project_id,
        )
    cmd = build_focus_eval_command(
        project_id=project_id,
        use_case=use_case,
        adapter_path=Path(args.adapter_path).resolve(),
        endpoint=args.endpoint,
        served_model_id=args.served_model_id,
        out_path=output_root / "eval" / "post_finetune_eval.json",
        summary_out_path=output_root / "eval" / "post_finetune_eval_summary.json",
        max_steps=int(args.max_steps),
        num_tasks=int(args.num_tasks),
        task_concurrency=int(args.task_concurrency),
        task_cache=task_cache_path,
    )
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2])
    return 0


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    output_root = focus_root(use_case=use_case, project_id=project_id)
    result = validate_focus_dataset(
        output_root,
        sft_dir_name=str(args.sft_dir_name).strip() or "sft",
        require_trace_files=not bool(args.allow_missing_traces),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def cmd_cost_report(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    output_root = focus_root(use_case=use_case, project_id=project_id)
    result = build_focus_cost_report(output_root)
    print(json.dumps(result, indent=2))
    return 0


def cmd_claude_brief(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    task_cache = _resolve_task_cache(task_cache_arg=args.task_cache, project_id=project_id)
    seed = _resolve_single_seed(use_case=use_case, seed=args.seed, task_cache=task_cache, project_id=project_id)
    output_root = focus_root(use_case=use_case, project_id=project_id)
    payload = generate_claude_brief(
        use_case=use_case,
        seed=seed,
        model=str(args.model).strip() or "gpt-5.4-mini",
        task_cache_path=Path(task_cache).resolve() if str(task_cache).strip() else None,
        web_project_id=project_id,
    )
    out_path = save_claude_brief(use_case=use_case, seed=seed, payload=payload, output_root=output_root)
    print(json.dumps({"brief_path": str(out_path), "payload": payload}, indent=2))
    return 0


def cmd_claude_harvest(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    task_cache = _resolve_task_cache(task_cache_arg=args.task_cache, project_id=project_id)
    output_root = focus_root(use_case=use_case, project_id=project_id)
    seeds = _resolve_seed_list(use_case=use_case, seed_spec=args.seeds, task_cache=task_cache, project_id=project_id)
    collect_workers = max(1, int(args.collect_workers))
    deterministic_only = bool(getattr(args, "deterministic_only", False))
    max_claude_attempts = int(args.max_claude_attempts)
    if deterministic_only and max_claude_attempts > 1:
        max_claude_attempts = 0
    config = HarvestConfig(
        use_case=use_case,
        output_root=output_root,
        provider=args.provider,
        model=args.model,
        task_cache_arg=task_cache,
        web_project_id=project_id,
        max_steps=args.max_steps,
        task_concurrency=args.task_concurrency,
        agent_workers=args.agent_workers,
        brief_model=args.brief_model,
        execution_mode=args.execution_mode,
        max_claude_attempts=max_claude_attempts,
        claude_workers=int(args.claude_workers),
        replay_workers=int(args.replay_workers),
        claude_timeout_seconds=int(args.claude_timeout_seconds),
        deterministic_only=deterministic_only,
        headed=bool(getattr(args, "headed", False)),
    )
    if deterministic_only:
        print(
            json.dumps(
                {
                    "run_mode": "deterministic_only",
                    "project_id": project_id,
                    "use_case": use_case,
                    "task_cache": task_cache,
                    "seed_count": len(seeds),
                    "max_claude_attempts_forced": max_claude_attempts,
                },
                indent=2,
            )
        )
    rows = collect_rows_for_seeds(config=config, seeds=seeds, strategy="code-aware", collect_workers=collect_workers)
    if deterministic_only:
        non_deterministic_rows = [row for row in rows if isinstance(row, dict) and not str(row.get("harvest_mode") or "").strip().startswith("deterministic_")]
        if non_deterministic_rows:
            raise RuntimeError("deterministic-only run produced non-deterministic rows; aborting")

    episodes_path, summary_path, summary = write_harvest_artifacts(
        output_root=output_root,
        use_case=use_case,
        target_seeds=seeds,
        rows=rows,
        merge_existing=not args.no_merge_existing,
    )
    output_payload = {"episodes_path": str(episodes_path), "summary_path": str(summary_path), "summary": summary}
    if deterministic_only:
        output_payload["deterministic_only"] = {
            "enabled": True,
            "ai_assisted_attempts_total": int(summary.get("ai_assisted_attempts_total") or 0),
            "deterministic_attempts_total": int(summary.get("deterministic_attempts_total") or 0),
            "ai_calls_detected": int(summary.get("ai_assisted_attempts_total") or 0) > 0,
        }
    print(json.dumps(output_payload, indent=2))
    return 0


def cmd_generate_candidates(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    task_cache = _resolve_task_cache(task_cache_arg=args.task_cache, project_id=project_id)
    output_root = focus_root(use_case=use_case, project_id=project_id)
    seeds = _resolve_seed_list(use_case=use_case, seed_spec=args.seeds, task_cache=task_cache, project_id=project_id)
    config = HarvestConfig(
        use_case=use_case,
        output_root=output_root,
        provider=args.provider,
        model=args.model,
        task_cache_arg=task_cache,
        web_project_id=project_id,
        max_steps=args.max_steps,
        task_concurrency=args.task_concurrency,
        agent_workers=args.agent_workers,
        brief_model=args.brief_model,
        execution_mode=args.execution_mode,
        max_claude_attempts=args.max_claude_attempts,
        claude_workers=int(args.claude_workers),
        replay_workers=int(args.replay_workers),
        claude_timeout_seconds=int(args.claude_timeout_seconds),
        deterministic_only=bool(getattr(args, "deterministic_only", False)),
        headed=bool(getattr(args, "headed", False)),
    )
    candidate_paths = generate_candidates_for_seeds(config=config, seeds=seeds)
    print(json.dumps({"candidate_paths": [str(path) for path in candidate_paths], "count": len(candidate_paths)}, indent=2))
    return 0


def cmd_replay_candidates(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    task_cache = _resolve_task_cache(task_cache_arg=args.task_cache, project_id=project_id)
    output_root = focus_root(use_case=use_case, project_id=project_id)
    seeds = _parse_seed_spec(args.seeds) if str(args.seeds).strip() else []
    config = HarvestConfig(
        use_case=use_case,
        output_root=output_root,
        provider=args.provider,
        model=args.model,
        task_cache_arg=task_cache,
        web_project_id=project_id,
        max_steps=args.max_steps,
        task_concurrency=args.task_concurrency,
        agent_workers=args.agent_workers,
        brief_model=args.brief_model,
        execution_mode=args.execution_mode,
        max_claude_attempts=args.max_claude_attempts,
        claude_workers=int(args.claude_workers),
        replay_workers=int(args.replay_workers),
        claude_timeout_seconds=int(args.claude_timeout_seconds),
        deterministic_only=bool(getattr(args, "deterministic_only", False)),
        headed=bool(getattr(args, "headed", False)),
    )
    candidate_paths = [Path(value).resolve() for value in (args.candidate_path or []) if str(value).strip()]
    if not candidate_paths:
        candidate_paths = list_candidates(output_root=output_root, seeds=seeds or None)
    rows = replay_candidates(config=config, candidate_paths=candidate_paths)
    target_seeds = seeds or sorted({int(row.get("seed") or 0) for row in rows})
    episodes_path, summary_path, summary = write_harvest_artifacts(
        output_root=output_root,
        use_case=use_case,
        target_seeds=target_seeds,
        rows=rows,
        merge_existing=not args.no_merge_existing,
    )
    print(
        json.dumps(
            {
                "candidate_count": len(candidate_paths),
                "episodes_path": str(episodes_path),
                "summary_path": str(summary_path),
                "summary": summary,
            },
            indent=2,
        )
    )
    return 0


def cmd_run_guided_brief(args: argparse.Namespace) -> int:
    use_case = str(args.use_case).upper()
    project_id = str(args.project_id).strip() or "autocinema"
    task_cache = _resolve_task_cache(task_cache_arg=args.task_cache, project_id=project_id)
    output_root = focus_root(use_case=use_case, project_id=project_id)
    seeds = _resolve_seed_list(use_case=use_case, seed_spec=args.seeds, task_cache=task_cache, project_id=project_id)
    collect_workers = max(1, int(getattr(args, "collect_workers", 1) or 1))
    brief_payload = json.loads(Path(args.brief_path).resolve().read_text(encoding="utf-8"))
    attempt_name = str(args.attempt_name).strip() or "guided"
    rows = collect_rows_from_guided_brief(
        use_case=use_case,
        seeds=seeds,
        brief_payload=brief_payload,
        task_cache=Path(task_cache).resolve(),
        output_root=output_root,
        attempt_name=attempt_name,
        max_steps=int(args.max_steps),
        collect_workers=collect_workers,
        success_texts=list(args.success_text or []),
        teacher_brief_path=str(Path(args.brief_path).resolve()),
    )
    episodes_path, summary_path, summary = write_harvest_artifacts(
        output_root=output_root,
        use_case=use_case,
        target_seeds=seeds,
        rows=rows,
        merge_existing=not args.no_merge_existing,
    )
    print(json.dumps({"episodes_path": str(episodes_path), "summary_path": str(summary_path), "summary": summary}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Focused multi-project use-case pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--project-id", default="autocinema")
    collect.add_argument("--use-case", default="LOGIN")
    collect.add_argument("--seeds", default="", help="Explicit seed spec like 1..10 or 1,2,3. Leave empty to use task URL seeds.")
    collect.add_argument("--provider", default="openai")
    collect.add_argument("--model", default="gpt-5.4")
    collect.add_argument("--attempt-models", default="")
    collect.add_argument("--max-steps", type=int, default=12)
    collect.add_argument("--task-concurrency", type=int, default=1)
    collect.add_argument("--agent-workers", type=int, default=1)
    collect.add_argument("--collect-workers", type=int, default=1)
    collect.add_argument("--task-cache", default="")
    collect.add_argument("--brief-dir", default="")
    collect.add_argument("--max-usd", type=float, default=0.0)
    collect.add_argument("--max-attempts", type=int, default=0)
    collect.add_argument("--headed", action="store_true")
    collect.add_argument("--no-merge-existing", action="store_true")
    collect.set_defaults(func=cmd_collect)

    export_sft = sub.add_parser("export-sft")
    export_sft.add_argument("--project-id", default="autocinema")
    export_sft.add_argument("--use-case", default="LOGIN")
    export_sft.add_argument("--split-seed", type=int, default=36)
    export_sft.add_argument("--val-ratio", type=float, default=0.2)
    export_sft.add_argument("--train-seeds", default="")
    export_sft.add_argument("--val-seeds", default="")
    export_sft.add_argument("--output-dir-name", default="sft")
    export_sft.set_defaults(func=cmd_export_sft)

    consolidate = sub.add_parser("consolidate-gold")
    consolidate.add_argument("--project-id", default="autocinema")
    consolidate.add_argument("--use-case", default="LOGIN")
    consolidate.set_defaults(func=cmd_consolidate_gold)

    train = sub.add_parser("train")
    train.add_argument("--project-id", default="autocinema")
    train.add_argument("--use-case", default="LOGIN")
    train.add_argument("--existing-pod-id", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--lora-rank", type=int, default=32)
    train.add_argument("--sft-dir-name", default="sft")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--project-id", default="autocinema")
    evaluate.add_argument("--task-cache", default="")
    evaluate.add_argument("--use-case", default="LOGIN")
    evaluate.add_argument("--adapter-path", required=True)
    evaluate.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    evaluate.add_argument("--served-model-id", default="autoppia")
    evaluate.add_argument("--max-steps", type=int, default=12)
    evaluate.add_argument("--num-tasks", type=int, default=10)
    evaluate.add_argument("--task-concurrency", type=int, default=1)
    evaluate.set_defaults(func=cmd_eval)

    validate = sub.add_parser("validate-dataset")
    validate.add_argument("--project-id", default="autocinema")
    validate.add_argument("--use-case", default="LOGIN")
    validate.add_argument("--sft-dir-name", default="sft")
    validate.add_argument("--allow-missing-traces", action="store_true")
    validate.set_defaults(func=cmd_validate_dataset)

    cost_report = sub.add_parser("cost-report")
    cost_report.add_argument("--project-id", default="autocinema")
    cost_report.add_argument("--use-case", default="LOGIN")
    cost_report.set_defaults(func=cmd_cost_report)

    claude_brief = sub.add_parser("teacher-brief", aliases=["claude-brief"])
    claude_brief.add_argument("--project-id", default="autocinema")
    claude_brief.add_argument("--task-cache", default="")
    claude_brief.add_argument("--use-case", default="CONTACT")
    claude_brief.add_argument("--seed", type=int, default=None, help="Optional explicit seed override. Defaults to the task URL seed.")
    claude_brief.add_argument("--model", default="gpt-5.4-mini")
    claude_brief.set_defaults(func=cmd_claude_brief)

    claude_harvest = sub.add_parser("teacher-harvest", aliases=["claude-harvest", "gpt-harvest"])
    claude_harvest.add_argument("--project-id", default="autocinema")
    claude_harvest.add_argument("--use-case", default="CONTACT")
    claude_harvest.add_argument("--seeds", default="", help="Explicit seed spec like 1..10 or 1,2,3. Leave empty to use task URL seeds.")
    claude_harvest.add_argument("--provider", default="openai")
    claude_harvest.add_argument("--model", default="gpt-5.4-mini")
    claude_harvest.add_argument("--brief-model", default="gpt-5.4-mini")
    claude_harvest.add_argument("--max-steps", type=int, default=12)
    claude_harvest.add_argument("--task-concurrency", type=int, default=1)
    claude_harvest.add_argument("--agent-workers", type=int, default=1)
    claude_harvest.add_argument("--collect-workers", type=int, default=1)
    claude_harvest.add_argument("--claude-workers", type=int, default=1)
    claude_harvest.add_argument("--replay-workers", type=int, default=1)
    claude_harvest.add_argument("--task-cache", default="")
    claude_harvest.add_argument("--max-claude-attempts", type=int, default=1)
    claude_harvest.add_argument("--claude-timeout-seconds", type=int, default=120)
    claude_harvest.add_argument("--execution-mode", choices=["direct", "operator"], default="direct")
    claude_harvest.add_argument("--deterministic-only", action="store_true")
    claude_harvest.add_argument("--headed", action="store_true")
    claude_harvest.add_argument("--no-merge-existing", action="store_true")
    claude_harvest.set_defaults(func=cmd_claude_harvest)

    generate_candidates = sub.add_parser("generate-candidates")
    generate_candidates.add_argument("--project-id", default="autocinema")
    generate_candidates.add_argument("--use-case", default="CONTACT")
    generate_candidates.add_argument("--seeds", default="", help="Explicit seed spec like 1..10 or 1,2,3. Leave empty to use task URL seeds.")
    generate_candidates.add_argument("--provider", default="openai")
    generate_candidates.add_argument("--model", default="gpt-5.4-mini")
    generate_candidates.add_argument("--brief-model", default="gpt-5.4-mini")
    generate_candidates.add_argument("--max-steps", type=int, default=12)
    generate_candidates.add_argument("--task-concurrency", type=int, default=1)
    generate_candidates.add_argument("--agent-workers", type=int, default=1)
    generate_candidates.add_argument("--claude-workers", type=int, default=1)
    generate_candidates.add_argument("--replay-workers", type=int, default=1)
    generate_candidates.add_argument("--task-cache", default="")
    generate_candidates.add_argument("--max-claude-attempts", type=int, default=3)
    generate_candidates.add_argument("--claude-timeout-seconds", type=int, default=120)
    generate_candidates.add_argument("--execution-mode", choices=["direct", "operator"], default="operator")
    generate_candidates.add_argument("--deterministic-only", action="store_true")
    generate_candidates.add_argument("--headed", action="store_true")
    generate_candidates.set_defaults(func=cmd_generate_candidates)

    replay_candidates_cmd = sub.add_parser("replay-candidates")
    replay_candidates_cmd.add_argument("--project-id", default="autocinema")
    replay_candidates_cmd.add_argument("--use-case", default="CONTACT")
    replay_candidates_cmd.add_argument("--seeds", default="")
    replay_candidates_cmd.add_argument("--candidate-path", action="append", default=[])
    replay_candidates_cmd.add_argument("--provider", default="openai")
    replay_candidates_cmd.add_argument("--model", default="gpt-5.4-mini")
    replay_candidates_cmd.add_argument("--brief-model", default="gpt-5.4-mini")
    replay_candidates_cmd.add_argument("--max-steps", type=int, default=12)
    replay_candidates_cmd.add_argument("--task-concurrency", type=int, default=1)
    replay_candidates_cmd.add_argument("--agent-workers", type=int, default=1)
    replay_candidates_cmd.add_argument("--claude-workers", type=int, default=1)
    replay_candidates_cmd.add_argument("--replay-workers", type=int, default=1)
    replay_candidates_cmd.add_argument("--task-cache", default="")
    replay_candidates_cmd.add_argument("--max-claude-attempts", type=int, default=3)
    replay_candidates_cmd.add_argument("--claude-timeout-seconds", type=int, default=120)
    replay_candidates_cmd.add_argument("--execution-mode", choices=["direct", "operator"], default="operator")
    replay_candidates_cmd.add_argument("--deterministic-only", action="store_true")
    replay_candidates_cmd.add_argument("--headed", action="store_true")
    replay_candidates_cmd.add_argument("--no-merge-existing", action="store_true")
    replay_candidates_cmd.set_defaults(func=cmd_replay_candidates)

    run_guided = sub.add_parser("run-guided-brief")
    run_guided.add_argument("--project-id", default="autocinema")
    run_guided.add_argument("--use-case", default="CONTACT")
    run_guided.add_argument("--seeds", default="", help="Explicit seed spec like 1..10 or 1,2,3. Leave empty to use task URL seeds.")
    run_guided.add_argument("--brief-path", required=True)
    run_guided.add_argument("--success-text", action="append", default=[])
    run_guided.add_argument("--attempt-name", default="guided")
    run_guided.add_argument("--max-steps", type=int, default=12)
    run_guided.add_argument("--task-cache", default="")
    run_guided.add_argument("--collect-workers", type=int, default=1)
    run_guided.add_argument("--no-merge-existing", action="store_true")
    run_guided.set_defaults(func=cmd_run_guided_brief)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
