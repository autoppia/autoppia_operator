"""Focused single-use-case trajectory pipeline for Autocinema.

This module is intentionally strict:
- every gold trajectory must come from a real evaluator run
- every gold trajectory must have success=true and score=1.0
- failures are kept separately for diagnosis / corrections

It supports a pragmatic DAgger loop by re-running the same task seed with a
prompt-corrected task cache when the baseline attempt fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.deterministic_harvester.normalizer import extract_seed_from_task_url
from training.layout import use_case_layout
from training.trace_compaction import compact_trace_dir
from training.trajectory_contract import build_provenance_fields
from training.use_case_registry import get_use_case_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK_CACHE = REPO_ROOT.parent / "autoppia_rl" / "data" / "tasks" / "cache" / "autoppia_cinema_tasks.json"


@dataclass
class AttemptResult:
    seed: int
    attempt_name: str
    out_path: Path
    trace_dir: Path
    report: dict[str, Any]
    row: dict[str, Any] | None

    @property
    def is_gold(self) -> bool:
        row = self.row or {}
        return bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _attempt_name_from_run_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("seed_"):
        return ""
    parts = stem.split("_", 2)
    if len(parts) < 3:
        return ""
    return parts[2]


def _seed_from_run_path(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("seed_"):
        return 0
    parts = stem.split("_", 2)
    if len(parts) < 2:
        return 0
    try:
        return int(parts[1])
    except Exception:
        return 0


def _rows_from_run_reports(*, gold_root: Path, use_case: str) -> list[dict[str, Any]]:
    runs_dir = gold_root / "runs"
    traces_root = gold_root / "traces"
    if not runs_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for run_path in sorted(runs_dir.glob("seed_*.json")):
        try:
            report = _load_json(run_path)
        except Exception:
            continue
        seed = _seed_from_run_path(run_path)
        attempt_name = _attempt_name_from_run_path(run_path)
        trace_dir = traces_root / f"seed_{seed:04d}_{attempt_name}"
        row = _episode_row_from_report(
            report=report,
            use_case=use_case,
            seed=seed,
            attempt_name=attempt_name,
            out_path=run_path,
            trace_dir=trace_dir,
        )
        if isinstance(row, dict):
            rows.append(row)
    return rows


def default_task_cache_for_project(project_id: str) -> Path:
    normalized_project = str(project_id or "").strip() or "autocinema"
    project_cache = REPO_ROOT / "data" / "task_cache" / f"{normalized_project}_tasks.json"
    if project_cache.exists():
        return project_cache
    project_cache_legacy = REPO_ROOT / "data" / "task_cache" / f"{normalized_project}_tasks_cache.json"
    if project_cache_legacy.exists():
        return project_cache_legacy
    iwa_cache = REPO_ROOT.parent / "autoppia_iwa" / "data" / "task_cache" / f"{normalized_project}_tasks.json"
    if iwa_cache.exists():
        return iwa_cache
    generic_cache = REPO_ROOT / "data" / "task_cache" / "tasks_cache.json"
    if generic_cache.exists():
        try:
            payload = _load_json(generic_cache)
            tasks = payload.get("tasks", payload if isinstance(payload, list) else [])
            for row in tasks:
                if not isinstance(row, dict):
                    continue
                row_project_id = str(row.get("web_project_id") or row.get("project_id") or "").strip()
                if row_project_id == normalized_project:
                    return generic_cache
        except Exception:
            pass
    if DEFAULT_TASK_CACHE.exists():
        return DEFAULT_TASK_CACHE
    return DEFAULT_TASK_CACHE


def focus_root(*, use_case: str, project_id: str = "autocinema") -> Path:
    return use_case_layout(repo_root=REPO_ROOT, web_project=str(project_id or "autocinema"), use_case=use_case).root


def build_prompt_override(*, use_case: str, extra_lines: list[str] | None = None) -> str:
    lines = list(get_use_case_spec(use_case).harvester_hints)
    if extra_lines:
        lines.extend(str(line).strip() for line in extra_lines if str(line).strip())
    return " ".join(line.strip() for line in lines if line.strip()).strip()


def _reseed_task_row(row: dict[str, Any], seed: int) -> None:
    """Update the task row's id and url to reflect the target seed in-place."""
    import re

    target = str(int(seed))
    old_id = str(row.get("id") or "")
    if old_id:
        row["id"] = re.sub(r"seed-\d+", f"seed-{target}", old_id)
    old_url = str(row.get("url") or "")
    if old_url:
        row["url"] = re.sub(r"([?&]seed=)\d+", rf"\g<1>{target}", old_url)


def build_task_cache_override(
    *,
    source_task_cache: Path,
    use_case: str,
    prompt_override: str,
    out_path: Path,
    project_id: str | None = None,
    seed: int | None = None,
) -> Path:
    payload = _load_json(source_task_cache)
    tasks: list[dict[str, Any]] | None = None
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        tasks = payload["tasks"]
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, dict) and isinstance(value.get("tasks"), list):
                tasks = value["tasks"]
                break
    if not isinstance(tasks, list):
        raise ValueError(f"Unexpected task cache format: {source_task_cache}")
    updated = False
    for row in tasks:
        if not isinstance(row, dict):
            continue
        if str(project_id or "").strip():
            row_project_id = str(row.get("web_project_id") or row.get("project_id") or "").strip()
            if row_project_id and row_project_id != str(project_id).strip():
                continue
        use_case_payload = row.get("use_case")
        if not isinstance(use_case_payload, dict):
            continue
        if str(use_case_payload.get("name") or "").upper() != use_case.upper():
            continue
        if seed is not None:
            _reseed_task_row(row, seed)
        base_prompt = str(row.get("prompt") or "").strip()
        row["prompt"] = f"{base_prompt} {prompt_override}".strip()
        if use_case.upper() == "LOGIN":
            row["prompt"] = (f"First, authenticate with username '<username>' and password '<password>' to log in successfully. {prompt_override}").strip()
            relevant_data = row.get("relevant_data")
            if isinstance(relevant_data, dict):
                user_for_login = relevant_data.get("user_for_login")
                if isinstance(user_for_login, dict):
                    user_for_login["username"] = "<username>"
                    user_for_login["password"] = "<password>"
            tests = row.get("tests")
            if isinstance(tests, list):
                for test in tests:
                    if not isinstance(test, dict):
                        continue
                    criteria = test.get("event_criteria")
                    if isinstance(criteria, dict):
                        criteria.pop("password", None)
            constraints = use_case_payload.get("constraints")
            if isinstance(constraints, list):
                for constraint in constraints:
                    if not isinstance(constraint, dict):
                        continue
                    field = str(constraint.get("field") or "").strip().lower()
                    if field == "username":
                        constraint["value"] = "<username>"
                    if field == "password":
                        constraint["value"] = "<password>"
            additional_prompt_info = str(use_case_payload.get("additional_prompt_info") or "")
            if additional_prompt_info:
                use_case_payload["additional_prompt_info"] = additional_prompt_info.replace("password123", "<password>")
            row["prompt"] = row["prompt"].replace("password123", "<password>")
        elif use_case.upper() == "REGISTRATION":
            row["prompt"] = (f"First, register with username '<signup_username>', email '<signup_email>', and password '<signup_password>'. {prompt_override}").strip()
            tests = row.get("tests")
            if isinstance(tests, list):
                for test in tests:
                    if not isinstance(test, dict):
                        continue
                    criteria = test.get("event_criteria")
                    if not isinstance(criteria, dict):
                        continue
                    criteria.pop("password", None)
        elif use_case.upper() == "LOGOUT":
            tests = row.get("tests")
            if isinstance(tests, list):
                for test in tests:
                    if not isinstance(test, dict):
                        continue
                    criteria = test.get("event_criteria")
                    if not isinstance(criteria, dict):
                        continue
                    criteria.pop("password", None)
        updated = True
    if not updated:
        project_suffix = f" project_id={project_id}" if str(project_id or "").strip() else ""
        raise ValueError(f"No task found for use_case={use_case}{project_suffix} in {source_task_cache}")
    _write_json(out_path, payload if isinstance(payload, dict) else {"tasks": tasks})
    return out_path


def run_eval_attempt(
    *,
    use_case: str,
    seed: int,
    attempt_name: str,
    output_root: Path,
    provider: str,
    model: str,
    max_steps: int,
    task_concurrency: int = 1,
    agent_workers: int = 1,
    task_cache: Path | None = None,
    web_project_id: str = "autocinema",
    env_overrides: dict[str, str] | None = None,
    headed: bool = False,
) -> AttemptResult:
    gold_root = output_root / "gold"
    runs_dir = gold_root / "runs"
    traces_dir = gold_root / "traces" / f"seed_{seed:04d}_{attempt_name}"
    out_path = runs_dir / f"seed_{seed:04d}_{attempt_name}.json"
    cmd = [
        sys.executable,
        "eval.py",
        "--provider",
        provider,
        "--model",
        model,
        "--web-project-id",
        str(web_project_id or "autocinema"),
        "--use-case",
        use_case,
        "--num-tasks",
        "1",
        "--seed",
        str(int(seed)),
        "--max-steps",
        str(int(max_steps)),
        "--task-concurrency",
        str(max(1, int(task_concurrency))),
        "--agent-workers",
        str(max(1, int(agent_workers))),
        "--out",
        str(out_path),
        "--save-act-traces",
        "--trace-dir",
        str(traces_dir),
        "--include-reasoning",
        "--use-local-html-context",
        "--use-site-knowledge",
        "--no-failure-judge",
    ]
    if task_cache is not None:
        cmd.extend(["--task-cache", str(task_cache)])
    env = {
        "FSM_DIRECT_LOOP": "1",
        "EVAL_CAPTURE_SCREENSHOT": "0",
        "EVALUATOR_HEADLESS": "0" if bool(headed) else "1",
        **{k: v for k, v in dict(env_overrides or {}).items() if v is not None},
    }
    subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, **env}, check=True)
    compact_trace_dir(traces_dir)
    report = _load_json(out_path)
    row = _episode_row_from_report(
        report=report,
        use_case=use_case,
        seed=seed,
        attempt_name=attempt_name,
        out_path=out_path,
        trace_dir=traces_dir,
        web_project_id=web_project_id,
    )
    return AttemptResult(seed=seed, attempt_name=attempt_name, out_path=out_path, trace_dir=traces_dir, report=report, row=row)


def _episode_row_from_report(
    *,
    report: dict[str, Any],
    use_case: str,
    seed: int,
    attempt_name: str,
    out_path: Path,
    trace_dir: Path,
    web_project_id: str = "autocinema",
) -> dict[str, Any] | None:
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return None
    episode = episodes[0] if isinstance(episodes[0], dict) else None
    if not isinstance(episode, dict):
        return None
    effective_seed = int(episode.get("seed") or 0) or extract_seed_from_task_url(str(episode.get("final_url") or "")) or int(seed)
    episode_task_id = str(episode.get("episode_task_id") or "")
    trace_file = trace_dir / "episodes" / f"{episode_task_id}.json"
    row = {
        "web_project_id": str(web_project_id or "autocinema"),
        "task_id": str(episode.get("task_id") or ""),
        "episode_task_id": episode_task_id,
        "use_case": use_case,
        "seed": int(effective_seed),
        "success": bool(episode.get("success")),
        "score": float(episode.get("score") or 0.0),
        "steps": int(episode.get("steps") or 0),
        "model": str(report.get("model") or episode.get("model") or ""),
        "prompt_tokens": int(episode.get("prompt_tokens") or 0),
        "completion_tokens": int(episode.get("completion_tokens") or 0),
        "total_tokens": int(episode.get("total_tokens") or 0),
        "estimated_cost_usd": float(episode.get("estimated_cost_usd") or report.get("estimated_cost_usd") or 0.0),
        "usage_breakdown": episode.get("usage_breakdown") if isinstance(episode.get("usage_breakdown"), dict) else {},
        "result_path": str(out_path),
        "trace_dir": str(trace_dir),
        "trace_root": str(trace_dir),
        "trace_file": str(trace_file),
        "trace_ref": episode_task_id,
        "attempt_name": attempt_name,
        "harvest_mode": "focus_use_case",
        "notes": f"{use_case} seed={effective_seed} attempt={attempt_name}",
        "final_url": str(episode.get("final_url") or ""),
    }
    row.update(build_provenance_fields(operator_version="step_engine", policy_mode="direct"))
    return row


def build_focus_summary(*, use_case: str, target_seeds: list[int], rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_rows = [row for row in rows if bool(row.get("success")) and float(row.get("score") or 0.0) >= 1.0]
    all_attempts = len(rows)
    gold_seeds = sorted({int(row.get("seed") or 0) for row in gold_rows})
    failed_seeds = sorted(set(target_seeds) - set(gold_seeds))
    total_prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    total_completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    total_estimated_cost_usd = round(sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows), 6)
    per_model_attempts: dict[str, int] = {}
    for row in rows:
        model = str(row.get("model") or "").strip()
        if not model:
            continue
        per_model_attempts[model] = int(per_model_attempts.get(model) or 0) + 1
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
        "passed_target": len(gold_seeds) >= len(target_seeds),
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def write_focus_artifacts(
    *,
    output_root: Path,
    use_case: str,
    target_seeds: list[int],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path, dict[str, Any]]:
    from training.harvester import write_harvest_artifacts

    return write_harvest_artifacts(
        output_root=output_root,
        use_case=use_case,
        target_seeds=target_seeds,
        rows=rows,
    )


def export_focus_sft(
    *,
    episodes_path: Path,
    summary_path: Path,
    output_dir: Path,
    split_seed: int = 36,
    val_ratio: float = 0.2,
    train_seeds: list[int] | None = None,
    val_seeds: list[int] | None = None,
) -> dict[str, Any]:
    from training.harvester import export_harvest_sft

    return export_harvest_sft(
        episodes_path=episodes_path,
        summary_path=summary_path,
        output_dir=output_dir,
        split_seed=split_seed,
        val_ratio=val_ratio,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
    )


def consolidate_focus_gold(*, output_root: Path, use_case: str) -> tuple[Path, Path, dict[str, Any]]:
    from training.harvester import consolidate_harvest_gold

    return consolidate_harvest_gold(output_root=output_root, use_case=use_case)


def build_runpod_job_command(
    *,
    sft_dir: Path,
    output_dir: Path,
    existing_pod_id: str,
    epochs: int = 1,
    lora_rank: int = 32,
    include_val_data: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "training.runpod_job",
        "--data",
        str(sft_dir / "train.jsonl"),
        "--output-dir",
        str(output_dir),
        "--existing-pod-id",
        str(existing_pod_id),
        "--epochs",
        str(int(epochs)),
        "--lora-rank",
        str(int(lora_rank)),
        "--keep-pod",
    ]
    if include_val_data:
        cmd[5:5] = ["--val-data", str(sft_dir / "val.jsonl")]
    return cmd


def build_focus_eval_command(
    *,
    project_id: str,
    use_case: str,
    adapter_path: Path,
    endpoint: str,
    served_model_id: str,
    out_path: Path,
    summary_out_path: Path,
    max_steps: int = 12,
    num_tasks: int = 10,
    task_concurrency: int = 1,
    task_cache: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "training.post_finetune_eval",
        "--project-id",
        str(project_id or "autocinema"),
        "--adapter-path",
        str(adapter_path),
        "--endpoint",
        endpoint,
        "--provider",
        "openai",
        "--served-model-id",
        served_model_id,
        "--use-case",
        use_case,
        "--num-tasks",
        str(max(1, int(num_tasks))),
        "--max-steps",
        str(int(max_steps)),
        "--task-concurrency",
        str(max(1, int(task_concurrency))),
        "--out",
        str(out_path),
        "--summary-out",
        str(summary_out_path),
    ]
    if task_cache is not None:
        cmd.extend(["--task-cache", str(task_cache)])
    return cmd
