from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.focus_pipeline import DEFAULT_TASK_CACHE, build_prompt_override, build_task_cache_override, run_eval_attempt
from training.layout import use_case_layout
from training.use_case_registry import dagger_extra_lines

REPO_ROOT = Path(__file__).resolve().parents[1]


def focus_root(*, use_case: str) -> Path:
    return use_case_layout(repo_root=REPO_ROOT, web_project="autocinema", use_case=use_case).root


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_results(summary_path: Path) -> list[dict[str, Any]]:
    payload = _load_json(summary_path)
    results = payload.get("results")
    return [row for row in results if isinstance(row, dict)] if isinstance(results, list) else []


def run_dagger_round(
    *,
    use_case: str,
    source_summary_path: Path,
    split_name: str,
    teacher_provider: str = "openai",
    teacher_model: str = "gpt-5.4",
    max_steps: int = 12,
    task_cache_path: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    normalized_use_case = str(use_case or "").strip().upper()
    source_results = _load_results(source_summary_path)
    failed = [row for row in source_results if not bool(row.get("success"))]
    output_root = focus_root(use_case=normalized_use_case) / "dagger" / split_name
    correction_rows: list[dict[str, Any]] = []
    merged_rows: list[dict[str, Any]] = []

    for row in failed:
        seed = int(row.get("seed") or 0)
        failure_category = str(row.get("failure_category") or "")
        extra_lines = dagger_extra_lines(use_case=normalized_use_case, failure_category=failure_category)
        prompt_override = build_prompt_override(use_case=normalized_use_case, extra_lines=extra_lines)
        out_task_cache = output_root / "task_cache" / f"{normalized_use_case.lower()}_seed_{seed:04d}.json"
        build_task_cache_override(
            source_task_cache=Path(task_cache_path or DEFAULT_TASK_CACHE).resolve(),
            use_case=normalized_use_case,
            prompt_override=prompt_override,
            out_path=out_task_cache,
        )
        corrected = run_eval_attempt(
            use_case=normalized_use_case,
            seed=seed,
            attempt_name="dagger_teacher",
            output_root=output_root,
            provider=teacher_provider,
            model=teacher_model,
            max_steps=max_steps,
            task_cache=out_task_cache,
            env_overrides=dict(env_overrides or {}),
        )
        correction_row = {
            "seed": seed,
            "split": split_name,
            "use_case": normalized_use_case,
            "failure_category": failure_category,
            "teacher_success": corrected.is_gold,
            "teacher_result_path": str(corrected.out_path),
            "teacher_trace_dir": str(corrected.trace_dir),
            "teacher_trace_file": str((corrected.row or {}).get("trace_file") or ""),
            "teacher_episode_task_id": str((corrected.row or {}).get("episode_task_id") or ""),
            "generated_at": _now_utc(),
        }
        correction_rows.append(correction_row)
        if corrected.row and corrected.is_gold:
            merged_rows.append(corrected.row)

    corrections_path = output_root / "corrections.jsonl"
    merged_path = output_root / "teacher_gold_episodes.jsonl"
    summary_path = output_root / "summary.json"
    _write_jsonl(corrections_path, correction_rows)
    _write_jsonl(merged_path, merged_rows)
    summary = {
        "split": split_name,
        "use_case": normalized_use_case,
        "source_summary_path": str(source_summary_path),
        "failed_seed_count": len(failed),
        "failed_seeds": [int(row.get("seed") or 0) for row in failed],
        "teacher_successes": sum(1 for row in correction_rows if row.get("teacher_success")),
        "teacher_failures": sum(1 for row in correction_rows if not row.get("teacher_success")),
        "generated_at": _now_utc(),
    }
    _write_json(summary_path, summary)
    return summary


def merge_dagger_episodes(*, base_episodes_path: Path, dagger_teacher_path: Path, output_path: Path) -> dict[str, Any]:
    base_rows = [json.loads(line) for line in base_episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    teacher_rows = [json.loads(line) for line in dagger_teacher_path.read_text(encoding="utf-8").splitlines() if line.strip()] if dagger_teacher_path.exists() else []
    merged = list(base_rows)
    merged.extend(row for row in teacher_rows if isinstance(row, dict))
    _write_jsonl(output_path, merged)
    return {
        "base_rows": len(base_rows),
        "teacher_rows": len(teacher_rows),
        "merged_rows": len(merged),
        "output_path": str(output_path),
    }
