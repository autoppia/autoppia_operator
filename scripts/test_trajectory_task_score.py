#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
IWA_ROOT = ROOT.parent / "autoppia_iwa"
iwa_root_str = str(IWA_ROOT)
if IWA_ROOT.exists() and iwa_root_str not in sys.path:
    sys.path.insert(0, iwa_root_str)

# autoppia_iwa config enforces provider keys at import time.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from autoppia_iwa.src.data_generation.tasks.classes import Task
from eval import _normalize_task_url_for_project
from src.operator.agents.fsm.trajectory import get_trajectory_replay_bundle
from src.operator.runtime.trajectory_executor import TrajectoryExecutor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run strict trajectory replay for one task or all use-cases in a project, "
            "then print final scores."
        )
    )
    parser.add_argument(
        "--task-cache",
        default="",
        help=(
            "Optional cache file path. If omitted, the script auto-detects a cache that contains "
            "the requested project and creates one if none is found."
        ),
    )
    parser.add_argument("--web-project-id", default="autocinema")
    parser.add_argument("--use-case", default=None)
    parser.add_argument(
        "--all-use-cases",
        action="store_true",
        help=(
            "Run strict replay for all use-cases in the selected project (one task per use-case). "
            "If --use-case and --task-id are both omitted, this mode is enabled automatically."
        ),
    )
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--web-agent-id", default="1")
    parser.add_argument("--validator-id", default="trajectory-test-validator")
    parser.add_argument(
        "--raw-placeholders",
        action="store_true",
        help=(
            "Run strict replay with raw trajectory values. By default placeholders are resolved "
            "from task prompt before execution."
        ),
    )
    parser.add_argument("--capture-screenshot", action="store_true")
    parser.add_argument(
        "--expect-non-zero",
        action="store_true",
        help="Exit code 1 if final score is 0.0 (useful for CI checks).",
    )
    parser.add_argument(
        "--iwa-log-level",
        default="ERROR",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help=(
            "Log level for internal autoppia_iwa loguru logs. "
            "Use INFO/DEBUG to inspect evaluator internals, ERROR to keep output focused on trajectory debugging."
        ),
    )
    return parser.parse_args()


def _load_raw_tasks(cache_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("tasks")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    raise ValueError(f"Unsupported cache format in {cache_path}")


def _has_project_tasks(raw_tasks: list[dict[str, Any]], project_id: str) -> bool:
    wanted = str(project_id or "").strip()
    if not wanted:
        return False
    for task in raw_tasks:
        if str(task.get("web_project_id") or "").strip() == wanted:
            return True
    return False


def _try_load_cache_for_project(cache_path: Path, project_id: str) -> tuple[bool, list[dict[str, Any]]]:
    try:
        raw_tasks = _load_raw_tasks(cache_path)
    except Exception:
        return False, []
    return _has_project_tasks(raw_tasks, project_id), raw_tasks


def _auto_generate_cache(cache_path: Path, project_id: str) -> None:
    cmd = [
        # Keep the current interpreter to preserve active venv/site-packages.
        str(Path(sys.executable)),
        str((ROOT / "scripts" / "eval" / "generate_tasks.py").resolve()),
        "--project-id",
        str(project_id),
        "--prompts-per-use-case",
        "1",
        "--out",
        str(cache_path.resolve()),
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if int(proc.returncode) != 0:
        raise RuntimeError(
            f"Could not auto-generate task cache for project={project_id!r} at {cache_path}. "
            "Run scripts/eval/generate_tasks.py manually."
        )


def _resolve_cache_and_tasks(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    project_id = str(args.web_project_id or "").strip()
    explicit_cache = str(args.task_cache or "").strip()

    if explicit_cache:
        cache_path = Path(explicit_cache)
        if not cache_path.is_absolute():
            cache_path = (ROOT / explicit_cache).resolve()
        if not cache_path.exists():
            raise RuntimeError(f"task cache not found: {cache_path}")
        ok, raw_tasks = _try_load_cache_for_project(cache_path, project_id)
        if not ok:
            raise RuntimeError(
                f"task cache {cache_path} does not contain tasks for web_project_id={project_id!r}"
            )
        return cache_path, raw_tasks

    candidates: list[Path] = [
        (ROOT / "data" / "task_cache" / f"{project_id}_tasks_cache.json").resolve(),
        (ROOT / "data" / "task_cache" / "tasks_cache.json").resolve(),
        Path(f"/tmp/{project_id}_tasks_cache_seed1.json").resolve(),
    ]
    tmp_glob = sorted(Path("/tmp").glob(f"{project_id}_tasks_cache*.json"))
    candidates.extend([p.resolve() for p in tmp_glob])

    seen: set[str] = set()
    for cache_path in candidates:
        key = str(cache_path)
        if key in seen or not cache_path.exists():
            continue
        seen.add(key)
        ok, raw_tasks = _try_load_cache_for_project(cache_path, project_id)
        if ok:
            return cache_path, raw_tasks

    generated = (ROOT / "data" / "task_cache" / f"{project_id}_tasks_cache.json").resolve()
    _auto_generate_cache(generated, project_id)
    ok, raw_tasks = _try_load_cache_for_project(generated, project_id)
    if not ok:
        raise RuntimeError(
            f"Auto-generated cache at {generated} still does not contain project={project_id!r} tasks."
        )
    return generated, raw_tasks


def _extract_use_case_name(task_dict: dict[str, Any]) -> str:
    use_case = task_dict.get("use_case")
    if isinstance(use_case, dict):
        return str(use_case.get("name") or "").strip()
    if isinstance(use_case, str):
        return use_case.strip()
    return ""


def _materialize_task(task_dict: dict[str, Any]) -> Task:
    normalized = _normalize_task_url_for_project(task_dict)
    task = Task(**normalized)
    task.url = str(task.url or "")
    return task


def _pick_task(raw_tasks: list[dict[str, Any]], args: argparse.Namespace) -> Task:
    project_id = str(args.web_project_id or "").strip()
    use_case_filter = str(args.use_case or "").strip().upper()
    task_id_filter = str(args.task_id or "").strip()
    selected: dict[str, Any] | None = None

    for task_dict in raw_tasks:
        if str(task_dict.get("web_project_id") or "").strip() != project_id:
            continue
        if task_id_filter and str(task_dict.get("id") or "").strip() != task_id_filter:
            continue
        if use_case_filter:
            uc = _extract_use_case_name(task_dict).upper()
            if uc != use_case_filter:
                continue
        selected = task_dict
        break

    if selected is None:
        raise RuntimeError(
            f"No task found for web_project_id={args.web_project_id!r}, "
            f"use_case={args.use_case!r}, task_id={args.task_id!r}"
        )

    return _materialize_task(selected)


def _pick_all_use_case_tasks(raw_tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[Task]:
    project_id = str(args.web_project_id or "").strip()
    first_by_use_case: dict[str, dict[str, Any]] = {}

    for task_dict in raw_tasks:
        if str(task_dict.get("web_project_id") or "").strip() != project_id:
            continue
        use_case_name = _extract_use_case_name(task_dict).strip()
        if not use_case_name:
            continue
        uc_key = use_case_name.upper()
        if uc_key not in first_by_use_case:
            first_by_use_case[uc_key] = task_dict

    if not first_by_use_case:
        raise RuntimeError(f"No tasks found for web_project_id={args.web_project_id!r}")

    ordered_keys = sorted(first_by_use_case.keys())
    return [_materialize_task(first_by_use_case[key]) for key in ordered_keys]


def _extract_first_navigate_url(actions: list[dict[str, Any]]) -> str:
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("type") or "") != "NavigateAction":
            continue
        url = str(action.get("url") or "").strip()
        if url:
            return url
    return ""


def _looks_like_placeholder_payload(actions: list[dict[str, Any]]) -> bool:
    import re

    token_re = re.compile(r"__[^_]{1,80}__|<[^>]{1,80}>")
    for action in actions:
        if not isinstance(action, dict):
            continue
        blob = json.dumps(action, ensure_ascii=False)
        if token_re.search(blob):
            return True
    return False


def _run_single_task(args: argparse.Namespace, cache_path: Path, task: Task) -> float:
    from eval import _ScopedAsyncStatefulEvaluator  # delayed import
    import asyncio

    use_case_name = _extract_use_case_name(task.model_dump())
    resolve_placeholders = not bool(args.raw_placeholders)
    selected_trajectory = get_trajectory_replay_bundle(
        web_project_id=str(task.web_project_id or args.web_project_id),
        use_case=use_case_name,
        prompt=str(task.prompt or ""),
        apply_prompt_overrides=resolve_placeholders,
    )
    actions = selected_trajectory.get("actions") if isinstance(selected_trajectory.get("actions"), list) else []
    if _looks_like_placeholder_payload(actions):
        print("Warning: trajectory actions still contain placeholder tokens.")
        print("         Use default mode (without --raw-placeholders) to resolve from task prompt.")

    trajectory_url = str(selected_trajectory.get("url") or "").strip()
    navigate_url = _extract_first_navigate_url(actions)
    replay_url = navigate_url or trajectory_url
    if replay_url:
        task.url = replay_url
    if not actions:
        print("No trajectory actions found for this task.")
        return 0.0

    executor = TrajectoryExecutor()
    mapped_payloads = executor.to_iwa_action_payloads(actions)
    iwa_actions = executor.to_iwa_actions(actions)

    print("Task selected:")
    print(f"- id: {task.id}")
    print(f"- web_project_id: {task.web_project_id}")
    print(f"- use_case: {use_case_name}")
    print(f"- mode: strict-replay")
    print(f"- task_cache: {cache_path}")
    print(f"- url: {task.url}")
    print(f"- prompt: {task.prompt}")
    print(f"- trajectory_url: {str(selected_trajectory.get('url') or '')}")
    print(f"- strict_resolve_placeholders: {resolve_placeholders}")
    print("")
    print("Task tests:")
    if getattr(task, "tests", None):
        for idx, test in enumerate(task.tests, 1):
            event_name = str(getattr(test, "event_name", "") or getattr(test, "type", "") or "UNKNOWN")
            criteria = getattr(test, "event_criteria", None)
            print(f"  {idx:02d}. {event_name} criteria={criteria}")
    else:
        print("  (no tests)")
    print("")
    print("Trajectory actions:")
    print(json.dumps(actions, indent=2, ensure_ascii=False))
    print("")
    print("Mapped IWA payloads:")
    print(json.dumps(mapped_payloads, indent=2, ensure_ascii=False))

    async def _async_run() -> float:
        evaluator = _ScopedAsyncStatefulEvaluator(
            task=task,
            web_agent_id=str(args.web_agent_id),
            validator_id=str(args.validator_id),
            enable_score_cheating=False,
            capture_screenshot=bool(args.capture_screenshot),
        )
        try:
            reset_result = await evaluator.reset()
            print("")
            print(
                f"After reset -> score={reset_result.score.raw_score:.3f} "
                f"({reset_result.score.tests_passed}/{reset_result.score.total_tests}) "
                f"url={reset_result.snapshot.url}"
            )
            last_score = float(reset_result.score.raw_score)
            step_reports: list[dict[str, Any]] = []
            for idx, action in enumerate(iwa_actions, 1):
                result = await evaluator.step(action)
                last_score = float(result.score.raw_score)
                action_result = getattr(result, "action_result", None)
                exec_ok = bool(getattr(action_result, "successfully_executed", False))
                error_text = str(getattr(action_result, "error", "") or "").strip()
                current_url = str(getattr(getattr(result, "snapshot", None), "url", "") or "")
                step_reports.append(
                    {
                        "step": idx,
                        "action": action.__class__.__name__,
                        "score": float(result.score.raw_score),
                        "tests_passed": int(result.score.tests_passed),
                        "total_tests": int(result.score.total_tests),
                        "exec_ok": exec_ok,
                        "error": error_text,
                        "url": current_url,
                    }
                )
                print(
                    f"Step {idx:02d} {action.__class__.__name__}: "
                    f"score={result.score.raw_score:.3f} "
                    f"({result.score.tests_passed}/{result.score.total_tests}) "
                    f"exec_ok={exec_ok} "
                    f"url={current_url}"
                )
                if error_text:
                    print(f"  error: {error_text}")
                if bool(result.score.success):
                    break
            final_score = await evaluator.get_score_details()
            print("")
            print(
                f"Final score={final_score.raw_score:.3f} "
                f"({final_score.tests_passed}/{final_score.total_tests}) "
                f"success={final_score.success}"
            )
            first_non_zero = next((s for s in step_reports if float(s["score"]) > 0.0), None)
            failed_steps = [s for s in step_reports if not bool(s["exec_ok"])]
            print("")
            print("Trajectory debug summary:")
            print(f"- steps_executed: {len(step_reports)}/{len(iwa_actions)}")
            if first_non_zero:
                print(
                    f"- first_non_zero_step: {int(first_non_zero['step']):02d} "
                    f"({first_non_zero['action']}) score={float(first_non_zero['score']):.3f}"
                )
            else:
                print("- first_non_zero_step: none")
            if failed_steps:
                print("- execution_failures:")
                for item in failed_steps:
                    print(
                        f"  step {int(item['step']):02d} {item['action']} "
                        f"error={str(item['error']) or 'unknown'}"
                    )
            else:
                print("- execution_failures: none")
            return float(final_score.raw_score if final_score.total_tests else last_score)
        finally:
            await evaluator.close()

    return asyncio.run(_async_run())


def _run(args: argparse.Namespace) -> int:
    _configure_iwa_logs(str(args.iwa_log_level))
    cache_path, raw_tasks = _resolve_cache_and_tasks(args)

    run_all_use_cases = bool(args.all_use_cases) or (not args.use_case and not args.task_id)
    if run_all_use_cases:
        tasks = _pick_all_use_case_tasks(raw_tasks, args)
        print(
            f"Batch mode: web_project_id={args.web_project_id} "
            f"use_cases={len(tasks)} strict_replay=True"
        )
        failed_use_cases: list[str] = []
        for index, task in enumerate(tasks, 1):
            use_case_name = _extract_use_case_name(task.model_dump()) or "UNKNOWN"
            print("")
            print("=" * 90)
            print(f"[{index}/{len(tasks)}] use_case={use_case_name}")
            print("=" * 90)
            final_score = _run_single_task(args, cache_path, task)
            if final_score <= 0.0:
                failed_use_cases.append(use_case_name)
                print("Score is still 0.0")

        print("")
        print("Batch summary:")
        print(f"- total_use_cases: {len(tasks)}")
        print(f"- passed: {len(tasks) - len(failed_use_cases)}")
        print(f"- failed: {len(failed_use_cases)}")
        if failed_use_cases:
            print(f"- failed_use_cases: {', '.join(failed_use_cases)}")
        if args.expect_non_zero and failed_use_cases:
            return 1
        return 0

    task = _pick_task(raw_tasks, args)
    final_score = _run_single_task(args, cache_path, task)
    if args.expect_non_zero and final_score <= 0.0:
        print("Score is still 0.0")
        return 1
    return 0


def _configure_iwa_logs(level: str) -> None:
    """
    Reduce or increase internal autoppia_iwa logging noise for trajectory-focused debugging.
    """
    try:
        from loguru import logger as loguru_logger

        normalized_level = str(level or "ERROR").upper()
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level=normalized_level)
    except Exception:
        # If loguru configuration fails, proceed with default logging behavior.
        return


def main() -> int:
    args = _parse_args()
    try:
        return _run(args)
    except Exception as exc:
        print(f"{exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
