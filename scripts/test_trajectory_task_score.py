#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# autoppia_iwa config enforces provider keys at import time.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from autoppia_iwa.src.data_generation.tasks.classes import Task
from eval import _normalize_task_url_for_project
from src.operator.agents.fsm.trajectory import get_trajectory_bootstrap_actions
from src.operator.runtime.trajectory_executor import TrajectoryExecutor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Autoppia IWA task with trajectory actions only, then print final score "
            "to check if it stays at 0 or not."
        )
    )
    parser.add_argument("--task-cache", default="data/task_cache/tasks_cache.json")
    parser.add_argument("--web-project-id", default="autocinema")
    parser.add_argument("--use-case", default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--web-agent-id", default="1")
    parser.add_argument("--validator-id", default="trajectory-test-validator")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument(
        "--keep-navigate",
        action="store_true",
        help="Keep NavigateAction steps from trajectories. By default they are skipped to mirror FSM bootstrap.",
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


def _extract_use_case_name(task_dict: dict[str, Any]) -> str:
    use_case = task_dict.get("use_case")
    if isinstance(use_case, dict):
        return str(use_case.get("name") or "").strip()
    if isinstance(use_case, str):
        return use_case.strip()
    return ""


def _pick_task(raw_tasks: list[dict[str, Any]], args: argparse.Namespace) -> Task:
    selected: dict[str, Any] | None = None
    for task_dict in raw_tasks:
        if str(task_dict.get("web_project_id") or "").strip() != str(args.web_project_id).strip():
            continue
        if args.task_id and str(task_dict.get("id") or "").strip() != str(args.task_id).strip():
            continue
        if args.use_case:
            uc = _extract_use_case_name(task_dict).upper()
            if uc != str(args.use_case).strip().upper():
                continue
        selected = task_dict
        break

    if selected is None:
        raise RuntimeError(
            f"No task found for web_project_id={args.web_project_id!r}, "
            f"use_case={args.use_case!r}, task_id={args.task_id!r}"
        )

    normalized = _normalize_task_url_for_project(selected)
    task = Task(**normalized)
    task.url = _force_seed_in_url(str(task.url or ""), int(args.seed))
    return task


def _force_seed_in_url(url: str, seed: int) -> str:
    split = urlsplit(str(url or "").strip())
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    query["seed"] = str(int(seed))
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))


def _run(args: argparse.Namespace) -> int:
    from eval import _ScopedAsyncStatefulEvaluator  # delayed import

    _configure_iwa_logs(str(args.iwa_log_level))

    cache_path = (ROOT / args.task_cache).resolve()
    raw_tasks = _load_raw_tasks(cache_path)
    task = _pick_task(raw_tasks, args)
    use_case_name = _extract_use_case_name(task.model_dump())

    actions = get_trajectory_bootstrap_actions(
        web_project_id=str(task.web_project_id or args.web_project_id),
        use_case=use_case_name,
        prompt=str(task.prompt or ""),
        max_actions=int(args.max_actions),
    )
    if not bool(args.keep_navigate):
        actions = [a for a in actions if str(a.get("type") or "") != "NavigateAction"]
    if not actions:
        print("No trajectory actions found for this task.")
        return 2

    executor = TrajectoryExecutor()
    mapped_payloads = executor.to_iwa_action_payloads(actions)
    iwa_actions = executor.to_iwa_actions(actions)

    print("Task selected:")
    print(f"- id: {task.id}")
    print(f"- web_project_id: {task.web_project_id}")
    print(f"- use_case: {use_case_name}")
    print(f"- url: {task.url}")
    print(f"- prompt: {task.prompt}")
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

    import asyncio

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

    final_score = asyncio.run(_async_run())
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
