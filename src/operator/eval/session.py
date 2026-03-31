from __future__ import annotations

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.evaluation.stateful_evaluator import TaskExecutionSession


def build_task_execution_session(
    *,
    task: Task,
    web_agent_id: str,
    validator_id: str,
    enable_score_cheating: bool,
    capture_screenshot: bool,
    headless: bool | None = None,
) -> TaskExecutionSession:
    return TaskExecutionSession(
        task=task,
        web_agent_id=web_agent_id,
        validator_id=validator_id,
        enable_score_cheating=bool(enable_score_cheating),
        should_record_gif=False,
        capture_screenshot=bool(capture_screenshot),
        headless=headless,
    )


__all__ = ["build_task_execution_session"]
