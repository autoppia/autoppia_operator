from __future__ import annotations

"""Compatibility imports for the historical Claude Code harvester module.

New operator code should live under `training.autoppia_operator`.
"""

from training.autoppia_operator.briefs import (
    DEFAULT_BRIEF_MODEL,
    REPO_ROOT,
    TASK_CACHE_PATH,
    WEB_REPO_ROOT,
    brief_prompt_lines,
    focus_root,
    generate_claude_brief,
    save_claude_brief,
    summarize_attempt_for_claude,
)
from training.autoppia_operator.discovery import run_claude_code_harvest

__all__ = [
    "DEFAULT_BRIEF_MODEL",
    "REPO_ROOT",
    "TASK_CACHE_PATH",
    "WEB_REPO_ROOT",
    "brief_prompt_lines",
    "focus_root",
    "generate_claude_brief",
    "run_claude_code_harvest",
    "save_claude_brief",
    "summarize_attempt_for_claude",
]
