from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.eval.focus_use_case as focus_use_case_module


def test_resolve_seed_list_is_project_scoped(tmp_path: Path) -> None:
    cache_path = tmp_path / "tasks.json"
    cache_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "a1",
                        "web_project_id": "autocinema",
                        "url": "http://localhost:3000/?seed=11",
                        "prompt": "Login",
                        "use_case": {"name": "LOGIN"},
                    },
                    {
                        "id": "b1",
                        "web_project_id": "autobooks",
                        "url": "http://localhost:3001/?seed=22",
                        "prompt": "Login",
                        "use_case": {"name": "LOGIN"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    seeds = focus_use_case_module._resolve_seed_list(
        use_case="LOGIN",
        seed_spec="",
        task_cache=str(cache_path),
        project_id="autobooks",
    )

    assert seeds == [22]


def test_teacher_harvest_deterministic_only_rejects_multi_attempts(tmp_path: Path) -> None:
    args = argparse.Namespace(
        project_id="autocinema",
        use_case="LOGIN",
        seeds="1",
        provider="openai",
        model="gpt-5.4-mini",
        brief_model="gpt-5.4-mini",
        max_steps=12,
        task_concurrency=1,
        agent_workers=1,
        collect_workers=1,
        claude_workers=1,
        replay_workers=1,
        task_cache=str(tmp_path / "tasks.json"),
        max_claude_attempts=2,
        claude_timeout_seconds=120,
        execution_mode="direct",
        deterministic_only=True,
        headed=False,
        no_merge_existing=False,
    )

    with pytest.raises(ValueError, match="deterministic-only"):
        focus_use_case_module.cmd_claude_harvest(args)
