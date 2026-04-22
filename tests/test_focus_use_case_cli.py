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


def test_teacher_harvest_deterministic_only_forces_single_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_collect(*, config, seeds, strategy, collect_workers):
        captured["max_claude_attempts"] = int(config.max_claude_attempts)
        captured["seeds"] = list(seeds)
        captured["strategy"] = str(strategy)
        captured["collect_workers"] = int(collect_workers)
        return [{"harvest_mode": "deterministic_rule", "model": "deterministic"}]

    def _fake_write(**_kwargs):
        return (tmp_path / "episodes.jsonl", tmp_path / "summary.json", {"ai_assisted_attempts_total": 0, "deterministic_attempts_total": 1})

    monkeypatch.setattr(focus_use_case_module, "collect_rows_for_seeds", _fake_collect)
    monkeypatch.setattr(focus_use_case_module, "write_harvest_artifacts", _fake_write)

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

    result = focus_use_case_module.cmd_claude_harvest(args)

    assert result == 0
    assert captured["max_claude_attempts"] == 1
