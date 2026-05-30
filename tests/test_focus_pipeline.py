from __future__ import annotations

import json
from pathlib import Path

import training.focus_pipeline as focus_pipeline_module
from training.focus_pipeline import (
    build_focus_eval_command,
    build_focus_summary,
    build_prompt_override,
    build_task_cache_override,
    default_task_cache_for_project,
    focus_root,
    write_focus_artifacts,
)
from training.login_dagger import login_dagger_extra_lines


def test_build_task_cache_override_appends_prompt_for_target_use_case(tmp_path: Path) -> None:
    source = tmp_path / "tasks.json"
    source.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "1",
                        "prompt": "Base login prompt.",
                        "use_case": {"name": "LOGIN"},
                    },
                    {
                        "id": "2",
                        "prompt": "Other prompt.",
                        "use_case": {"name": "SEARCH_FILM"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "override.json"
    build_task_cache_override(
        source_task_cache=source,
        use_case="LOGIN",
        prompt_override="Navigate directly to /login first.",
        out_path=out,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    assert "<password>" in tasks[0]["prompt"]
    assert "<username>" in tasks[0]["prompt"]
    assert "Navigate directly to /login first." in tasks[0]["prompt"]
    assert tasks[1]["prompt"] == "Other prompt."


def test_build_task_cache_override_supports_nested_project_task_cache(tmp_path: Path) -> None:
    source = tmp_path / "nested_tasks.json"
    source.write_text(
        json.dumps(
            {
                "autocinema": {
                    "project_id": "autocinema",
                    "tasks": [
                        {
                            "id": "1",
                            "prompt": "Base watchlist prompt.",
                            "use_case": {"name": "ADD_TO_WATCHLIST"},
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "override_nested.json"
    build_task_cache_override(
        source_task_cache=source,
        use_case="ADD_TO_WATCHLIST",
        prompt_override="Open the target movie detail page before adding it.",
        out_path=out,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    tasks = payload["autocinema"]["tasks"]
    assert "Base watchlist prompt." in tasks[0]["prompt"]
    assert "Open the target movie detail page before adding it." in tasks[0]["prompt"]


def test_build_focus_summary_counts_only_gold_rows() -> None:
    summary = build_focus_summary(
        use_case="LOGIN",
        target_seeds=[1, 2, 3],
        rows=[
            {"seed": 1, "success": True, "score": 1.0},
            {"seed": 2, "success": False, "score": 0.0},
            {"seed": 2, "success": True, "score": 1.0},
            {"seed": 3, "success": False, "score": 0.5},
        ],
    )
    assert summary["gold_episodes_total"] == 2
    assert summary["gold_seeds"] == [1, 2]
    assert summary["failed_seeds"] == [3]
    assert summary["passed_target"] is False


def test_build_prompt_override_for_known_use_case() -> None:
    prompt = build_prompt_override(use_case="LOGIN")
    assert "/login" in prompt
    assert "registration" in prompt.lower()


def test_focus_root_uses_use_case_slug() -> None:
    root = focus_root(use_case="LOGIN")
    assert str(root).endswith("data/autocinema/login")


def test_focus_root_uses_canonical_layout_for_non_login() -> None:
    root = focus_root(use_case="CONTACT")
    assert str(root).endswith("data/autocinema/contact")


def test_default_task_cache_for_project_uses_generic_cache_when_project_specific_file_missing(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "autoppia_operator"
    task_cache_dir = repo_root / "data" / "task_cache"
    task_cache_dir.mkdir(parents=True)
    generic_cache = task_cache_dir / "tasks_cache.json"
    generic_cache.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "film-detail-1",
                        "web_project_id": "autocinema",
                        "url": "http://localhost:8090/?seed=1",
                        "prompt": "Open a film detail page.",
                        "use_case": {"name": "FILM_DETAIL"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(focus_pipeline_module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(focus_pipeline_module, "DEFAULT_TASK_CACHE", repo_root / "missing_default.json")

    resolved = default_task_cache_for_project("autocinema")

    assert resolved == generic_cache


def test_login_dagger_extra_lines_emphasize_non_repetition() -> None:
    lines = login_dagger_extra_lines(failure_category="NO_PROGRESS_LOOP")
    text = " ".join(lines).lower()
    assert "do not repeat" in text or "break repetition" in text
    assert "password" in text


def test_write_focus_artifacts_merges_existing_attempts(tmp_path: Path) -> None:
    output_root = tmp_path / "focus"
    _, _, summary1 = write_focus_artifacts(
        output_root=output_root,
        use_case="LOGIN",
        target_seeds=[1, 2],
        rows=[
            {"seed": 1, "attempt_name": "baseline", "success": True, "score": 1.0},
        ],
    )
    assert summary1["gold_seeds"] == [1]

    episodes_path, _, summary2 = write_focus_artifacts(
        output_root=output_root,
        use_case="LOGIN",
        target_seeds=[1, 2],
        rows=[
            {"seed": 2, "attempt_name": "baseline", "success": True, "score": 1.0},
        ],
    )
    episode_rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert summary2["gold_seeds"] == [1, 2]
    assert [int(row["seed"]) for row in episode_rows] == [1, 2]


def test_episode_row_from_report_prefers_episode_seed(tmp_path: Path) -> None:
    row = focus_pipeline_module._episode_row_from_report(
        report={
            "model": "gpt-5.4",
            "episodes": [
                {
                    "task_id": "task-1",
                    "episode_task_id": "ep-1",
                    "seed": 418,
                    "success": True,
                    "score": 1.0,
                    "steps": 1,
                    "final_url": "http://localhost:8000/contact?seed=418",
                }
            ],
        },
        use_case="CONTACT",
        seed=96,
        attempt_name="baseline",
        out_path=tmp_path / "runs" / "seed_0096_baseline.json",
        trace_dir=tmp_path / "traces" / "seed_0096_baseline",
    )

    assert row is not None
    assert row["seed"] == 418
    assert "seed=418" in row["notes"]


def test_build_focus_eval_command_supports_parallel_eval() -> None:
    cmd = build_focus_eval_command(
        project_id="autocinema",
        use_case="LOGIN",
        adapter_path=Path("/tmp/adapter"),
        endpoint="http://127.0.0.1:8000/v1",
        served_model_id="autoppia",
        out_path=Path("/tmp/out.json"),
        summary_out_path=Path("/tmp/summary.json"),
        max_steps=12,
        num_tasks=100,
        task_concurrency=4,
        task_cache=Path("/tmp/task_cache.json"),
    )
    joined = " ".join(cmd)
    assert "--num-tasks 100" in joined
    assert "--task-concurrency 4" in joined
    assert "--task-cache /tmp/task_cache.json" in joined


def test_run_eval_attempt_keeps_direct_loop_enabled(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, cwd, env, check):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["check"] = check
        out_idx = cmd.index("--out") + 1
        Path(cmd[out_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[out_idx]).write_text(json.dumps({"episodes": []}), encoding="utf-8")

    monkeypatch.setattr(focus_pipeline_module.subprocess, "run", fake_run)
    monkeypatch.setattr(focus_pipeline_module, "_episode_row_from_report", lambda **_: None)

    focus_pipeline_module.run_eval_attempt(
        use_case="LOGIN",
        seed=252,
        attempt_name="baseline",
        output_root=tmp_path,
        provider="openai",
        model="gpt-5.4",
        max_steps=12,
        web_project_id="autobooks",
    )

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["FSM_DIRECT_LOOP"] == "1"
    assert env["EVAL_CAPTURE_SCREENSHOT"] == "0"
    assert env["EVALUATOR_HEADLESS"] == "1"
    assert "--web-project-id" in captured["cmd"]
    assert "autobooks" in captured["cmd"]
    assert "--trace-full-payloads" not in captured["cmd"]


def test_run_eval_attempt_supports_headed_browser(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, cwd, env, check):
        captured["env"] = env
        out_idx = cmd.index("--out") + 1
        Path(cmd[out_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[out_idx]).write_text(json.dumps({"episodes": []}), encoding="utf-8")

    monkeypatch.setattr(focus_pipeline_module.subprocess, "run", fake_run)
    monkeypatch.setattr(focus_pipeline_module, "_episode_row_from_report", lambda **_: None)

    focus_pipeline_module.run_eval_attempt(
        use_case="LOGIN",
        seed=7,
        attempt_name="baseline",
        output_root=tmp_path,
        provider="openai",
        model="gpt-5.4",
        max_steps=12,
        headed=True,
    )

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["EVALUATOR_HEADLESS"] == "0"
