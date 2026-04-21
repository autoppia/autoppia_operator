from __future__ import annotations

import json
from pathlib import Path

import training.harvester as harvester_module
from training.harvester import HarvestConfig, collect_rows_for_seeds
from training.trajectory_candidate import TrajectoryCandidate, write_candidate


def test_collect_rows_for_seeds_baseline_uses_unified_worker(monkeypatch, tmp_path: Path) -> None:
    captured: list[int] = []

    def fake_collect_seed_rows(*, config, seed):
        captured.append(seed)
        return [{"seed": seed, "attempt_name": "baseline", "success": True, "score": 1.0}]

    monkeypatch.setattr(harvester_module, "collect_seed_rows", fake_collect_seed_rows)
    config = HarvestConfig(
        use_case="LOGIN",
        output_root=tmp_path,
        provider="openai",
        model="gpt-5.4",
        task_cache_arg=str(tmp_path / "tasks.json"),
    )
    rows = collect_rows_for_seeds(config=config, seeds=[1, 2], strategy="baseline", collect_workers=1)
    assert [row["seed"] for row in rows] == [1, 2]
    assert captured == [1, 2]


def test_collect_rows_for_seeds_code_aware_uses_code_aware_worker(monkeypatch, tmp_path: Path) -> None:
    generated: list[int] = []
    replayed: list[int] = []
    (tmp_path / "tasks.json").write_text(json.dumps({"tasks": []}), encoding="utf-8")

    def fake_generate_candidate_attempt(*, config, seed, attempt_idx, prior_attempts):
        candidate = TrajectoryCandidate(
            use_case="CONTACT",
            seed=int(seed),
            attempt_name=f"claude_{attempt_idx:02d}",
            teacher_model="claude-sonnet-4-5",
            generation_mode="claude_code_brief",
            brief_path=str(tmp_path / f"brief_{seed}.json"),
            prompt_lines=(),
            actions=({"type": "NavigateAction", "url": f"http://example.test/contact?seed={seed}"},),
            metadata={},
        )
        candidate_file = tmp_path / f"candidate_{seed}.json"
        write_candidate(candidate_file, candidate)
        generated.append(seed)
        return {
            "seed": seed,
            "attempt_idx": attempt_idx,
            "attempt_name": f"claude_{attempt_idx:02d}",
            "brief_path": tmp_path / f"brief_{seed}.json",
            "stored_candidate_path": candidate_file,
            "prompt_override": "",
            "extra_lines": [],
            "task_cache_path": tmp_path / "tasks.json",
            "candidate": candidate,
            "brief_payload": {},
        }

    def fake_execute_candidate_attempt(*, config, bundle):
        replayed.append(int(bundle["seed"]))
        return {
            "seed": int(bundle["seed"]),
            "row": {"seed": int(bundle["seed"]), "attempt_name": "claude_01", "success": False, "score": 0.0},
            "feedback": {"attempt_name": "claude_01"},
            "is_gold": False,
            "report_payload": {},
        }

    monkeypatch.setattr(harvester_module, "_generate_candidate_attempt", fake_generate_candidate_attempt)
    monkeypatch.setattr(harvester_module, "_execute_candidate_attempt", fake_execute_candidate_attempt)
    monkeypatch.setattr(harvester_module, "_build_task_cache_for_seed", lambda **kwargs: tmp_path / "tasks.json")
    config = HarvestConfig(
        use_case="CONTACT",
        output_root=tmp_path,
        provider="openai",
        model="gpt-5.4-mini",
        task_cache_arg=str(tmp_path / "tasks.json"),
        max_claude_attempts=1,
    )
    rows = collect_rows_for_seeds(config=config, seeds=[7], strategy="code-aware", collect_workers=1)
    assert len(rows) == 2
    assert [row["attempt_name"] for row in rows] == ["claude_01", "claude_01"]
    assert generated == [7, 7]
    assert replayed == [7, 7]


def test_collect_rows_for_seeds_code_aware_uses_separate_phase_workers(monkeypatch, tmp_path: Path) -> None:
    generated: list[int] = []
    replayed: list[int] = []
    (tmp_path / "tasks.json").write_text(json.dumps({"tasks": []}), encoding="utf-8")

    def fake_generate_candidate_attempt(*, config, seed, attempt_idx, prior_attempts):
        candidate = TrajectoryCandidate(
            use_case="CONTACT",
            seed=int(seed),
            attempt_name=f"claude_{attempt_idx:02d}",
            teacher_model="claude-sonnet-4-5",
            generation_mode="claude_code_brief",
            brief_path=str(tmp_path / f"brief_{seed}.json"),
            prompt_lines=(),
            actions=({"type": "NavigateAction", "url": f"http://example.test/contact?seed={seed}"},),
            metadata={},
        )
        candidate_file = tmp_path / f"candidate_{seed}.json"
        write_candidate(candidate_file, candidate)
        generated.append(seed)
        return {
            "seed": seed,
            "attempt_idx": attempt_idx,
            "attempt_name": f"claude_{attempt_idx:02d}",
            "brief_path": tmp_path / f"brief_{seed}.json",
            "stored_candidate_path": candidate_file,
            "prompt_override": "",
            "extra_lines": [],
            "task_cache_path": tmp_path / "tasks.json",
            "candidate": candidate,
            "brief_payload": {},
        }

    def fake_execute_candidate_attempt(*, config, bundle):
        replayed.append(int(bundle["seed"]))
        return {
            "seed": int(bundle["seed"]),
            "row": {"seed": int(bundle["seed"]), "attempt_name": "claude_01", "success": int(bundle["seed"]) == 1, "score": 1.0 if int(bundle["seed"]) == 1 else 0.0},
            "feedback": {"attempt_name": "claude_01"},
            "is_gold": int(bundle["seed"]) == 1,
            "report_payload": {},
        }

    monkeypatch.setattr(harvester_module, "_generate_candidate_attempt", fake_generate_candidate_attempt)
    monkeypatch.setattr(harvester_module, "_execute_candidate_attempt", fake_execute_candidate_attempt)
    monkeypatch.setattr(harvester_module, "_build_task_cache_for_seed", lambda **kwargs: tmp_path / "tasks.json")
    config = HarvestConfig(
        use_case="CONTACT",
        output_root=tmp_path,
        provider="openai",
        model="gpt-5.4-mini",
        task_cache_arg=str(tmp_path / "tasks.json"),
        max_claude_attempts=2,
        claude_workers=2,
        replay_workers=3,
    )
    rows = collect_rows_for_seeds(config=config, seeds=[1, 2], strategy="code-aware", collect_workers=5)
    assert len(rows) == 4
    assert sorted(generated) == [1, 2, 2, 2]
    assert sorted(replayed) == [1, 2, 2, 2]


def test_collect_rows_for_seeds_code_aware_deterministic_only_skips_teacher_fallback(monkeypatch, tmp_path: Path) -> None:
    generated_attempts: list[int] = []
    replayed_attempts: list[int] = []
    (tmp_path / "tasks.json").write_text(json.dumps({"tasks": []}), encoding="utf-8")

    def fake_generate_candidate_attempt(*, config, seed, attempt_idx, prior_attempts):
        generated_attempts.append(int(attempt_idx))
        candidate = TrajectoryCandidate(
            use_case="CONTACT",
            seed=int(seed),
            attempt_name="deterministic_01",
            teacher_model="",
            generation_mode="deterministic_plan",
            brief_path=str(tmp_path / f"brief_{seed}.json"),
            prompt_lines=(),
            actions=({"type": "NavigateAction", "url": f"http://example.test/contact?seed={seed}"},),
            metadata={},
        )
        candidate_file = tmp_path / f"candidate_{seed}.json"
        write_candidate(candidate_file, candidate)
        return {
            "seed": seed,
            "attempt_idx": attempt_idx,
            "attempt_name": "deterministic_01",
            "brief_path": tmp_path / f"brief_{seed}.json",
            "stored_candidate_path": candidate_file,
            "prompt_override": "",
            "extra_lines": [],
            "task_cache_path": tmp_path / "tasks.json",
            "candidate": candidate,
            "brief_payload": {},
        }

    def fake_execute_candidate_attempt(*, config, bundle):
        replayed_attempts.append(int(bundle["attempt_idx"]))
        return {
            "seed": int(bundle["seed"]),
            "row": {"seed": int(bundle["seed"]), "attempt_name": "deterministic_01", "success": False, "score": 0.0},
            "feedback": {"attempt_name": "deterministic_01"},
            "is_gold": False,
            "report_payload": {},
        }

    monkeypatch.setattr(harvester_module, "_generate_candidate_attempt", fake_generate_candidate_attempt)
    monkeypatch.setattr(harvester_module, "_execute_candidate_attempt", fake_execute_candidate_attempt)
    config = HarvestConfig(
        use_case="CONTACT",
        output_root=tmp_path,
        provider="openai",
        model="gpt-5.4-mini",
        task_cache_arg=str(tmp_path / "tasks.json"),
        deterministic_only=True,
        max_claude_attempts=3,
    )
    rows = collect_rows_for_seeds(config=config, seeds=[7], strategy="code-aware", collect_workers=1)
    assert len(rows) == 1
    assert generated_attempts == [1]
    assert replayed_attempts == [1]


def test_generate_candidates_for_seeds_writes_candidate_files(monkeypatch, tmp_path: Path) -> None:
    def fake_generate_candidate_attempt(*, config, seed, attempt_idx, prior_attempts):
        candidate = TrajectoryCandidate(
            use_case="CONTACT",
            seed=int(seed),
            attempt_name=f"claude_{attempt_idx:02d}",
            teacher_model="claude-sonnet-4-5",
            generation_mode="claude_code_brief",
            brief_path=str(tmp_path / f"brief_{seed}.json"),
            prompt_lines=("go to contact",),
            actions=({"type": "NavigateAction", "url": f"http://example.test/contact?seed={seed}"},),
            metadata={},
        )
        path = tmp_path / "candidates" / f"seed_{int(seed):04d}_claude_{attempt_idx:02d}.json"
        write_candidate(path, candidate)
        return {
            "seed": seed,
            "attempt_idx": attempt_idx,
            "attempt_name": candidate.attempt_name,
            "brief_path": tmp_path / f"brief_{seed}.json",
            "stored_candidate_path": path,
            "prompt_override": "",
            "extra_lines": [],
            "task_cache_path": tmp_path / "tasks.json",
            "candidate": candidate,
            "brief_payload": {},
        }

    monkeypatch.setattr(harvester_module, "_generate_candidate_attempt", fake_generate_candidate_attempt)
    config = HarvestConfig(
        use_case="CONTACT",
        output_root=tmp_path,
        provider="openai",
        model="gpt-5.4-mini",
        task_cache_arg=str(tmp_path / "tasks.json"),
        max_claude_attempts=2,
        claude_workers=2,
    )
    candidate_paths = harvester_module.generate_candidates_for_seeds(config=config, seeds=[1, 2])
    assert len(candidate_paths) == 4
    assert all(path.exists() for path in candidate_paths)


def test_list_candidates_filters_by_seed(tmp_path: Path) -> None:
    for seed in (1, 2):
        candidate = TrajectoryCandidate(
            use_case="CONTACT",
            seed=seed,
            attempt_name="claude_01",
            teacher_model="claude-sonnet-4-5",
            generation_mode="claude_code_brief",
            brief_path=str(tmp_path / f"brief_{seed}.json"),
            prompt_lines=(),
            actions=({"type": "NavigateAction", "url": f"http://example.test/contact?seed={seed}"},),
            metadata={},
        )
        write_candidate(tmp_path / "candidates" / f"seed_{seed:04d}_claude_01.json", candidate)
    paths = harvester_module.list_candidates(output_root=tmp_path, seeds=[2])
    assert len(paths) == 1
    assert "seed_0002" in str(paths[0])


def test_replay_candidates_uses_saved_candidate_files(monkeypatch, tmp_path: Path) -> None:
    candidate = TrajectoryCandidate(
        use_case="CONTACT",
        seed=4,
        attempt_name="claude_01",
        teacher_model="claude-sonnet-4-5",
        generation_mode="claude_code_brief",
        brief_path=str(tmp_path / "brief.json"),
        prompt_lines=("go to contact",),
        actions=({"type": "NavigateAction", "url": "http://example.test/contact?seed=4"},),
        metadata={},
    )
    candidate_file = tmp_path / "candidates" / "seed_0004_claude_01.json"
    write_candidate(candidate_file, candidate)
    (tmp_path / "tasks.json").write_text(json.dumps({"tasks": []}), encoding="utf-8")

    def fake_execute_candidate_attempt(*, config, bundle):
        return {
            "seed": 4,
            "row": {"seed": 4, "attempt_name": "claude_01", "success": True, "score": 1.0},
            "feedback": {"attempt_name": "claude_01"},
            "is_gold": True,
            "report_payload": {},
        }

    monkeypatch.setattr(harvester_module, "_execute_candidate_attempt", fake_execute_candidate_attempt)
    monkeypatch.setattr(harvester_module, "_build_task_cache_for_seed", lambda **kwargs: tmp_path / "tasks.json")
    config = HarvestConfig(
        use_case="CONTACT",
        output_root=tmp_path,
        provider="openai",
        model="gpt-5.4-mini",
        task_cache_arg=str(tmp_path / "tasks.json"),
        replay_workers=2,
    )
    rows = harvester_module.replay_candidates(config=config, candidate_paths=[candidate_file])
    assert len(rows) == 1
    assert rows[0]["score"] == 1.0


def test_apply_row_provenance_sets_non_empty_hashes(tmp_path: Path) -> None:
    row = {
        "episode_task_id": "ep-1",
        "seed": 1,
        "use_case": "LOGIN",
        "success": True,
        "score": 1.0,
        "result_path": "result.json",
        "trace_dir": "trace-dir",
        "trace_file": "trace.json",
        "attempt_name": "baseline",
    }
    out = harvester_module._apply_row_provenance(
        row,
        task_cache_path=None,
        prompt_override="go to login",
        policy_mode="direct",
    )
    assert out is not None
    assert out["operator_version"] == "step_engine"
    assert out["policy_mode"] == "direct"
    assert out["task_cache_hash"]
    assert out["prompt_hash"]


def test_write_harvest_artifacts_merges_existing_attempts(tmp_path: Path) -> None:
    output_root = tmp_path / "harvest"
    _, _, summary1 = harvester_module.write_harvest_artifacts(
        output_root=output_root,
        use_case="LOGIN",
        target_seeds=[1, 2],
        rows=[{"seed": 1, "attempt_name": "baseline", "success": True, "score": 1.0}],
    )
    assert summary1["gold_seeds"] == [1]

    episodes_path, _, summary2 = harvester_module.write_harvest_artifacts(
        output_root=output_root,
        use_case="LOGIN",
        target_seeds=[1, 2],
        rows=[{"seed": 2, "attempt_name": "baseline", "success": True, "score": 1.0}],
    )
    rows = [line for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert summary2["gold_seeds"] == [1, 2]


def test_collect_rows_from_guided_brief_uses_shared_guided_row_builder(monkeypatch, tmp_path: Path) -> None:
    def fake_run_guided_brief(*, use_case, seed, brief_payload, task_cache, web_project_id=None, max_steps, headless=None):
        return {
            "model": "claude-sonnet-4-5",
            "episodes": [
                {
                    "task_id": f"task-{seed}",
                    "episode_task_id": f"ep-{seed}",
                    "success": True,
                    "score": 1.0,
                    "steps": 3,
                    "final_url": "http://example.test/done",
                }
            ],
        }

    monkeypatch.setattr(harvester_module, "run_guided_brief", fake_run_guided_brief)
    rows = harvester_module.collect_rows_from_guided_brief(
        use_case="CONTACT",
        seeds=[7],
        brief_payload={"meta": {"model": "claude-sonnet-4-5"}},
        task_cache=tmp_path / "tasks.json",
        output_root=tmp_path,
        attempt_name="guided",
        teacher_brief_path=str(tmp_path / "brief.json"),
    )
    assert len(rows) == 1
    assert rows[0]["episode_task_id"] == "ep-7"
    assert rows[0]["teacher_model"] == "claude-sonnet-4-5"
    assert rows[0]["teacher_brief_path"].endswith("brief.json")


def test_build_guided_row_prefers_episode_seed_over_requested_seed(tmp_path: Path) -> None:
    row = harvester_module.build_guided_row(
        use_case="CONTACT",
        seed=96,
        attempt_name="guided",
        report={
            "model": "claude-guided",
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
        out_path=tmp_path / "gold" / "runs" / "seed_0096_guided.json",
    )

    assert row is not None
    assert row["seed"] == 418
    assert "seed=418" in row["notes"]


def test_build_candidate_from_brief_uses_guided_actions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        harvester_module,
        "_guided_actions_from_brief",
        lambda *, task_url, brief, web_project_id="autocinema": [{"type": "NavigateAction", "url": task_url.replace("/?seed=", "/contact?seed=")}],
    )
    candidate = harvester_module.build_candidate_from_brief(
        use_case="CONTACT",
        seed=3,
        attempt_name="claude_01",
        brief_payload={"brief": {"route": ["/contact"]}, "meta": {"model": "claude-sonnet-4-5"}},
        brief_path=tmp_path / "brief.json",
        task_url="http://84.247.180.192:8000/?seed=3",
    )
    assert candidate.use_case == "CONTACT"
    assert candidate.teacher_model == "claude-sonnet-4-5"
    assert candidate.actions[0]["type"] == "NavigateAction"


def test_replay_candidate_uses_guided_runner(monkeypatch, tmp_path: Path) -> None:
    candidate = harvester_module.TrajectoryCandidate(
        use_case="CONTACT",
        seed=9,
        attempt_name="claude_01",
        teacher_model="claude-sonnet-4-5",
        generation_mode="claude_code_brief",
        brief_path=str(tmp_path / "brief.json"),
        prompt_lines=(),
        actions=({"type": "NavigateAction", "url": "http://example.test/contact?seed=9"},),
        metadata={},
    )

    def fake_run_guided_brief(*, use_case, seed, brief_payload, task_cache, web_project_id=None, max_steps, planned_actions_override, headless=None):
        assert planned_actions_override[0]["type"] == "NavigateAction"
        assert headless is True
        return {
            "model": "claude-sonnet-4-5",
            "episodes": [
                {
                    "task_id": "task-9",
                    "episode_task_id": "ep-9",
                    "success": True,
                    "score": 1.0,
                    "steps": 1,
                    "final_url": "http://example.test/contact?done=1",
                }
            ],
        }

    monkeypatch.setattr(harvester_module, "run_guided_brief", fake_run_guided_brief)
    report, row, result_path = harvester_module.replay_candidate(
        candidate=candidate,
        output_root=tmp_path,
        task_cache=tmp_path / "tasks.json",
        max_steps=12,
    )
    assert result_path.exists()
    assert report["episodes"][0]["success"] is True
    assert row is not None
    assert row["harvest_mode"] == "candidate_replay"


def test_replay_candidate_supports_headed_browser(monkeypatch, tmp_path: Path) -> None:
    candidate = harvester_module.TrajectoryCandidate(
        use_case="CONTACT",
        seed=5,
        attempt_name="deterministic_01",
        teacher_model="",
        generation_mode="deterministic_plan",
        brief_path=str(tmp_path / "brief.json"),
        prompt_lines=(),
        actions=({"type": "NavigateAction", "url": "http://example.test/contact?seed=5"},),
        metadata={},
    )

    seen: dict[str, object] = {}

    def fake_run_guided_brief(*, use_case, seed, brief_payload, task_cache, web_project_id=None, max_steps, planned_actions_override, headless=None):
        seen["headless"] = headless
        return {
            "model": "deterministic",
            "episodes": [
                {
                    "task_id": "task-5",
                    "episode_task_id": "ep-5",
                    "success": True,
                    "score": 1.0,
                    "steps": 1,
                    "final_url": "http://example.test/contact?done=1",
                }
            ],
        }

    monkeypatch.setattr(harvester_module, "run_guided_brief", fake_run_guided_brief)
    harvester_module.replay_candidate(
        candidate=candidate,
        output_root=tmp_path,
        task_cache=tmp_path / "tasks.json",
        max_steps=12,
        headed=True,
    )
    assert seen["headless"] is False
