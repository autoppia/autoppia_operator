from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_harvest_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "scripts" / "autocinema_harvest.py"
    spec = importlib.util.spec_from_file_location("autocinema_harvest", mod_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _write_eval_result(path: Path, *, use_cases: list[str]) -> None:
    episodes = []
    seed_base = 2000
    for idx, use_case in enumerate(use_cases):
        for j in range(2):
            seed = seed_base + idx * 10 + j
            success = bool(j % 2 == 0)
            score = 1.0 if success else 0.0
            episodes.append(
                {
                    "task_id": f"task-{idx}",
                    "episode_task_id": f"task-{idx}-{seed}-{j}",
                    "use_case": use_case,
                    "seed": seed,
                    "success": success,
                    "score": score,
                }
            )
    payload = {
        "provider": "openai",
        "model": "gpt-5.2",
        "meta": {"repeat": 2, "seed_start": seed_base, "max_steps": 12},
        "episodes": episodes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_trace_root(trace_root: Path, *, use_cases: list[str]) -> None:
    episodes_dir = trace_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seed_base = 2000
    for idx, use_case in enumerate(use_cases):
        for j in range(2):
            seed = seed_base + idx * 10 + j
            episode_task_id = f"task-{idx}-{seed}-{j}"
            episode_path = episodes_dir / f"{episode_task_id}.json"
            episode_path.write_text(
                json.dumps(
                    {
                        "episode": {
                            "episode_task_id": episode_task_id,
                            "use_case": use_case,
                            "seed": seed,
                            "score": 1.0 if j % 2 == 0 else 0.0,
                            "success": bool(j % 2 == 0),
                        },
                        "steps": [{"step_index": 0, "actions": []}],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            rows.append(
                {
                    "episode_task_id": episode_task_id,
                    "task_id": f"task-{idx}",
                    "use_case": use_case,
                    "success": bool(j % 2 == 0),
                    "score": 1.0 if j % 2 == 0 else 0.0,
                    "file": f"episodes/{episode_task_id}.json",
                    "final_url": f"https://example.com/{use_case.lower()}/{seed}",
                }
            )
    (trace_root / "trace_index.json").write_text(json.dumps({"episodes": rows}, indent=2), encoding="utf-8")


def test_build_harvest_artifacts_has_required_contract(tmp_path: Path):
    harvest = _load_harvest_module()
    result_path = tmp_path / "eval_autocinema.json"
    trace_root = tmp_path / "traces_eval_autocinema"
    _write_eval_result(result_path, use_cases=list(harvest.AUTOCINEMA_USE_CASES))
    _write_trace_root(trace_root, use_cases=list(harvest.AUTOCINEMA_USE_CASES))

    artifacts = harvest.build_harvest_artifacts(
        project_id="autocinema",
        result_paths=[result_path],
        trace_roots=[trace_root],
        near_miss_threshold=0.5,
        branch="daryxx",
        iwa_branch="daryxx",
        require_trace_files=True,
        command_sources=[],
    )

    summary = artifacts.summary
    assert summary["project_id"] == "autocinema"
    assert summary["branch"] == "daryxx"
    assert summary["iwa_branch"] == "daryxx"
    assert set(summary["use_cases"]) == set(harvest.AUTOCINEMA_USE_CASES)
    assert summary["episodes_total"] == len(harvest.AUTOCINEMA_USE_CASES) * 2
    assert summary["successes_total"] > 0
    assert summary["failures_total"] > 0

    for use_case in harvest.AUTOCINEMA_USE_CASES:
        uc_stats = summary["per_use_case"][use_case]
        assert uc_stats["attempted"] == 2
        assert uc_stats["successes"] == 1
        assert uc_stats["failures"] == 1
        assert isinstance(uc_stats["trace_files"], list)
        assert uc_stats["trace_files"]
        assert len(uc_stats["distinct_seeds"]) == 2

    assert len(artifacts.episodes) == summary["episodes_total"]
    required_episode_keys = {
        "web_project_id",
        "use_case",
        "seed",
        "task_id",
        "success",
        "score",
        "trace_ref",
        "trace_file",
        "result_path",
    }
    assert required_episode_keys.issubset(set(artifacts.episodes[0].keys()))
    assert artifacts.episodes[0]["trace_file"]
    assert summary["replayable_episodes_total"] == summary["episodes_total"]
    assert "golden_by_use_case" in artifacts.golden_seeds
    assert "run_sources" in artifacts.manifest


def test_main_writes_summary_and_episodes(tmp_path: Path):
    harvest = _load_harvest_module()
    result_path = tmp_path / "eval_autocinema.json"
    trace_root = tmp_path / "traces_eval_autocinema"
    _write_eval_result(result_path, use_cases=list(harvest.AUTOCINEMA_USE_CASES))
    _write_trace_root(trace_root, use_cases=list(harvest.AUTOCINEMA_USE_CASES))
    out_dir = tmp_path / "harvest_out"

    rc = harvest.main(
        [
            "--project-id",
            "autocinema",
            "--result-glob",
            str(result_path),
            "--trace-root",
            str(trace_root),
            "--out-dir",
            str(out_dir),
            "--iwa-repo",
            str(tmp_path),
        ]
    )
    assert rc == 0

    summary_path = out_dir / "summary.json"
    episodes_path = out_dir / "episodes.jsonl"
    manifest_path = out_dir / "collection_manifest.json"
    golden_path = out_dir / "golden_seeds.json"
    assert summary_path.exists()
    assert episodes_path.exists()
    assert manifest_path.exists()
    assert golden_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["episodes_total"] == len(harvest.AUTOCINEMA_USE_CASES) * 2
    lines = [line for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == summary["episodes_total"]
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert "LOGIN" in golden["golden_by_use_case"]


def test_explicit_trace_roots_are_matched_to_result_file(tmp_path: Path):
    harvest = _load_harvest_module()
    use_cases = ["ADD_FILM", "LOGIN"]

    result_a = tmp_path / "eval_autocinema_add_film_run_a.json"
    trace_a = tmp_path / "traces_eval_autocinema_add_film_run_a"
    _write_eval_result(result_a, use_cases=use_cases)
    _write_trace_root(trace_a, use_cases=use_cases)

    result_b = tmp_path / "eval_autocinema_add_film_run_b.json"
    trace_b = tmp_path / "traces_eval_autocinema_add_film_run_b"
    _write_eval_result(result_b, use_cases=use_cases)
    _write_trace_root(trace_b, use_cases=use_cases)

    artifacts = harvest.build_harvest_artifacts(
        project_id="autocinema",
        result_paths=[result_a, result_b],
        trace_roots=[trace_b, trace_a],  # reversed on purpose
        near_miss_threshold=0.5,
        branch="daryxx",
        iwa_branch="daryxx",
        require_trace_files=True,
        command_sources=[],
    )

    assert artifacts.summary["episodes_total"] == 8
    assert artifacts.summary["replayable_episodes_total"] == 8
    for ep in artifacts.episodes:
        assert ep["trace_file"]
        result_name = Path(ep["result_path"]).name
        trace_root = Path(ep["trace_root"]).name
        if result_name.endswith("run_a.json"):
            assert trace_root.endswith("run_a")
        if result_name.endswith("run_b.json"):
            assert trace_root.endswith("run_b")


def test_trace_roots_can_be_matched_by_episode_overlap(tmp_path: Path):
    harvest = _load_harvest_module()
    use_cases = ["ADD_FILM", "LOGIN"]

    result_path = tmp_path / "eval_autocinema_alias_name.json"
    trace_root = tmp_path / "totally_different_trace_dir"
    _write_eval_result(result_path, use_cases=use_cases)
    _write_trace_root(trace_root, use_cases=use_cases)

    artifacts = harvest.build_harvest_artifacts(
        project_id="autocinema",
        result_paths=[result_path],
        trace_roots=[trace_root],
        near_miss_threshold=0.5,
        branch="daryxx",
        iwa_branch="daryxx",
        require_trace_files=True,
        command_sources=[],
    )

    assert artifacts.summary["episodes_total"] == 4
    assert artifacts.summary["replayable_episodes_total"] == 4
    assert {Path(ep["trace_root"]).name for ep in artifacts.episodes} == {"totally_different_trace_dir"}
