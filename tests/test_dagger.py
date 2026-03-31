from __future__ import annotations

import json
from pathlib import Path

import training.dagger as dagger_module
from training.login_dagger import login_dagger_extra_lines
from training.use_case_registry import dagger_extra_lines


def test_login_dagger_extra_lines_now_comes_from_registry() -> None:
    assert login_dagger_extra_lines(failure_category="NO_PROGRESS_LOOP") == dagger_extra_lines(
        use_case="LOGIN",
        failure_category="NO_PROGRESS_LOOP",
    )


def test_run_dagger_round_uses_generic_use_case_and_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "results": [
                    {"seed": 7, "success": False, "failure_category": "NO_PROGRESS_LOOP"},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(dagger_module, "focus_root", lambda *, use_case: tmp_path / use_case.lower())

    built: dict[str, object] = {}

    def fake_build_task_cache_override(*, source_task_cache, use_case, prompt_override, out_path):
        built["use_case"] = use_case
        built["prompt_override"] = prompt_override
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}", encoding="utf-8")
        return out_path

    class FakeResult:
        is_gold = True
        out_path = tmp_path / "run.json"
        trace_dir = tmp_path / "trace-dir"
        row = {"trace_file": str(tmp_path / "trace.json"), "episode_task_id": "ep-7"}

    monkeypatch.setattr(dagger_module, "build_task_cache_override", fake_build_task_cache_override)
    monkeypatch.setattr(dagger_module, "run_eval_attempt", lambda **kwargs: FakeResult())

    result = dagger_module.run_dagger_round(
        use_case="CONTACT",
        source_summary_path=source,
        split_name="holdout",
        teacher_provider="openai",
        teacher_model="gpt-5.4",
    )

    assert result["use_case"] == "CONTACT"
    assert result["teacher_successes"] == 1
    assert built["use_case"] == "CONTACT"
    assert "contact" in str((tmp_path / "contact" / "dagger" / "holdout" / "summary.json")).lower()
