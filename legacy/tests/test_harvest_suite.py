from __future__ import annotations

from pathlib import Path

import training.harvest_suite as suite_module
from training.harvest_suite import HarvestSuiteConfig, collect_suite, parse_use_case_spec


def test_parse_use_case_spec_all_uses_registry() -> None:
    use_cases = parse_use_case_spec("all")
    assert "LOGIN" in use_cases
    assert "CONTACT" in use_cases
    assert len(use_cases) >= 10


def test_parse_use_case_spec_dedupes_and_normalizes() -> None:
    use_cases = parse_use_case_spec(" login,CONTACT,login ")
    assert use_cases == ["LOGIN", "CONTACT"]


def test_collect_suite_writes_per_use_case_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(suite_module, "_focus_root", lambda *, use_case: tmp_path / use_case.lower())

    def fake_collect_rows_for_seeds(*, config, seeds, strategy, collect_workers):
        return [
            {
                "seed": int(seeds[0]),
                "attempt_name": "baseline",
                "success": True,
                "score": 1.0,
            }
        ]

    def fake_write_harvest_artifacts(*, output_root, use_case, target_seeds, rows, merge_existing):
        return (
            output_root / "gold" / "episodes.jsonl",
            output_root / "gold" / "summary.json",
            {
                "gold_episodes_total": len(rows),
                "attempts_total": len(rows),
            },
        )

    monkeypatch.setattr(suite_module, "collect_rows_for_seeds", fake_collect_rows_for_seeds)
    monkeypatch.setattr(suite_module, "write_harvest_artifacts", fake_write_harvest_artifacts)
    config = HarvestSuiteConfig(
        use_cases=("LOGIN", "CONTACT"),
        seeds=(1, 2),
        provider="openai",
        model="gpt-5.4-mini",
        task_cache_arg=str(tmp_path / "tasks.json"),
        max_seeds_per_use_case=1,
    )
    result = collect_suite(config)

    assert result["use_case_count"] == 2
    assert result["attempt_rows_added_total"] == 2
    assert len(result["per_use_case"]) == 2
    assert {row["use_case"] for row in result["per_use_case"]} == {"LOGIN", "CONTACT"}


def test_collect_suite_skips_use_case_when_gold_target_already_met(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "login"
    gold_dir = output_root / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    (gold_dir / "episodes.jsonl").write_text('{"seed":1,"success":true,"score":1.0}\n', encoding="utf-8")
    monkeypatch.setattr(suite_module, "_focus_root", lambda *, use_case: output_root)

    called: dict[str, bool] = {"collect": False}

    def fake_collect_rows_for_seeds(*, config, seeds, strategy, collect_workers):
        called["collect"] = True
        return []

    monkeypatch.setattr(suite_module, "collect_rows_for_seeds", fake_collect_rows_for_seeds)
    config = HarvestSuiteConfig(
        use_cases=("LOGIN",),
        seeds=(1, 2),
        provider="openai",
        model="gpt-5.4-mini",
        task_cache_arg=str(tmp_path / "tasks.json"),
        target_gold_per_use_case=1,
    )
    result = collect_suite(config)

    assert called["collect"] is False
    assert result["per_use_case"][0]["status"] == "target_already_met"
