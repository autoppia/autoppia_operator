from __future__ import annotations

import json
from pathlib import Path

from training.focus_cost_analytics import build_focus_cost_report


def test_build_focus_cost_report_aggregates_by_seed_and_model(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "seed": 1,
            "success": True,
            "score": 1.0,
            "model": "gpt-5.4-mini",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "estimated_cost_usd": 0.01,
        },
        {
            "seed": 1,
            "success": False,
            "score": 0.0,
            "model": "gpt-5.4",
            "prompt_tokens": 200,
            "completion_tokens": 40,
            "total_tokens": 240,
            "estimated_cost_usd": 0.03,
        },
    ]
    (gold / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = build_focus_cost_report(tmp_path)
    assert report["totals"]["attempts"] == 2
    assert report["totals"]["total_tokens"] == 360
    assert report["totals"]["estimated_cost_usd"] == 0.04
    assert report["by_seed"][0]["seed"] == 1
    assert report["by_seed"][0]["attempts"] == 2
    assert report["by_seed"][0]["successes"] == 1
    models = {row["model"]: row for row in report["by_model"]}
    assert models["gpt-5.4-mini"]["total_tokens"] == 120
    assert models["gpt-5.4"]["estimated_cost_usd"] == 0.03
