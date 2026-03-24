from __future__ import annotations

import json
from pathlib import Path

from training.exporters import (
    export_ppo_bootstrap,
    export_sft,
    load_cleaned_trajectories,
)
from training.models import TrajectoryRecord


def _record() -> TrajectoryRecord:
    return TrajectoryRecord.from_dict(
        {
            "trajectory_id": "r1:t1:abc",
            "run_id": "r1",
            "task_id": "t1",
            "source_url": "s3://b/k",
            "task": {
                "prompt": "Do thing",
                "url": "https://example.com",
                "website": "example",
                "use_case": {"name": "FLOW"},
            },
            "summary": {
                "status": "success",
                "success": True,
                "eval_score": 1.0,
                "reward": 1.0,
                "eval_time_sec": 0.3,
                "steps_total": 1,
                "steps_success": 1,
            },
            "actions": [
                {
                    "type": "ClickAction",
                    "selector": {
                        "type": "attributeValueSelector",
                        "attribute": "id",
                        "value": "go",
                    },
                }
            ],
            "steps": [
                {
                    "step_index": 0,
                    "success": True,
                    "agent_input": {
                        "prompt": "Do thing",
                        "current_url": "https://example.com",
                    },
                    "post_execute_output": {"current_url": "https://example.com/next"},
                    "llm_calls": [],
                    "agent_output": {"action": {"type": "ClickAction"}},
                }
            ],
        }
    )


def test_exporters_roundtrip(tmp_path: Path) -> None:
    records = [_record()]
    cleaned = tmp_path / "cleaned.jsonl"
    with cleaned.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    loaded = load_cleaned_trajectories(cleaned)
    assert len(loaded) == 1

    sft_out = export_sft(
        records=loaded,
        out_dir=tmp_path / "sft",
        val_ratio=0.2,
        seed=42,
        system_prompt="sys",
    )
    assert Path(sft_out["sft_all"]).exists()
    assert Path(sft_out["sft_train"]).exists()
    assert Path(sft_out["sft_val"]).exists()

    ppo_out = export_ppo_bootstrap(records=loaded, out_path=tmp_path / "ppo" / "ppo_bootstrap.jsonl")
    assert Path(ppo_out["ppo_bootstrap"]).exists()
    assert ppo_out["stats"]["ppo_rows"] > 0
