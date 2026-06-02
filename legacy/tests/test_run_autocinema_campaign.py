from __future__ import annotations

import json
from pathlib import Path

import scripts.eval.run_autocinema_campaign as campaign


def test_merge_sft_manifests_and_plan(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    monkeypatch.setattr(campaign, "REPO_ROOT", repo_root)

    source_a = repo_root / "data" / "autocinema" / "login" / "sft_campaign"
    source_b = repo_root / "data" / "autocinema" / "contact" / "sft_campaign"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)

    def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    for source, use_case in ((source_a, "LOGIN"), (source_b, "CONTACT")):
        write_jsonl(source / "train.jsonl", [{"messages": [{"role": "user", "content": use_case}]}])
        write_jsonl(source / "val.jsonl", [{"messages": [{"role": "assistant", "content": use_case}]}])
        (source / "manifest.json").write_text(
            json.dumps(
                {
                    "manifest_path": str(source / "manifest.json"),
                    "train_path": str(source / "train.jsonl"),
                    "val_path": str(source / "val.jsonl"),
                    "system_prompt": "SYS",
                    "episodes_total": 5,
                    "runtime_aligned": True,
                    "source_mode": "trace_only",
                }
            ),
            encoding="utf-8",
        )

    prepared = [
        {
            "use_case": "LOGIN",
            "sft_dir": str(source_a),
            "sft_manifest": json.loads((source_a / "manifest.json").read_text(encoding="utf-8")),
        },
        {
            "use_case": "CONTACT",
            "sft_dir": str(source_b),
            "sft_manifest": json.loads((source_b / "manifest.json").read_text(encoding="utf-8")),
        },
    ]

    merged = campaign._merge_sft_manifests(run_name="demo_run", prepared=prepared, base_model="Qwen/Qwen3-30B-A3B")
    assert merged["use_cases"] == ["LOGIN", "CONTACT"]
    assert merged["train_examples"] == 2
    assert merged["val_examples"] == 2

    monkeypatch.setattr(campaign, "build_runpod_job_command", lambda **_: ["python", "-m", "training.runpod_job"])
    plan = campaign._write_campaign_plan(
        run_name="demo_run",
        merged_manifest=merged,
        use_cases=["LOGIN", "CONTACT"],
        existing_pod_id="pod-123",
        epochs=2,
        lora_rank=32,
        all_use_cases_eval_tasks=3,
    )
    assert plan["use_cases"] == ["LOGIN", "CONTACT"]
    assert plan["train_command"] == ["python", "-m", "training.runpod_job"]
    assert "--all-use-cases" in plan["eval_command"]
