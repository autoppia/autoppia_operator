from __future__ import annotations

import json
from pathlib import Path

from training.format_for_sft import BASE_MODEL, export_harvest_to_sft


def test_export_harvest_to_sft_generates_manifest_and_non_empty_split(tmp_path: Path) -> None:
    manifest = export_harvest_to_sft(
        input_path="data/autocinema_trajectory_harvest/episodes.jsonl",
        summary_path="data/autocinema_trajectory_harvest/summary.json",
        output_dir=str(tmp_path / "sft"),
        val_ratio=0.1,
        seed=36,
    )

    train_path = tmp_path / "sft" / "train.jsonl"
    val_path = tmp_path / "sft" / "val.jsonl"
    manifest_path = tmp_path / "sft" / "manifest.json"

    assert train_path.exists()
    assert val_path.exists()
    assert manifest_path.exists()

    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_rows = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert train_rows
    assert val_rows
    assert train_rows[0]["messages"][0]["role"] == "system"
    assert train_rows[0]["messages"][-1]["role"] == "assistant"
    assert manifest["base_model"] == BASE_MODEL
    assert manifest["train_examples"] == len(train_rows)
    assert manifest["val_examples"] == len(val_rows)
    assert manifest["successful_episodes_used"] == len(manifest["selected_episode_ids"])
