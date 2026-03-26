from __future__ import annotations

import json
import os
from pathlib import Path


REPO = Path(os.environ["ARBOS_TARGET_REPO"]).resolve()


def test_committed_sft_artifacts_exist_and_are_non_empty() -> None:
    base = REPO / "data" / "autocinema_trajectory_harvest" / "sft"
    train_path = base / "train.jsonl"
    val_path = base / "val.jsonl"
    manifest_path = base / "manifest.json"

    assert train_path.exists()
    assert val_path.exists()
    assert manifest_path.exists()

    train_count = sum(1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip())
    val_count = sum(1 for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip())

    assert train_count > 0
    assert val_count > 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["base_model"] == "browser-use/bu-30b-a3b-preview"
    assert manifest["train_examples"] == train_count
    assert manifest["val_examples"] == val_count
    assert manifest["source_episodes_path"].endswith("data/autocinema_trajectory_harvest/episodes.jsonl")


def test_runpod_plan_and_runbook_are_concrete() -> None:
    runbook_path = REPO / "docs" / "browser_use_finetune_runpod.md"
    plan_path = REPO / "training" / "runpod_bootstrap_plan.json"

    assert runbook_path.exists()
    assert plan_path.exists()

    runbook = runbook_path.read_text(encoding="utf-8")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert "browser-use/bu-30b-a3b-preview" in runbook
    assert "NVIDIA A100 80GB PCIe" in runbook
    assert "59dd4g1snevkqs" in runbook
    assert "ssh" in runbook.lower()
    assert "training.finetune_bu" in runbook

    assert plan["base_model"] == "browser-use/bu-30b-a3b-preview"
    assert plan["training_method"] in {"lora", "qlora", "lora_qlora"}
    assert plan["preferred_gpu"] == "NVIDIA A100 80GB PCIe"
    assert plan["existing_bootstrap_pod_id"] == "59dd4g1snevkqs"
    assert plan["sft_dataset"]["train_path"].endswith("data/autocinema_trajectory_harvest/sft/train.jsonl")
    assert plan["sft_dataset"]["val_path"].endswith("data/autocinema_trajectory_harvest/sft/val.jsonl")


def test_model_and_runpod_defaults_match_recommended_path() -> None:
    finetune_path = REPO / "training" / "finetune_bu.py"
    serve_path = REPO / "training" / "serve_model.py"
    runpod_config_path = REPO / "training" / "runpod_config.py"

    finetune_text = finetune_path.read_text(encoding="utf-8")
    serve_text = serve_path.read_text(encoding="utf-8")
    runpod_text = runpod_config_path.read_text(encoding="utf-8")

    assert 'browser-use/bu-30b-a3b-preview' in finetune_text
    assert 'browser-use/bu-30b-a3b-preview' in serve_text
    assert 'NVIDIA A100 80GB PCIe' in runpod_text
