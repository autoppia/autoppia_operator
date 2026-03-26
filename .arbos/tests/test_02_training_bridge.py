from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(os.environ["ARBOS_TARGET_REPO"]).resolve()


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO) if not existing else f"{REPO}:{existing}"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_training_export_module_imports() -> None:
    sys.path.insert(0, str(REPO))
    mod = importlib.import_module("training.export")
    assert hasattr(mod, "export_sft")


def test_real_harvest_exports_non_empty_sft(tmp_path: Path) -> None:
    out_dir = tmp_path / "sft_probe"
    code = f"""
from training.format_for_sft import format_episodes_file
format_episodes_file('data/autocinema_trajectory_harvest/episodes.jsonl', r'{out_dir}', val_ratio=0.1)
"""
    _run_python(code)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    assert train_path.exists(), "train.jsonl missing"
    assert val_path.exists(), "val.jsonl missing"

    train_lines = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_lines = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert train_lines, "train.jsonl is empty"
    assert val_lines, "val.jsonl is empty"
    assert all("messages" in row for row in train_lines[:3])
    assert all(row["messages"][0]["role"] == "system" for row in train_lines[:3])
    assert all(row["messages"][-1]["role"] == "assistant" for row in train_lines[:3])


def test_training_clis_expose_help() -> None:
    for module in ("training.finetune_bu", "training.runpod_job", "training.serve_model"):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "usage" in result.stdout.lower()
