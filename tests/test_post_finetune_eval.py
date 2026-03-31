from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from training.post_finetune_eval import _build_summary, _validate_real_success

REPO = Path(__file__).resolve().parents[1]


def test_post_finetune_eval_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "training.post_finetune_eval", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage" in result.stdout.lower()


def test_post_finetune_eval_refuses_stub_adapter() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir)
        (adapter_dir / "adapter_config.json").write_text(
            '{"base_model_name_or_path":"browser-use/bu-30b-a3b-preview","stub":true}',
            encoding="utf-8",
        )
        (adapter_dir / "adapter_model.safetensors").write_text('{"stub":true}', encoding="utf-8")
        (adapter_dir / "train_metrics.json").write_text(
            '{"base_model":"browser-use/bu-30b-a3b-preview","stub":true}',
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "training.post_finetune_eval",
                "--adapter-path",
                str(adapter_dir),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode != 0
    combined = f"{result.stdout}\n{result.stderr}".lower()
    assert "stub" in combined or "small" in combined


def test_build_summary_records_successful_episodes_and_gate() -> None:
    summary = _build_summary(
        endpoint="http://127.0.0.1:8012/v1",
        adapter_path=REPO / "models" / "bu-30b-lora",
        raw_report={
            "num_tasks": 2,
            "successes": 1,
            "avg_score": 0.81,
            "episodes": [
                {"task_id": "task-1", "use_case": "LOGIN", "success": True},
                {"task_id": "task-2", "use_case": "SEARCH", "success": False},
            ],
        },
        provider="openai",
        model="autoppia",
        raw_report_path=REPO / "data" / "autocinema_trajectory_harvest" / "post_finetune_eval.json",
        success_threshold=0.4,
        avg_score_threshold=0.6,
    )

    assert summary["successful_episode_ids"] == ["task-1"]
    assert summary["use_cases_evaluated"] == ["LOGIN", "SEARCH"]
    assert summary["passed_gate"] is True


def test_validate_real_success_requires_successful_episode() -> None:
    with pytest.raises(RuntimeError, match="successful Autocinema episode"):
        _validate_real_success(
            {
                "successful_episode_ids": [],
            }
        )
