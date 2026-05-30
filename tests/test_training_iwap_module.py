from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.normalize import (
    dedupe_trajectories,
    extract_task_payload,
    normalize_trajectory,
)


def test_extract_task_payload_accepts_wrapped_and_plain() -> None:
    wrapped = {
        "task_id": "t-1",
        "payload": {"schema_version": "1.0", "task_id": "t-1", "steps": []},
    }
    plain = {"schema_version": "1.0", "task_id": "t-2", "steps": []}

    got_wrapped = extract_task_payload(wrapped)
    got_plain = extract_task_payload(plain)

    assert got_wrapped is not None
    assert got_wrapped["task_id"] == "t-1"
    assert got_plain is not None
    assert got_plain["task_id"] == "t-2"


def test_normalize_trajectory_strips_html_and_extracts_actions() -> None:
    payload = {
        "schema_version": "1.0",
        "task_id": "task-1",
        "task": {
            "prompt": "Login and open dashboard",
            "url": "https://example.com/login",
            "website": "example",
            "use_case": {"name": "LOGIN"},
        },
        "summary": {
            "status": "success",
            "eval_score": 1.0,
            "reward": 1.0,
            "eval_time_sec": 2.4,
            "steps_total": 2,
            "steps_success": 2,
        },
        "steps": [
            {
                "step_index": 0,
                "timestamp": "2026-03-01T10:00:00Z",
                "agent_input": {
                    "current_url": "https://example.com/login",
                    "html": "<html>...</html>",
                    "prompt": "Login",
                },
                "agent_output": {
                    "action": {
                        "type": "TypeAction",
                        "attributes": {
                            "selector": {"value": "#email"},
                            "value": "user@example.com",
                        },
                    }
                },
                "post_execute_output": {
                    "current_url": "https://example.com/login",
                    "html": "<html>after...</html>",
                },
                "success": True,
                "execution_time_ms": 500,
            },
            {
                "step_index": 1,
                "agent_output": {"action": {"type": "ClickAction", "selector": {"value": "#submit"}}},
                "success": True,
            },
        ],
    }

    trajectory = normalize_trajectory(
        payload=payload,
        run_id="run-1",
        source_url="https://bucket/task-log.json.gz",
        task_meta={"taskId": "task-1", "status": "completed", "eval_score": 1.0},
        keep_html=False,
        min_eval_score=0.5,
        max_steps=30,
        max_actions=20,
        dedupe_actions=True,
    )

    assert trajectory["summary"]["success"] is True
    assert len(trajectory["actions"]) == 2
    first_step = trajectory["steps"][0]
    assert "html" not in first_step["agent_input"]
    assert "html" not in first_step["post_execute_output"]


def test_dedupe_trajectories_keeps_highest_score_variant() -> None:
    base_actions = [{"type": "ClickAction", "selector": {"value": "#submit"}}]
    low = {
        "task": {"prompt": "p", "url": "u"},
        "actions": base_actions,
        "summary": {"eval_score": 0.2},
    }
    high = {
        "task": {"prompt": "p", "url": "u"},
        "actions": base_actions,
        "summary": {"eval_score": 1.0},
    }

    out = dedupe_trajectories([low, high])
    assert len(out) == 1
    assert out[0]["summary"]["eval_score"] == 1.0
