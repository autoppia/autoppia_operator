from __future__ import annotations

import gzip
import json
from pathlib import Path

from training.models import TrajectoryRecord
from training.pipeline import (
    TrajectoryBuildConfig,
    export_training_bundle,
    ingest_from_s3,
)
from training.s3_source import S3ObjectRef, decode_json_blob


def _sample_normalized() -> dict:
    return {
        "trajectory_id": "run-1:task-1:abc",
        "run_id": "run-1",
        "task_id": "task-1",
        "source_url": "s3://bucket/logs/a.json.gz",
        "task": {
            "prompt": "Login and open dashboard",
            "url": "https://example.com/login",
            "website": "example",
            "use_case": {"name": "LOGIN"},
        },
        "summary": {
            "status": "success",
            "success": True,
            "eval_score": 1.0,
            "reward": 1.0,
            "eval_time_sec": 2.3,
            "steps_total": 2,
            "steps_success": 2,
        },
        "actions": [
            {"type": "TypeAction", "text": "user@example.com"},
            {
                "type": "ClickAction",
                "selector": {
                    "type": "attributeValueSelector",
                    "attribute": "id",
                    "value": "submit",
                },
            },
        ],
        "steps": [
            {
                "step_index": 0,
                "success": True,
                "agent_input": {
                    "prompt": "Login",
                    "current_url": "https://example.com/login",
                },
                "post_execute_output": {"current_url": "https://example.com/login"},
                "llm_calls": [],
                "agent_output": {"action": {"type": "TypeAction", "text": "user@example.com"}},
            },
            {
                "step_index": 1,
                "success": True,
                "agent_input": {
                    "prompt": "Login",
                    "current_url": "https://example.com/login",
                },
                "post_execute_output": {"current_url": "https://example.com/dashboard"},
                "llm_calls": [],
                "agent_output": {"action": {"type": "ClickAction"}},
            },
        ],
    }


def test_trajectory_record_roundtrip_and_sft() -> None:
    record = TrajectoryRecord.from_dict(_sample_normalized())
    dumped = record.to_dict()

    assert dumped["trajectory_id"] == "run-1:task-1:abc"
    assert len(dumped["actions"]) == 2
    assert dumped["summary"]["success"] is True

    sft = record.to_sft_record(system_prompt="You are an agent")
    assert isinstance(sft.get("messages"), list)
    assert len(sft["messages"]) == 3
    assert sft["messages"][0]["role"] == "system"


def test_decode_json_blob_plain_and_gzip() -> None:
    payload = {"task_id": "t1", "steps": []}

    plain = json.dumps(payload).encode("utf-8")
    parsed_plain = decode_json_blob(plain, key_hint="x.json")
    assert parsed_plain["task_id"] == "t1"

    gz = gzip.compress(plain)
    parsed_gz = decode_json_blob(gz, key_hint="x.json.gz")
    assert parsed_gz["task_id"] == "t1"


class _FakeS3Source:
    def __init__(self, payloads: list[dict]):
        self._payloads = payloads

    def iter_objects(self, *, max_objects=None, suffixes=(".json", ".json.gz", ".gz")):
        refs = [S3ObjectRef(bucket="fake", key=f"logs/{i}.json") for i in range(len(self._payloads))]
        if max_objects is not None:
            refs = refs[: int(max_objects)]
        yield from refs

    def fetch_json(self, ref: S3ObjectRef):
        idx = int(Path(ref.key).stem)
        return self._payloads[idx]


def test_ingest_from_s3_and_export_bundle(tmp_path: Path) -> None:
    raw_payload = {
        "run_id": "run-42",
        "payload": {
            "task_id": "task-42",
            "task": {
                "prompt": "Open profile",
                "url": "https://example.com/profile",
                "website": "example",
                "use_case": {"name": "PROFILE"},
            },
            "summary": {
                "status": "success",
                "eval_score": 1.0,
                "steps_total": 1,
                "steps_success": 1,
            },
            "steps": [
                {
                    "step_index": 0,
                    "success": True,
                    "agent_input": {
                        "prompt": "Open profile",
                        "current_url": "https://example.com",
                    },
                    "agent_output": {
                        "action": {
                            "type": "ClickAction",
                            "selector": {
                                "type": "attributeValueSelector",
                                "attribute": "id",
                                "value": "profile",
                            },
                        }
                    },
                    "post_execute_output": {"current_url": "https://example.com/profile"},
                }
            ],
        },
    }

    records, stats = ingest_from_s3(
        source=_FakeS3Source([raw_payload]),
        cfg=TrajectoryBuildConfig(min_actions=1, only_successful=True),
    )

    assert len(records) == 1
    assert stats.trajectories_kept == 1

    artifacts = export_training_bundle(
        out_dir=tmp_path / "dataset",
        records=records,
        stats=stats,
        val_ratio=0.2,
        seed=42,
        system_prompt="You are an agent",
    )

    assert Path(artifacts.cleaned_trajectories).exists()
    assert Path(artifacts.sft_train).exists()
    assert Path(artifacts.ppo_bootstrap).exists()
