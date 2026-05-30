from __future__ import annotations

import gzip
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

from training.models import TrajectoryRecord
from training.pipeline import (
    TrajectoryBuildConfig,
    export_training_bundle,
    ingest_from_s3,
)
from training.s3_source import S3ObjectRef, S3TrajectorySource, decode_json_blob, parse_s3_uri


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


def test_parse_s3_uri_and_object_ref_uri() -> None:
    bucket, key = parse_s3_uri(" s3://bucket-name/path/to/file.json ")
    ref = S3ObjectRef(bucket=bucket, key=key)

    assert (bucket, key) == ("bucket-name", "path/to/file.json")
    assert ref.uri == "s3://bucket-name/path/to/file.json"


@pytest.mark.parametrize(
    ("raw_uri", "message"),
    [
        ("https://bucket/key.json", "Not an s3:// URI"),
        ("s3://bucket-only", "S3 URI missing key path"),
        ("s3:///missing-bucket.json", "Invalid S3 URI"),
        ("s3://bucket/", "Invalid S3 URI"),
    ],
)
def test_parse_s3_uri_rejects_invalid_values(raw_uri: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_s3_uri(raw_uri)


def test_decode_json_blob_rejects_invalid_payloads() -> None:
    with pytest.raises(ValueError, match="S3 object is empty"):
        decode_json_blob(b"")

    with pytest.raises(ValueError, match="S3 object is empty"):
        decode_json_blob("not-bytes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unable to parse JSON payload"):
        decode_json_blob(b"{broken")

    with pytest.raises(ValueError, match="Expected JSON object, got list"):
        decode_json_blob(json.dumps([1, 2, 3]).encode("utf-8"))


def test_decode_json_blob_tolerates_already_decompressed_gzip_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"task_id": "plain-under-gzip-hint"}).encode("utf-8")

    def _boom(_payload: bytes) -> bytes:
        raise OSError("already decompressed")

    monkeypatch.setattr(gzip, "decompress", _boom)

    parsed = decode_json_blob(payload, key_hint="logs/file.json.gz")

    assert parsed["task_id"] == "plain-under-gzip-hint"


class _FakeBody:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class _FakeS3Client:
    def __init__(self, pages: list[dict] | None = None, bodies: dict[str, object] | None = None):
        self.pages = list(pages or [])
        self.bodies = dict(bodies or {})
        self.list_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def list_objects_v2(self, **kwargs):
        self.list_calls.append(kwargs)
        if self.pages:
            return self.pages.pop(0)
        return {"Contents": [], "IsTruncated": False}

    def get_object(self, **kwargs):
        self.get_calls.append(kwargs)
        return {"Body": self.bodies.get(kwargs["Key"])}


def test_s3_source_client_import_error() -> None:
    source = S3TrajectorySource(bucket="demo")
    sys.modules.pop("boto3", None)

    with pytest.raises(ImportError, match="boto3 is required for S3 ingestion"):
        source._client()


def test_s3_source_client_builds_session_with_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    fake_client = object()

    class _SessionFactory:
        def __init__(self, profile_name=None):
            captured["profile_name"] = profile_name

        def client(self, service_name: str, *, region_name=None, endpoint_url=None):
            captured["service_name"] = service_name
            captured["region_name"] = region_name
            captured["endpoint_url"] = endpoint_url
            return fake_client

    fake_boto3 = types.SimpleNamespace(session=types.SimpleNamespace(Session=_SessionFactory))
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    source = S3TrajectorySource(
        bucket="demo-bucket",
        prefix="/logs",
        region_name="eu-west-1",
        profile_name="sandbox",
        endpoint_url="http://localhost:9000",
    )

    client = source._client()

    assert client is fake_client
    assert captured == {
        "profile_name": "sandbox",
        "service_name": "s3",
        "region_name": "eu-west-1",
        "endpoint_url": "http://localhost:9000",
    }
    assert source.prefix == "logs"


def test_iter_objects_handles_pagination_suffixes_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    first_page = {
        "Contents": [
            {"Key": "", "Size": 0},
            {"Key": "logs/ignore.txt", "Size": 10},
            {"Key": "logs/keep-1.json", "Size": 12, "ETag": '"etag-1"', "LastModified": datetime(2026, 1, 1)},
        ],
        "IsTruncated": True,
        "NextContinuationToken": "token-2",
    }
    second_page = {
        "Contents": [
            {"Key": "logs/keep-2.json.gz", "Size": 13, "ETag": '"etag-2"'},
            {"Key": "logs/keep-3.gz", "Size": "14"},
        ],
        "IsTruncated": False,
    }
    fake_client = _FakeS3Client(pages=[first_page, second_page])
    source = S3TrajectorySource(bucket="demo-bucket", prefix="logs")
    monkeypatch.setattr(source, "_client", lambda: fake_client)

    refs = list(source.iter_objects(max_objects=2))

    assert [ref.key for ref in refs] == ["logs/keep-1.json", "logs/keep-2.json.gz"]
    assert refs[0].size == 12
    assert refs[0].etag == "etag-1"
    assert refs[1].etag == "etag-2"
    assert fake_client.list_calls == [
        {"Bucket": "demo-bucket", "Prefix": "logs", "MaxKeys": 1000},
        {"Bucket": "demo-bucket", "Prefix": "logs", "MaxKeys": 1000, "ContinuationToken": "token-2"},
    ]


def test_iter_objects_stops_when_truncated_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeS3Client(pages=[{"Contents": [], "IsTruncated": True}])
    source = S3TrajectorySource(bucket="demo-bucket")
    monkeypatch.setattr(source, "_client", lambda: fake_client)

    assert list(source.iter_objects(suffixes=())) == []
    assert fake_client.list_calls == [{"Bucket": "demo-bucket", "Prefix": "", "MaxKeys": 1000}]


def test_iter_objects_returns_cleanly_when_not_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeS3Client(pages=[{"Contents": [], "IsTruncated": False}])
    source = S3TrajectorySource(bucket="demo-bucket")
    monkeypatch.setattr(source, "_client", lambda: fake_client)

    assert list(source.iter_objects()) == []


def test_fetch_object_bytes_and_json_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"task_id": "t-fetch"}).encode("utf-8")
    fake_client = _FakeS3Client(bodies={"logs/ok.json": _FakeBody(payload)})
    source = S3TrajectorySource(bucket="demo-bucket")
    monkeypatch.setattr(source, "_client", lambda: fake_client)
    ref = S3ObjectRef(bucket="demo-bucket", key="logs/ok.json")

    assert source.fetch_object_bytes(ref) == payload
    assert source.fetch_json(ref)["task_id"] == "t-fetch"
    assert fake_client.get_calls == [
        {"Bucket": "demo-bucket", "Key": "logs/ok.json"},
        {"Bucket": "demo-bucket", "Key": "logs/ok.json"},
    ]


def test_fetch_object_bytes_rejects_missing_or_non_bytes_body(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = S3ObjectRef(bucket="demo-bucket", key="logs/missing.json")
    source = S3TrajectorySource(bucket="demo-bucket")

    missing_body_client = _FakeS3Client(bodies={"logs/missing.json": None})
    monkeypatch.setattr(source, "_client", lambda: missing_body_client)
    with pytest.raises(ValueError, match="returned no Body"):
        source.fetch_object_bytes(ref)

    wrong_body_client = _FakeS3Client(bodies={"logs/missing.json": _FakeBody("not-bytes")})
    monkeypatch.setattr(source, "_client", lambda: wrong_body_client)
    with pytest.raises(ValueError, match="is not bytes"):
        source.fetch_object_bytes(ref)


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
