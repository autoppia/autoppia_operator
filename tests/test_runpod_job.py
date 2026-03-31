from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.runpod_job import (
    _classify_snapshot_pod_status,
    _credential_blockers,
    _credential_warnings,
    _extract_ssh_metadata,
    _fetch_inventory_snapshot,
    _require_runpodctl,
    _seconds_since,
    _select_best_existing_pod,
    create_pod,
    run_job,
)


def test_runpod_job_bootstrap_requires_only_runpod_key(monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    assert _credential_blockers() == ["RUNPOD_API_KEY missing in local shell"]


def test_runpod_job_training_hf_token_is_optional_by_default(monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    assert _credential_blockers(require_runpod=False, require_hf_token=False) == []


def test_runpod_job_training_can_require_hf_token(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    assert _credential_blockers(require_runpod=False, require_hf_token=True) == ["HF_TOKEN or HUGGINGFACE_HUB_TOKEN missing in local shell"]


def test_runpod_job_records_hf_warning_for_public_model_path(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    warnings = _credential_warnings(base_model="browser-use/bu-30b-a3b-preview")
    assert warnings
    assert "browser-use/bu-30b-a3b-preview" in warnings[0]


def test_select_best_existing_pod_prefers_running_a100_name_prefix() -> None:
    pods = [
        {
            "id": "anchor",
            "name": "bu-30b-a3b-bootstrap",
            "desiredStatus": "EXITED",
            "machine": {"gpuDisplayName": "A100 PCIe"},
        },
        {
            "id": "fresh",
            "name": "bu-30b-a3b-bootstrap-arbos",
            "desiredStatus": "RUNNING",
            "machine": {"gpuDisplayName": "A100 PCIe"},
        },
        {
            "id": "other",
            "name": "unrelated",
            "desiredStatus": "RUNNING",
            "machine": {"gpuDisplayName": "A100 PCIe"},
        },
    ]

    selected = _select_best_existing_pod(
        pods=pods,
        preferred_gpu="NVIDIA A100 80GB PCIe",
        anchor_pod_id="anchor",
        preferred_name_prefix="bu-30b-a3b-bootstrap-",
    )

    assert selected is not None
    assert selected["id"] == "fresh"


def test_extract_ssh_metadata_reads_public_ssh_port() -> None:
    runtime = {
        "ports": [
            {"ip": "1.2.3.4", "privatePort": 22, "publicPort": 21043, "type": "tcp"},
            {"ip": "1.2.3.4", "privatePort": 8000, "publicPort": 28000, "type": "http"},
        ]
    }
    assert _extract_ssh_metadata(runtime) == {
        "host": "1.2.3.4",
        "port": 21043,
        "type": "tcp",
        "connection": "ssh",
    }


def test_classify_snapshot_pod_status_marks_running_without_runtime_as_pending_runtime() -> None:
    status, reason = _classify_snapshot_pod_status(
        {
            "desiredStatus": "RUNNING",
            "runtime": None,
        }
    )

    assert status == "running_pending_runtime"
    assert reason is not None
    assert "runtime ports are not exposed yet" in reason


def test_seconds_since_handles_utc_timestamp() -> None:
    assert _seconds_since("2026-03-26T15:00:00Z", now_utc="2026-03-26T15:30:00Z") == 1800.0


def test_fetch_inventory_snapshot_uses_client_balance(monkeypatch) -> None:
    monkeypatch.setattr(
        "training.runpod_job._runpod_api",
        lambda payload: {
            "data": {
                "myself": {
                    "clientBalance": 42.5,
                    "currentSpendPerHr": 1.19,
                    "pods": [{"id": "pod-1", "name": "candidate"}],
                }
            }
        },
    )

    snapshot = _fetch_inventory_snapshot()

    assert snapshot["account_balance_usd"] == 42.5
    assert snapshot["current_spend_per_hour_usd"] == 1.19
    assert snapshot["pods"] == [{"id": "pod-1", "name": "candidate"}]


def test_require_runpodctl_fails_fast_when_binary_missing(monkeypatch) -> None:
    monkeypatch.setattr("training.runpod_job.shutil.which", lambda binary: None)
    monkeypatch.setattr("training.runpod_job.RUNPODCTL_FALLBACK_PATHS", [])

    with pytest.raises(RuntimeError, match="runpodctl is not installed"):
        _require_runpodctl()


def test_create_pod_falls_back_to_api_when_runpodctl_flags_are_unsupported(monkeypatch) -> None:
    monkeypatch.setattr("training.runpod_job._find_runpodctl", lambda: "/tmp/runpodctl")
    monkeypatch.setattr(
        "training.runpod_job.subprocess.run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 1, "stderr": "unknown flag: --cloudType", "stdout": ""})(),
    )
    monkeypatch.setattr("training.runpod_job._create_pod_via_api", lambda **kwargs: "api-pod")

    pod_id = create_pod(cloud_type="SECURE", name="fallback-pod")

    assert pod_id == "api-pod"


def test_run_job_bootstrap_only_refreshes_live_snapshot_before_claiming_ready(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    monkeypatch.setattr("training.runpod_job.get_balance", lambda: 42.0)
    monkeypatch.setattr("training.runpod_job.wait_for_pod", lambda pod_id: True)
    monkeypatch.setattr("training.runpod_job.create_pod", lambda **kwargs: "fresh-pod")
    monkeypatch.setattr("training.runpod_job.shutil.which", lambda binary: None)

    snapshot_path = tmp_path / "inventory.json"

    def fake_refresh(path: Path) -> dict[str, object]:
        payload = {
            "observed_at": "2026-03-26T16:30:00Z",
            "provider": "runpod",
            "observed_via": "test",
            "account_balance_usd": 42.0,
            "pods": [
                {
                    "id": "fresh-pod",
                    "name": "bu-30b-a3b-bootstrap",
                    "desiredStatus": "RUNNING",
                    "runtime": None,
                    "machine": {"gpuDisplayName": "A100 PCIe"},
                }
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("training.runpod_job.refresh_inventory_snapshot", fake_refresh)

    bootstrap_path = tmp_path / "bootstrap.json"
    job = run_job(
        output_dir=str(tmp_path / "adapter"),
        bootstrap_status_path=str(bootstrap_path),
        inventory_snapshot_path=str(snapshot_path),
        bootstrap_only=True,
    )

    status = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert job["status"] == "bootstrap_candidate_pending_runtime"
    assert status["pod_id"] == "fresh-pod"
    assert status["status"] == "running_pending_runtime"
    assert status["runtime"] is None
    assert status["ssh"] is None


def test_run_job_can_record_snapshot_backed_bootstrap_without_local_runpod_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    snapshot = {
        "observed_at": "2026-03-26T15:01:00Z",
        "provider": "runpod",
        "observed_via": "test",
        "account_balance_usd": 10.0,
        "pods": [
            {
                "id": "fresh",
                "name": "bu-30b-a3b-bootstrap-arbos",
                "desiredStatus": "RUNNING",
                "gpuCount": 1,
                "vcpuCount": 12,
                "memoryInGb": 125,
                "volumeInGb": 80,
                "containerDiskInGb": 100,
                "costPerHr": 1.19,
                "runtime": None,
                "machine": {"gpuDisplayName": "A100 PCIe"},
            }
        ],
        "resume_attempt": {"attempted": True, "result": "failed"},
        "capacity_attempts": [{"action": "create_new_pod", "target_pod_id": "fresh"}],
    }
    snapshot_path = tmp_path / "inventory.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    output_dir = tmp_path / "adapter"
    bootstrap_path = tmp_path / "bootstrap.json"
    job = run_job(
        output_dir=str(output_dir),
        bootstrap_status_path=str(bootstrap_path),
        existing_pod_id="59dd4g1snevkqs",
        bootstrap_only=True,
        inventory_snapshot_path=str(snapshot_path),
    )

    status = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert job["status"] == "bootstrap_candidate_pending_runtime"
    assert status["pod_id"] == "fresh"
    assert status["status"] == "running_pending_runtime"
    assert status["active_candidate_pod_id"] == "fresh"
    assert status["anchor_pod_id"] == "59dd4g1snevkqs"
    assert status["credential_blockers"] == []


def test_run_job_preserves_running_bootstrap_artifact_when_training_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("training.runpod_job.get_balance", lambda: 42.0)
    monkeypatch.setattr("training.runpod_job.wait_for_pod", lambda pod_id: True)
    monkeypatch.setattr("training.runpod_job._require_runpodctl", lambda: None)
    monkeypatch.setattr("training.runpod_job.upload_to_pod", lambda *args, **kwargs: None)
    monkeypatch.setattr("training.runpod_job.run_on_pod", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("train exploded")))
    monkeypatch.setattr("training.runpod_job._probe_ssh_endpoint", lambda *args, **kwargs: {"reachable": True})
    monkeypatch.setattr(
        "training.runpod_job.refresh_inventory_snapshot",
        lambda path: {
            "observed_at": "2026-03-26T16:40:00Z",
            "provider": "runpod",
            "observed_via": "test",
            "account_balance_usd": 42.0,
            "pods": [
                {
                    "id": "pod-1",
                    "name": "bu-30b-a3b-bootstrap-arbos",
                    "desiredStatus": "RUNNING",
                    "runtime": {
                        "ports": [
                            {"ip": "1.2.3.4", "privatePort": 22, "publicPort": 2200, "type": "tcp"},
                        ]
                    },
                    "machine": {"gpuDisplayName": "A100 PCIe"},
                }
            ],
        },
    )

    bootstrap_path = tmp_path / "bootstrap.json"
    with pytest.raises(RuntimeError, match="train exploded"):
        run_job(
            output_dir=str(tmp_path / "adapter"),
            bootstrap_status_path=str(bootstrap_path),
            inventory_snapshot_path=str(tmp_path / "inventory.json"),
            require_hf_token=True,
        )

    status = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert status["ssh"]["host"] == "1.2.3.4"
    assert status["status_reason"] == "train exploded"


def test_run_job_preserves_existing_started_at_when_reconciling_same_pod(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    snapshot = {
        "observed_at": "2026-03-26T15:02:03Z",
        "provider": "runpod",
        "observed_via": "test",
        "account_balance_usd": 10.0,
        "pods": [
            {
                "id": "fresh",
                "name": "bu-30b-a3b-bootstrap-arbos",
                "desiredStatus": "RUNNING",
                "runtime": None,
                "machine": {"gpuDisplayName": "A100 PCIe"},
            }
        ],
    }
    snapshot_path = tmp_path / "inventory.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(
            {
                "pod_id": "fresh",
                "started_at": "2026-03-26T15:01:00Z",
            }
        ),
        encoding="utf-8",
    )

    run_job(
        output_dir=str(tmp_path / "adapter"),
        bootstrap_status_path=str(bootstrap_path),
        existing_pod_id="59dd4g1snevkqs",
        bootstrap_only=True,
        inventory_snapshot_path=str(snapshot_path),
    )

    status = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert status["started_at"] == "2026-03-26T15:01:00Z"


def test_run_job_preserves_prior_anchor_when_refreshing_active_pod(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    snapshot = {
        "observed_at": "2026-03-26T15:59:12Z",
        "provider": "runpod",
        "observed_via": "test",
        "account_balance_usd": 10.0,
        "pods": [
            {
                "id": "active-pod",
                "name": "bu-30b-a3b-bootstrap-arbos-r3",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "ports": [
                        {"ip": "1.2.3.4", "privatePort": 22, "publicPort": 2200, "type": "tcp"},
                    ]
                },
                "machine": {"gpuDisplayName": "A100 PCIe"},
            }
        ],
    }
    snapshot_path = tmp_path / "inventory.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(
            {
                "anchor_pod_id": "59dd4g1snevkqs",
                "resumed": False,
                "newly_created": True,
                "pod_id": "active-pod",
                "started_at": "2026-03-26T15:33:49Z",
            }
        ),
        encoding="utf-8",
    )

    run_job(
        output_dir=str(tmp_path / "adapter"),
        bootstrap_status_path=str(bootstrap_path),
        existing_pod_id="active-pod",
        bootstrap_only=True,
        inventory_snapshot_path=str(snapshot_path),
    )

    status = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert status["anchor_pod_id"] == "59dd4g1snevkqs"
    assert status["active_candidate_pod_id"] == "active-pod"
    assert status["resumed"] is False
    assert status["newly_created"] is True


def test_run_job_can_replace_stalled_pending_runtime_pod(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    snapshot_path = tmp_path / "inventory.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "observed_at": "2026-03-26T16:30:00Z",
                "provider": "runpod",
                "observed_via": "test",
                "account_balance_usd": 10.0,
                "pods": [
                    {
                        "id": "stalled",
                        "name": "bu-30b-a3b-bootstrap-arbos",
                        "desiredStatus": "RUNNING",
                        "runtime": None,
                        "machine": {"gpuDisplayName": "A100 PCIe"},
                    }
                ],
                "capacity_attempts": [],
            }
        ),
        encoding="utf-8",
    )
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(json.dumps({"pod_id": "stalled", "started_at": "2026-03-26T15:00:00Z"}), encoding="utf-8")

    observed_calls: list[tuple[str, str]] = []

    def fake_terminate(pod_id: str) -> None:
        observed_calls.append(("terminate", pod_id))

    def fake_create_pod(**kwargs: object) -> str:
        observed_calls.append(("create", str(kwargs["name"])))
        return "replacement"

    refresh_calls = {"count": 0}

    def fake_refresh(path: Path) -> dict[str, object]:
        refresh_calls["count"] += 1
        if refresh_calls["count"] == 1:
            return json.loads(path.read_text(encoding="utf-8"))
        refreshed = {
            "observed_at": "2026-03-26T16:31:00Z",
            "provider": "runpod",
            "observed_via": "test",
            "account_balance_usd": 10.0,
            "pods": [
                {
                    "id": "replacement",
                    "name": "bu-30b-a3b-bootstrap-arbos-r123",
                    "desiredStatus": "RUNNING",
                    "runtime": None,
                    "machine": {"gpuDisplayName": "A100 PCIe"},
                }
            ],
            "capacity_attempts": json.loads(path.read_text(encoding="utf-8")).get("capacity_attempts", []),
        }
        path.write_text(json.dumps(refreshed), encoding="utf-8")
        return refreshed

    monkeypatch.setattr("training.runpod_job.terminate_pod", fake_terminate)
    monkeypatch.setattr("training.runpod_job.create_pod", fake_create_pod)
    monkeypatch.setattr("training.runpod_job.refresh_inventory_snapshot", fake_refresh)
    monkeypatch.setattr("training.runpod_job.time.time", lambda: 123.0)

    job = run_job(
        output_dir=str(tmp_path / "adapter"),
        bootstrap_status_path=str(bootstrap_path),
        existing_pod_id="59dd4g1snevkqs",
        bootstrap_only=True,
        inventory_snapshot_path=str(snapshot_path),
        replace_pending_runtime=True,
        pending_runtime_threshold_seconds=300,
    )

    status = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert job["pod_id"] == "replacement"
    assert status["pod_id"] == "replacement"
    assert status["status"] == "running_pending_runtime"
    assert observed_calls == [("terminate", "stalled"), ("create", "bu-30b-a3b-bootstrap-arbos-r123")]
    attempts = status["capacity_attempts"]
    assert attempts[-2]["action"] == "terminate_stalled_pod"
    assert attempts[-1]["action"] == "create_replacement_pod"


def test_run_job_can_replace_ssh_unreachable_pod(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    snapshot_path = tmp_path / "inventory.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "observed_at": "2026-03-26T16:30:00Z",
                "provider": "runpod",
                "observed_via": "test",
                "account_balance_usd": 10.0,
                "pods": [
                    {
                        "id": "stalled",
                        "name": "bu-30b-a3b-bootstrap-arbos-r2",
                        "desiredStatus": "RUNNING",
                        "runtime": {
                            "ports": [
                                {
                                    "ip": "1.2.3.4",
                                    "privatePort": 22,
                                    "publicPort": 17795,
                                    "type": "tcp",
                                }
                            ]
                        },
                        "machine": {"gpuDisplayName": "A100 PCIe"},
                    }
                ],
                "capacity_attempts": [],
            }
        ),
        encoding="utf-8",
    )
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(json.dumps({"pod_id": "stalled", "started_at": "2026-03-26T15:00:00Z"}), encoding="utf-8")

    observed_calls: list[tuple[str, str]] = []

    def fake_terminate(pod_id: str) -> None:
        observed_calls.append(("terminate", pod_id))

    def fake_create_pod(**kwargs: object) -> str:
        observed_calls.append(("create", str(kwargs["name"])))
        return "replacement"

    refresh_calls = {"count": 0}

    def fake_refresh(path: Path) -> dict[str, object]:
        refresh_calls["count"] += 1
        if refresh_calls["count"] == 1:
            return json.loads(path.read_text(encoding="utf-8"))
        refreshed = {
            "observed_at": "2026-03-26T16:31:00Z",
            "provider": "runpod",
            "observed_via": "test",
            "account_balance_usd": 10.0,
            "pods": [
                {
                    "id": "replacement",
                    "name": "bu-30b-a3b-bootstrap-arbos-r123",
                    "desiredStatus": "RUNNING",
                    "runtime": None,
                    "machine": {"gpuDisplayName": "A100 PCIe"},
                }
            ],
            "capacity_attempts": json.loads(path.read_text(encoding="utf-8")).get("capacity_attempts", []),
        }
        path.write_text(json.dumps(refreshed), encoding="utf-8")
        return refreshed

    monkeypatch.setattr(
        "training.runpod_job._probe_ssh_endpoint",
        lambda ssh, timeout_seconds=5.0: (
            None
            if ssh is None
            else {
                "attempted_at": "2026-03-26T16:30:00Z",
                "host": ssh["host"],
                "port": ssh["port"],
                "timeout_seconds": timeout_seconds,
                "reachable": False,
                "result": "connect_failed",
                "error": "Connection refused",
            }
        ),
    )
    monkeypatch.setattr("training.runpod_job.terminate_pod", fake_terminate)
    monkeypatch.setattr("training.runpod_job.create_pod", fake_create_pod)
    monkeypatch.setattr("training.runpod_job.refresh_inventory_snapshot", fake_refresh)
    monkeypatch.setattr("training.runpod_job.time.time", lambda: 123.0)

    job = run_job(
        output_dir=str(tmp_path / "adapter"),
        bootstrap_status_path=str(bootstrap_path),
        existing_pod_id="59dd4g1snevkqs",
        bootstrap_only=True,
        inventory_snapshot_path=str(snapshot_path),
        replace_unreachable_ssh=True,
        unreachable_ssh_threshold_seconds=300,
    )

    status = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert job["pod_id"] == "replacement"
    assert status["pod_id"] == "replacement"
    assert status["status"] == "running_pending_runtime"
    assert observed_calls == [("terminate", "stalled"), ("create", "bu-30b-a3b-bootstrap-arbos-r123")]
    attempts = status["capacity_attempts"]
    assert attempts[-2]["action"] == "terminate_unreachable_ssh_pod"
    assert attempts[-1]["action"] == "create_replacement_pod"


def test_run_job_download_only_uses_custom_remote_adapter_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("training.runpod_job.get_balance", lambda: 42.0)
    monkeypatch.setattr("training.runpod_job.wait_for_pod", lambda pod_id: True)
    monkeypatch.setattr("training.runpod_job._probe_ssh_endpoint", lambda *args, **kwargs: {"reachable": True})
    monkeypatch.setattr(
        "training.runpod_job.refresh_inventory_snapshot",
        lambda path: {
            "observed_at": "2026-03-26T16:40:00Z",
            "provider": "runpod",
            "observed_via": "test",
            "account_balance_usd": 42.0,
            "pods": [
                {
                    "id": "pod-1",
                    "name": "bu-30b-a3b-bootstrap-arbos",
                    "desiredStatus": "RUNNING",
                    "runtime": {
                        "ports": [
                            {"ip": "1.2.3.4", "privatePort": 22, "publicPort": 2200, "type": "tcp"},
                        ]
                    },
                    "machine": {"gpuDisplayName": "A100 PCIe"},
                }
            ],
        },
    )

    downloads: list[str] = []

    def fake_download_from_pod(pod_id: str, remote_path: str, local_path: str) -> None:
        downloads.append(remote_path)
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "adapter_model.safetensors":
            path.write_bytes(b"x" * 2048)
        elif path.name == "adapter_config.json":
            path.write_text(json.dumps({"base_model_name_or_path": "browser-use/bu-30b-a3b-preview"}), encoding="utf-8")
        elif path.name == "train_metrics.json":
            path.write_text(json.dumps({"base_model": "browser-use/bu-30b-a3b-preview", "train_loss": 0.42}), encoding="utf-8")
        else:
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("training.runpod_job.download_from_pod", fake_download_from_pod)

    output_dir = tmp_path / "adapter"
    job = run_job(
        output_dir=str(output_dir),
        bootstrap_status_path=str(tmp_path / "bootstrap.json"),
        inventory_snapshot_path=str(tmp_path / "inventory.json"),
        remote_adapter_dir="/workspace/autoppia_operator_run/models/bu-30b-lora",
        download_only=True,
    )

    assert job["status"] == "downloaded_existing_adapter"
    assert downloads == [
        "/workspace/autoppia_operator_run/models/bu-30b-lora/adapter_model.safetensors",
        "/workspace/autoppia_operator_run/models/bu-30b-lora/adapter_config.json",
        "/workspace/autoppia_operator_run/models/bu-30b-lora/tokenizer_config.json",
        "/workspace/autoppia_operator_run/models/bu-30b-lora/train_metrics.json",
    ]


def test_run_job_download_only_records_adapter_not_ready(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr("training.runpod_job.get_balance", lambda: 42.0)
    monkeypatch.setattr("training.runpod_job.wait_for_pod", lambda pod_id: True)
    monkeypatch.setattr("training.runpod_job._probe_ssh_endpoint", lambda *args, **kwargs: {"reachable": True})
    monkeypatch.setattr(
        "training.runpod_job.refresh_inventory_snapshot",
        lambda path: {
            "observed_at": "2026-03-26T16:40:00Z",
            "provider": "runpod",
            "observed_via": "test",
            "account_balance_usd": 42.0,
            "pods": [
                {
                    "id": "pod-1",
                    "name": "bu-30b-a3b-bootstrap-arbos",
                    "desiredStatus": "RUNNING",
                    "runtime": {
                        "ports": [
                            {"ip": "1.2.3.4", "privatePort": 22, "publicPort": 2200, "type": "tcp"},
                        ]
                    },
                    "machine": {"gpuDisplayName": "A100 PCIe"},
                }
            ],
        },
    )

    def fake_download_from_pod(pod_id: str, remote_path: str, local_path: str) -> None:
        raise RuntimeError(f"scp: {remote_path}: No such file or directory")

    monkeypatch.setattr("training.runpod_job.download_from_pod", fake_download_from_pod)
    monkeypatch.setattr(
        "training.runpod_job._probe_remote_training_progress",
        lambda pod_id, remote_adapter_dir: {
            "observed_at": "2026-03-26T16:41:00Z",
            "remote_adapter_dir": remote_adapter_dir,
            "checkpoints": [7, 14],
            "highest_checkpoint": 14,
            "trainer_processes": ["4052 python training/finetune_bu.py --base-model browser-use/bu-30b-a3b-preview"],
            "remote_train_metrics": {
                "status": "training",
                "max_steps": 21,
                "epoch": 2.0,
            },
        },
    )

    output_dir = tmp_path / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "base_model": "browser-use/bu-30b-a3b-preview",
                "epochs": 3,
                "lora_rank": 32,
                "train_examples": 105,
                "val_examples": 12,
                "stub": True,
            }
        ),
        encoding="utf-8",
    )

    bootstrap_path = tmp_path / "bootstrap.json"
    job = run_job(
        output_dir=str(output_dir),
        bootstrap_status_path=str(bootstrap_path),
        inventory_snapshot_path=str(tmp_path / "inventory.json"),
        remote_adapter_dir="/workspace/autoppia_operator_run/models/bu-30b-lora",
        download_only=True,
    )

    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))

    assert job["status"] == "adapter_not_ready"
    assert "no such file" in job["error"].lower()
    assert job["remote_probe"]["highest_checkpoint"] == 14
    assert bootstrap["status"] == "running"
    assert "adapter bundle not ready yet" in bootstrap["status_reason"].lower()
    assert "latest remote checkpoint: 14" in bootstrap["status_reason"].lower()
    assert metrics["stub"] is True
    assert metrics["global_step"] == 14
    assert metrics["max_steps"] == 21
    assert metrics["status"] == "training"
    assert metrics["trainer_processes"]
