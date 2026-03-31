"""Orchestrate real LoRA fine-tuning on RunPod and persist bootstrap state.

Usage:
    python -m training.runpod_job \
        --data data/autocinema_trajectory_harvest/sft/train.jsonl \
        --val-data data/autocinema_trajectory_harvest/sft/val.jsonl \
        --output-dir models/bu-30b-lora \
        --existing-pod-id 59dd4g1snevkqs \
        --inventory-snapshot-path training/runpod_inventory_snapshot.json

The command writes:
- ``models/bu-30b-lora/job.json`` for per-run state
- ``training/runpod_bootstrap_status.json`` for the repo-level deployment artifact

It refuses to report success if the downloaded adapter still looks like a stub.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "browser-use/bu-30b-a3b-preview"
DEFAULT_GPU = "NVIDIA A100 80GB PCIe"
DEFAULT_POD_NAME = "bu-30b-a3b-bootstrap"
DEFAULT_BOOTSTRAP_STATUS_PATH = REPO_ROOT / "training" / "runpod_bootstrap_status.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "bu-30b-lora"
DEFAULT_REMOTE_ADAPTER_DIR = "/workspace/lora_output"
RUNNING_POD_STATES = {"RUNNING"}
RUNPODCTL_FALLBACK_PATHS = (
    Path.home() / ".local" / "bin" / "runpodctl",
    Path.home() / ".runpod" / "bin" / "runpodctl",
)
RUNPOD_INVENTORY_QUERY = """
query RunpodInventory {
  myself {
    currentSpendPerHr
    clientBalance
    pods {
      id
      name
      desiredStatus
      imageName
      gpuCount
      vcpuCount
      memoryInGb
      volumeInGb
      containerDiskInGb
      costPerHr
      runtime {
        uptimeInSeconds
        ports {
          ip
          isIpPublic
          privatePort
          publicPort
          type
        }
      }
      machine {
        gpuDisplayName
      }
    }
  }
}
""".strip()
RUNPOD_CREATE_POD_MUTATION = """
mutation ($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
  }
}
""".strip()
RUNPOD_RESUME_POD_MUTATION = """
mutation ($input: PodResumeInput!) {
  podResume(input: $input) {
    id
    desiredStatus
  }
}
""".strip()
RUNPOD_STOP_POD_MUTATION = """
mutation ($input: PodStopInput!) {
  podStop(input: $input) {
    id
    desiredStatus
  }
}
""".strip()
RUNPOD_TERMINATE_POD_MUTATION = """
mutation ($input: PodTerminateInput!) {
  podTerminate(input: $input)
}
""".strip()
HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{8,}")


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_runpodctl_created_at(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f %z UTC")
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seconds_since(timestamp: str | None, *, now_utc: str | None = None) -> float | None:
    observed = _parse_utc_timestamp(timestamp)
    if observed is None:
        return None
    now_value = _parse_utc_timestamp(now_utc or _now_utc())
    if now_value is None:
        return None
    return max(0.0, (now_value - observed).total_seconds())


def _runpod_api(data: Dict[str, Any]) -> Dict[str, Any]:
    import requests

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY not set")

    resp = requests.post(
        "https://api.runpod.io/graphql",
        json=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"]))
    return payload


def _runpod_mutation(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    return _runpod_api({"query": query, "variables": variables}).get("data", {})


def _credential_blockers(*, require_runpod: bool = True, require_hf_token: bool = False) -> list[str]:
    blockers: list[str] = []
    if require_runpod and not os.environ.get("RUNPOD_API_KEY"):
        blockers.append("RUNPOD_API_KEY missing in local shell")
    if require_hf_token and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        blockers.append("HF_TOKEN or HUGGINGFACE_HUB_TOKEN missing in local shell")
    return blockers


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _save_job_info(output_dir: Path, info: Dict[str, Any]) -> None:
    _save_json(output_dir / "job.json", info)


def _save_bootstrap_status(path: Path, payload: Dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = _load_json_file(path)
        except Exception:
            existing = {}
        if isinstance(existing, dict):
            for key in ("control_plane", "anchor_pod_id", "active_candidate_pod_id", "active_candidate_pod_name"):
                if key not in payload and key in existing:
                    payload[key] = existing[key]
    _save_json(path, payload)


def _sanitize_for_artifact(value: Any) -> Any:
    if isinstance(value, str):
        return HF_TOKEN_PATTERN.sub("hf_[REDACTED]", value)
    if isinstance(value, list):
        return [_sanitize_for_artifact(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_for_artifact(item) for key, item in value.items()}
    return value


def _build_bootstrap_payload(
    *,
    base_model: str,
    preferred_gpu: str,
    pod_id: Optional[str],
    pod_name: str,
    status: str,
    started_at: Optional[str],
    resumed: bool,
    newly_created: bool,
    ssh: Optional[dict[str, Any]],
    ssh_probe: Optional[dict[str, Any]],
    runtime: Optional[dict[str, Any]],
    machine: Optional[dict[str, Any]],
    resume_attempt: Dict[str, Any],
    source: Dict[str, Any],
    status_reason: Optional[str] = None,
    credential_blockers: Optional[list[str]] = None,
    credential_warnings: Optional[list[str]] = None,
    capacity_attempts: Optional[list[dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "base_model": base_model,
        "preferred_gpu": preferred_gpu,
        "pod_id": pod_id,
        "pod_name": pod_name,
        "status": status,
        "status_reason": status_reason,
        "observed_at": _now_utc(),
        "started_at": started_at,
        "resumed": resumed,
        "newly_created": newly_created,
        "machine": machine,
        "ssh": ssh,
        "ssh_probe": ssh_probe,
        "runtime": runtime,
        "credential_blockers": credential_blockers if credential_blockers is not None else _credential_blockers(),
        "credential_warnings": credential_warnings or [],
        "resume_attempt": resume_attempt,
        "source": source,
        "capacity_attempts": capacity_attempts or [],
    }
    return payload


def _credential_warnings(*, base_model: str) -> list[str]:
    warnings: list[str] = []
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        warnings.append(
            "HF_TOKEN/HUGGINGFACE_HUB_TOKEN not present; this is acceptable only if "
            f"{base_model} remains publicly downloadable from the pod."
        )
    return warnings


def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prior_bootstrap_field(
    prior_bootstrap_status: dict[str, Any] | None,
    *,
    pod_id: str | None,
    field: str,
) -> Any:
    if not isinstance(prior_bootstrap_status, dict):
        return None
    if str(prior_bootstrap_status.get("pod_id") or "") != str(pod_id or ""):
        return None
    return prior_bootstrap_status.get(field)


def _load_existing_bootstrap_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json_file(path)


def _find_snapshot_pod(snapshot: dict[str, Any] | None, pod_id: str | None) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict) or not pod_id:
        return None
    pods = snapshot.get("pods")
    if not isinstance(pods, list):
        return None
    for pod in pods:
        if isinstance(pod, dict) and str(pod.get("id") or "") == pod_id:
                return pod
    return None


def _resolve_anchor_pod_id(
    *,
    prior_bootstrap_status: dict[str, Any] | None,
    existing_pod_id: str | None,
    selected_pod_id: str | None,
) -> str | None:
    if isinstance(prior_bootstrap_status, dict):
        prior_anchor = prior_bootstrap_status.get("anchor_pod_id")
        if isinstance(prior_anchor, str) and prior_anchor:
            return prior_anchor

    if existing_pod_id and existing_pod_id != selected_pod_id:
        return existing_pod_id

    return None


def _resolve_bootstrap_origin_flags(
    *,
    prior_bootstrap_status: dict[str, Any] | None,
    selected_pod_id: str | None,
    resumed: bool,
    newly_created: bool,
) -> tuple[bool, bool]:
    if not isinstance(prior_bootstrap_status, dict):
        return resumed, newly_created

    prior_pod_id = prior_bootstrap_status.get("pod_id")
    if not isinstance(prior_pod_id, str) or prior_pod_id != selected_pod_id:
        return resumed, newly_created

    prior_resumed = prior_bootstrap_status.get("resumed")
    prior_newly_created = prior_bootstrap_status.get("newly_created")
    if isinstance(prior_resumed, bool) and isinstance(prior_newly_created, bool):
        return prior_resumed, prior_newly_created

    return resumed, newly_created


def _fetch_inventory_snapshot() -> dict[str, Any]:
    payload = _runpod_api({"query": RUNPOD_INVENTORY_QUERY})
    myself = payload.get("data", {}).get("myself", {})
    pods = myself.get("pods")
    return {
        "observed_at": _now_utc(),
        "provider": "runpod",
        "observed_via": "runpod graphql api",
        "account_balance_usd": myself.get("clientBalance"),
        "current_spend_per_hour_usd": myself.get("currentSpendPerHr"),
        "pods": pods if isinstance(pods, list) else [],
    }


def refresh_inventory_snapshot(path: Path) -> dict[str, Any]:
    snapshot = _fetch_inventory_snapshot()
    if path.exists():
        prior = _load_json_file(path)
        for key in ("resume_attempt", "capacity_attempts"):
            value = prior.get(key)
            if value:
                snapshot[key] = value
    _save_json(path, snapshot)
    return snapshot


def _normalize_pod_machine(pod: dict[str, Any]) -> dict[str, Any] | None:
    machine = pod.get("machine")
    if not isinstance(machine, dict):
        return None
    return {
        "gpu_display_name": machine.get("gpuDisplayName"),
        "gpu_count": pod.get("gpuCount"),
        "vcpu_count": pod.get("vcpuCount"),
        "memory_gb": pod.get("memoryInGb"),
        "volume_gb": pod.get("volumeInGb"),
        "container_disk_gb": pod.get("containerDiskInGb"),
        "cost_per_hour_usd": pod.get("costPerHr"),
    }


def _extract_ssh_metadata(runtime: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(runtime, dict):
        return None
    ports = runtime.get("ports")
    if not isinstance(ports, list):
        return None
    for port in ports:
        if not isinstance(port, dict):
            continue
        if int(port.get("privatePort") or 0) != 22:
            continue
        return {
            "host": port.get("ip"),
            "port": port.get("publicPort"),
            "type": port.get("type"),
            "connection": "ssh",
        }
    return None


def _probe_ssh_endpoint(ssh: dict[str, Any] | None, *, timeout_seconds: float = 5.0) -> dict[str, Any] | None:
    if not isinstance(ssh, dict):
        return None
    host = str(ssh.get("host") or "").strip()
    port = ssh.get("port")
    if not host or not isinstance(port, int):
        return None
    probe: dict[str, Any] = {
        "attempted_at": _now_utc(),
        "host": host,
        "port": port,
        "timeout_seconds": timeout_seconds,
    }
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            probe["reachable"] = True
            probe["result"] = "connected"
    except OSError as exc:
        probe["reachable"] = False
        probe["result"] = "connect_failed"
        probe["error"] = str(exc)
    return probe


def _classify_snapshot_pod_status(pod: dict[str, Any]) -> tuple[str, str | None]:
    runtime = pod.get("runtime") if isinstance(pod.get("runtime"), dict) else None
    if runtime:
        return "running", None

    desired_status = str(pod.get("desiredStatus") or "").upper()
    if desired_status in RUNNING_POD_STATES:
        return (
            "running_pending_runtime",
            "RunPod reports the alternate A100 bootstrap pod as RUNNING, but runtime ports are not exposed yet.",
        )
    if desired_status:
        return (
            "provisioning",
            f"RunPod reports desiredStatus={desired_status}, so the bootstrap pod is not ready yet.",
        )
    return "provisioning", "RunPod bootstrap pod status is unknown from the latest inventory snapshot."


def _resolve_status_with_ssh_probe(
    *,
    status: str,
    status_reason: str | None,
    ssh_metadata: dict[str, Any] | None,
) -> tuple[str, str | None, dict[str, Any] | None]:
    ssh_probe = _probe_ssh_endpoint(ssh_metadata)
    if status == "running" and isinstance(ssh_probe, dict) and ssh_probe.get("reachable") is False:
        error = str(ssh_probe.get("error") or "ssh endpoint refused connection")
        return (
            "running_ssh_unreachable",
            f"RunPod exposed runtime/SSH metadata, but the SSH endpoint is still unreachable from the local shell: {error}",
            ssh_probe,
    )
    return status, status_reason, ssh_probe


def _snapshot_runtime_status(
    pod: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str, str | None, dict[str, Any] | None, dict[str, Any] | None]:
    runtime = pod.get("runtime") if isinstance(pod, dict) and isinstance(pod.get("runtime"), dict) else None
    status, status_reason = _classify_snapshot_pod_status(pod or {})
    ssh_metadata = _extract_ssh_metadata(runtime)
    status, status_reason, ssh_probe = _resolve_status_with_ssh_probe(
        status=status,
        status_reason=status_reason,
        ssh_metadata=ssh_metadata,
    )
    return runtime, status, status_reason, ssh_metadata, ssh_probe


def _select_best_existing_pod(
    *,
    pods: list[dict[str, Any]],
    preferred_gpu: str,
    anchor_pod_id: str | None,
    preferred_name_prefix: str,
) -> dict[str, Any] | None:
    normalized_gpu = preferred_gpu.lower()
    candidates: list[dict[str, Any]] = []
    for pod in pods:
        if not isinstance(pod, dict):
            continue
        machine = pod.get("machine")
        gpu_name = ""
        if isinstance(machine, dict):
            gpu_name = str(machine.get("gpuDisplayName") or "").lower()
        desired_status = str(pod.get("desiredStatus") or "").upper()
        if desired_status not in RUNNING_POD_STATES:
            continue
        if "a100" not in normalized_gpu or "a100" not in gpu_name:
            continue
        candidates.append(pod)

    def _score(pod: dict[str, Any]) -> tuple[int, int, str]:
        pod_id = str(pod.get("id") or "")
        pod_name = str(pod.get("name") or "")
        return (
            0 if anchor_pod_id and pod_id == anchor_pod_id else 1,
            0 if pod_name.startswith(preferred_name_prefix) else 1,
            pod_id,
        )

    return sorted(candidates, key=_score)[0] if candidates else None


def _pod_is_running(pod: dict[str, Any] | None) -> bool:
    if not isinstance(pod, dict):
        return False
    return str(pod.get("desiredStatus") or "").upper() in RUNNING_POD_STATES


def get_balance() -> float:
    result = _fetch_inventory_snapshot()
    return float(result.get("account_balance_usd") or 0.0)


def _require_runpodctl() -> None:
    if _find_runpodctl() is None:
        raise RuntimeError(
            "runpodctl is not installed in the local shell; cannot push code, execute training, or download adapters from RunPod."
        )


def _find_runpodctl() -> str | None:
    discovered = shutil.which("runpodctl")
    if discovered:
        return discovered
    for candidate in RUNPODCTL_FALLBACK_PATHS:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _runpodctl_cmd(*args: str) -> list[str]:
    executable = _find_runpodctl()
    if executable is None:
        _require_runpodctl()
        raise AssertionError("unreachable")
    return [executable, *args]


def _create_pod_via_api(
    *,
    gpu_type: str,
    image: str,
    cloud_type: str,
    volume_gb: int,
    container_disk_gb: int,
    min_vcpu_count: int,
    min_memory_gb: int,
    name: str,
) -> str:
    data = _runpod_mutation(
        RUNPOD_CREATE_POD_MUTATION,
        {
            "input": {
                "name": name,
                "gpuTypeId": gpu_type,
                "gpuCount": 1,
                "imageName": image,
                "volumeInGb": volume_gb,
                "containerDiskInGb": container_disk_gb,
                "minVcpuCount": min_vcpu_count,
                "minMemoryInGb": min_memory_gb,
                "cloudType": cloud_type,
                "dockerArgs": "",
                "ports": "22/tcp,8000/http",
                "volumeMountPath": "/workspace",
                "env": [],
            }
        },
    )
    pod = data.get("podFindAndDeployOnDemand", {})
    pod_id = str(pod.get("id") or "").strip()
    if not pod_id:
        raise RuntimeError(f"RunPod create pod returned no id for {name}")
    return pod_id


def _resume_pod_via_api(pod_id: str, *, gpu_count: int = 1) -> None:
    _runpod_mutation(
        RUNPOD_RESUME_POD_MUTATION,
        {"input": {"podId": pod_id, "gpuCount": gpu_count}},
    )


def _stop_pod_via_api(pod_id: str) -> None:
    _runpod_mutation(
        RUNPOD_STOP_POD_MUTATION,
        {"input": {"podId": pod_id}},
    )


def _terminate_pod_via_api(pod_id: str) -> None:
    _runpod_mutation(
        RUNPOD_TERMINATE_POD_MUTATION,
        {"input": {"podId": pod_id}},
    )


def create_pod(
    gpu_type: str = DEFAULT_GPU,
    image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
    cloud_type: str = "COMMUNITY",
    volume_gb: int = 80,
    container_disk_gb: int = 100,
    min_vcpu_count: int = 12,
    min_memory_gb: int = 125,
    name: str = DEFAULT_POD_NAME,
) -> str:
    if _find_runpodctl() is None:
        return _create_pod_via_api(
            gpu_type=gpu_type,
            image=image,
            cloud_type=cloud_type,
            volume_gb=volume_gb,
            container_disk_gb=container_disk_gb,
            min_vcpu_count=min_vcpu_count,
            min_memory_gb=min_memory_gb,
            name=name,
        )

    _require_runpodctl()
    cmd = _runpodctl_cmd(
        "create",
        "pod",
        "--name",
        name,
        "--gpuType",
        gpu_type,
        "--imageName",
        image,
        "--volumeSize",
        str(volume_gb),
        "--containerDiskSize",
        str(container_disk_gb),
        "--vcpu",
        str(min_vcpu_count),
        "--mem",
        str(min_memory_gb),
        "--volumePath",
        "/workspace",
        "--ports",
        "22/tcp",
        "--ports",
        "8000/http",
        "--startSSH",
    )
    if str(cloud_type).upper() == "SECURE":
        cmd.append("--secureCloud")
    else:
        cmd.append("--communityCloud")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        combined_output = result.stderr.strip() or result.stdout.strip()
        if "unknown flag" in combined_output.lower():
            return _create_pod_via_api(
                gpu_type=gpu_type,
                image=image,
                cloud_type=cloud_type,
                volume_gb=volume_gb,
                container_disk_gb=container_disk_gb,
                min_vcpu_count=min_vcpu_count,
                min_memory_gb=min_memory_gb,
                name=name,
            )
        raise RuntimeError(f"Pod creation failed: {combined_output}")

    for line in result.stdout.splitlines():
        lower = line.lower()
        if "pod" in lower and "id" in lower:
            return line.split()[-1]
    return result.stdout.strip().split()[-1]


def wait_for_pod(pod_id: str, timeout: int = 600) -> bool:
    if _find_runpodctl() is None:
        start = time.time()
        while time.time() - start < timeout:
            inventory = _fetch_inventory_snapshot()
            for pod in inventory.get("pods", []):
                if isinstance(pod, dict) and str(pod.get("id") or "") == pod_id:
                    return _pod_is_running(pod)
            time.sleep(15)
        return False

    _require_runpodctl()
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            _runpodctl_cmd("pod", "get", pod_id),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0 and "RUNNING" in result.stdout.upper():
            return True
        time.sleep(15)
    return False


def _runpod_ssh_info(pod_id: str) -> dict[str, Any]:
    _require_runpodctl()
    result = subprocess.run(
        _runpodctl_cmd("ssh", "info", pod_id),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"failed to query ssh info for pod {pod_id}")
    return json.loads(result.stdout)


def _runpod_pod_get(pod_id: str) -> dict[str, Any]:
    _require_runpodctl()
    result = subprocess.run(
        _runpodctl_cmd("pod", "get", pod_id),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"failed to query pod details for pod {pod_id}")
    return json.loads(result.stdout)


def _runpodctl_runtime_metadata(pod_id: str) -> tuple[str | None, dict[str, Any] | None]:
    try:
        pod_info = _runpod_pod_get(pod_id)
    except Exception:
        return None, None
    started_at = _parse_runpodctl_created_at(str(pod_info.get("createdAt") or "").strip())
    ssh_info = pod_info.get("ssh")
    if not isinstance(ssh_info, dict):
        return started_at, None
    host = str(ssh_info.get("ip") or "").strip()
    port = ssh_info.get("port")
    if not host or not isinstance(port, int) or port <= 0:
        return started_at, None
    return started_at, {
        "host": host,
        "port": port,
        "type": "tcp",
        "connection": "ssh",
    }


def _ssh_base_command(pod_id: str) -> tuple[list[str], dict[str, Any]]:
    ssh_info = _runpod_ssh_info(pod_id)
    key_path = str((ssh_info.get("ssh_key") or {}).get("path") or "").strip()
    host = str(ssh_info.get("ip") or "").strip()
    port = int(ssh_info.get("port") or 0)
    if not key_path or not host or port <= 0:
        raise RuntimeError(f"incomplete ssh info for pod {pod_id}: {ssh_info}")
    command = [
        "ssh",
        "-i",
        key_path,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        "-p",
        str(port),
        f"root@{host}",
    ]
    return command, ssh_info


def upload_to_pod(pod_id: str, local_path: str, remote_path: str) -> None:
    ssh_cmd, ssh_info = _ssh_base_command(pod_id)
    scp_cmd = [
        "scp",
        "-i",
        str((ssh_info.get("ssh_key") or {}).get("path")),
        "-P",
        str(ssh_info.get("port")),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        local_path,
        f"root@{ssh_info.get('ip')}:{remote_path}",
    ]
    remote_dir = str(Path(remote_path).parent)
    subprocess.run(
        [*ssh_cmd, f"mkdir -p {shlex.quote(remote_dir)}"],
        check=True,
        timeout=60,
    )
    subprocess.run(
        scp_cmd,
        check=True,
        timeout=300,
    )


def run_on_pod(pod_id: str, command: str) -> str:
    ssh_cmd, _ = _ssh_base_command(pod_id)
    result = subprocess.run(
        [*ssh_cmd, f"bash -lc {shlex.quote(command)}"],
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"pod command failed: {command}")
    return result.stdout


def download_from_pod(pod_id: str, remote_path: str, local_path: str) -> None:
    _, ssh_info = _ssh_base_command(pod_id)
    result = subprocess.run(
        [
            "scp",
            "-i",
            str((ssh_info.get("ssh_key") or {}).get("path")),
            "-P",
            str(ssh_info.get("port")),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "BatchMode=yes",
            f"root@{ssh_info.get('ip')}:{remote_path}",
            local_path,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or f"failed to download {remote_path} from pod {pod_id}"
        )


def _probe_remote_training_progress(pod_id: str, remote_adapter_dir: str) -> dict[str, Any] | None:
    ssh_cmd, _ = _ssh_base_command(pod_id)
    remote_dir = remote_adapter_dir.rstrip("/")
    probe_script = f"""
import json
import subprocess
import time
from pathlib import Path

base = Path({remote_dir!r})
payload = {{
    "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "remote_adapter_dir": str(base),
    "exists": base.exists(),
    "files": {{}},
    "checkpoints": [],
    "highest_checkpoint": None,
    "trainer_processes": [],
    "adapter_ready": False,
}}
if base.exists():
    for path in sorted(base.iterdir(), key=lambda item: item.name):
        entry = {{
            "name": path.name,
            "is_dir": path.is_dir(),
            "size": path.stat().st_size,
        }}
        payload["files"][path.name] = entry
        if path.is_dir() and path.name.startswith("checkpoint-"):
            suffix = path.name.split("-", 1)[1]
            if suffix.isdigit():
                payload["checkpoints"].append(int(suffix))
metrics_path = base / "train_metrics.json"
if metrics_path.exists():
    try:
        payload["remote_train_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive on remote host
        payload["remote_train_metrics_error"] = str(exc)
process_result = subprocess.run(
    ["ps", "-eo", "pid=,ppid=,etimes=,args="],
    capture_output=True,
    text=True,
)
if process_result.returncode == 0:
    trainer_processes = []
    for line in process_result.stdout.splitlines():
        line = line.strip()
        if "python training/finetune_bu.py" not in line:
            continue
        if "python - <<'PY'" in line:
            continue
        if "bash -lc" in line:
            continue
        trainer_processes.append(line)
    payload["trainer_processes"] = trainer_processes
if payload["checkpoints"]:
    payload["highest_checkpoint"] = max(payload["checkpoints"])
required_files = ("adapter_model.safetensors", "adapter_config.json", "train_metrics.json")
payload["adapter_ready"] = all((base / name).exists() for name in required_files)
print(json.dumps(payload))
""".strip()
    result = subprocess.run(
        [*ssh_cmd, f"python - <<'PY'\n{probe_script}\nPY"],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if result.returncode != 0:
        return {
            "observed_at": _now_utc(),
            "remote_adapter_dir": remote_dir,
            "probe_error": result.stderr.strip() or result.stdout.strip() or f"probe failed for pod {pod_id}",
        }
    stdout = result.stdout.strip()
    if not stdout:
        return {
            "observed_at": _now_utc(),
            "remote_adapter_dir": remote_dir,
            "probe_error": "remote training probe returned empty stdout",
        }
    try:
        return _sanitize_for_artifact(json.loads(stdout.splitlines()[-1]))
    except json.JSONDecodeError:
        return {
            "observed_at": _now_utc(),
            "remote_adapter_dir": remote_dir,
            "probe_error": f"remote training probe returned non-JSON stdout: {stdout[-200:]}",
        }


def _refresh_local_stub_train_metrics(
    *,
    output_dir: Path,
    remote_probe: dict[str, Any] | None,
    base_model: str,
    epochs: int,
    lora_rank: int,
) -> None:
    metrics_path = output_dir / "train_metrics.json"
    payload: dict[str, Any]
    if metrics_path.exists():
        payload = _load_json_file(metrics_path)
    else:
        payload = {
            "base_model": base_model,
            "epochs": epochs,
            "lora_rank": lora_rank,
            "stub": True,
        }

    if payload.get("stub") is not True:
        return

    payload["base_model"] = base_model
    payload["epochs"] = epochs
    payload["lora_rank"] = lora_rank

    if isinstance(remote_probe, dict):
        observed_at = remote_probe.get("observed_at")
        if isinstance(observed_at, str) and observed_at:
            payload["last_observed_at"] = observed_at
        highest_checkpoint = remote_probe.get("highest_checkpoint")
        if isinstance(highest_checkpoint, int):
            payload["global_step"] = highest_checkpoint
        remote_metrics = remote_probe.get("remote_train_metrics")
        if isinstance(remote_metrics, dict):
            for key in ("max_steps", "epoch", "learning_rate", "train_loss", "eval_loss", "status", "started_at", "updated_at"):
                value = remote_metrics.get(key)
                if value is not None:
                    payload[key] = value
        trainer_processes = remote_probe.get("trainer_processes")
        if isinstance(trainer_processes, list):
            payload["trainer_processes"] = trainer_processes
        payload["remote_progress"] = remote_probe
        checkpoint_summary = remote_probe.get("checkpoints")
        note_parts = [
            "Stub weights only.",
            "The remote RunPod trainer is still in progress.",
        ]
        if isinstance(checkpoint_summary, list) and checkpoint_summary:
            note_parts.append(f"Observed checkpoints: {', '.join(str(item) for item in checkpoint_summary)}.")
        probe_error = remote_probe.get("probe_error")
        if isinstance(probe_error, str) and probe_error:
            note_parts.append(f"Remote probe warning: {probe_error}.")
        payload["note"] = " ".join(note_parts)

    _save_json(metrics_path, payload)


def terminate_pod(pod_id: str) -> None:
    if _find_runpodctl() is None:
        _terminate_pod_via_api(pod_id)
        return

    _require_runpodctl()
    subprocess.run(_runpodctl_cmd("pod", "delete", pod_id), check=True, timeout=30)


def resume_pod(pod_id: str) -> None:
    if _find_runpodctl() is None:
        _resume_pod_via_api(pod_id)
        return

    _require_runpodctl()
    subprocess.run(_runpodctl_cmd("pod", "start", pod_id), check=True, timeout=30)


def stop_pod(pod_id: str) -> None:
    if _find_runpodctl() is None:
        _stop_pod_via_api(pod_id)
        return

    _require_runpodctl()
    subprocess.run(_runpodctl_cmd("pod", "stop", pod_id), check=True, timeout=30)


def _validate_downloaded_adapter(output_dir: Path) -> None:
    config_path = output_dir / "adapter_config.json"
    model_path = output_dir / "adapter_model.safetensors"
    metrics_path = output_dir / "train_metrics.json"
    for path in (config_path, model_path, metrics_path):
        if not path.exists():
            raise RuntimeError(f"missing downloaded artifact: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if config.get("stub") is True:
        raise RuntimeError(f"downloaded adapter is still marked stub: {config_path}")
    if metrics.get("stub") is True:
        raise RuntimeError(f"downloaded training metrics are still marked stub: {metrics_path}")
    if str(metrics.get("base_model")) != DEFAULT_BASE_MODEL:
        raise RuntimeError(
            "downloaded training metrics do not point at browser-use/bu-30b-a3b-preview"
        )
    if model_path.stat().st_size < 1024:
        raise RuntimeError(f"downloaded adapter weights look too small: {model_path}")


def _download_adapter_bundle(*, pod_id: str, remote_adapter_dir: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = remote_adapter_dir.rstrip("/")
    for filename in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "train_metrics.json",
    ):
        download_from_pod(pod_id, f"{remote_dir}/{filename}", str(output_dir / filename))
    _validate_downloaded_adapter(output_dir)


def _adapter_bundle_not_ready_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "no such file",
            "not a regular file",
            "missing downloaded artifact",
            "still marked stub",
            "look too small",
            "failed to download",
        )
    )


def run_job(
    *,
    data_path: str = "data/autocinema_trajectory_harvest/sft/train.jsonl",
    val_data_path: str = "data/autocinema_trajectory_harvest/sft/val.jsonl",
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    gpu_type: str = DEFAULT_GPU,
    base_model: str = DEFAULT_BASE_MODEL,
    pod_name: str = DEFAULT_POD_NAME,
    existing_pod_id: Optional[str] = None,
    bootstrap_status_path: str = str(DEFAULT_BOOTSTRAP_STATUS_PATH),
    epochs: int = 3,
    lora_rank: int = 32,
    keep_pod: bool = False,
    bootstrap_only: bool = False,
    require_hf_token: bool = False,
    inventory_snapshot_path: Optional[str] = None,
    replace_pending_runtime: bool = False,
    pending_runtime_threshold_seconds: int = 1800,
    replace_unreachable_ssh: bool = False,
    unreachable_ssh_threshold_seconds: int = 600,
    cloud_type: str = "SECURE",
    volume_gb: int = 80,
    remote_adapter_dir: str = DEFAULT_REMOTE_ADAPTER_DIR,
    download_only: bool = False,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    bootstrap_path = Path(bootstrap_status_path)
    started_at = None
    pod_id = existing_pod_id
    pod_created = False
    pod_resumed = existing_pod_id is not None
    job_info: Dict[str, Any] = {
        "status": "starting",
        "base_model": base_model,
        "pod_id": existing_pod_id,
        "pod_name": pod_name,
        "data_path": data_path,
        "val_data_path": val_data_path,
        "output_dir": str(output_path),
        "remote_adapter_dir": remote_adapter_dir,
        "started_at": _now_utc(),
    }
    resume_attempt: Dict[str, Any] = {"attempted": bool(existing_pod_id), "attempted_at": None}
    source: Dict[str, Any] = {"image_name": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"}
    machine = {
        "gpu_display_name": gpu_type,
    }
    runtime: dict[str, Any] | None = None
    status = "starting"
    status_reason: str | None = None
    ssh_metadata: dict[str, Any] | None = None
    ssh_probe: dict[str, Any] | None = None
    inventory_snapshot: dict[str, Any] | None = None
    selected_pod: dict[str, Any] | None = None
    selected_pod_running = False
    prior_bootstrap_status = _load_existing_bootstrap_status(bootstrap_path)
    capacity_attempts: list[dict[str, Any]] = []
    credential_blockers = _credential_blockers()
    credential_warnings = _credential_warnings(base_model=base_model)
    prior_pod_name = _prior_bootstrap_field(prior_bootstrap_status, pod_id=pod_id, field="pod_name")
    prior_started_at = _prior_bootstrap_field(prior_bootstrap_status, pod_id=pod_id, field="started_at")
    prior_runtime = _prior_bootstrap_field(prior_bootstrap_status, pod_id=pod_id, field="runtime")
    prior_ssh = _prior_bootstrap_field(prior_bootstrap_status, pod_id=pod_id, field="ssh")
    prior_ssh_probe = _prior_bootstrap_field(prior_bootstrap_status, pod_id=pod_id, field="ssh_probe")
    prior_machine = _prior_bootstrap_field(prior_bootstrap_status, pod_id=pod_id, field="machine")
    prior_status = _prior_bootstrap_field(prior_bootstrap_status, pod_id=pod_id, field="status")
    prior_status_reason = _prior_bootstrap_field(prior_bootstrap_status, pod_id=pod_id, field="status_reason")

    if isinstance(prior_pod_name, str) and prior_pod_name:
        pod_name = prior_pod_name
        job_info["pod_name"] = pod_name
    if isinstance(prior_started_at, str) and prior_started_at:
        started_at = prior_started_at
    if isinstance(prior_runtime, dict):
        runtime = prior_runtime
    if isinstance(prior_ssh, dict):
        ssh_metadata = prior_ssh
    if isinstance(prior_ssh_probe, dict):
        ssh_probe = prior_ssh_probe
    if isinstance(prior_machine, dict):
        machine = prior_machine

    if inventory_snapshot_path and os.environ.get("RUNPOD_API_KEY"):
        inventory_snapshot = refresh_inventory_snapshot(Path(inventory_snapshot_path))

    if inventory_snapshot_path and inventory_snapshot is None:
        inventory_snapshot = _load_json_file(Path(inventory_snapshot_path))

    if inventory_snapshot_path and inventory_snapshot is not None:
        if isinstance(inventory_snapshot.get("capacity_attempts"), list):
            capacity_attempts = list(inventory_snapshot["capacity_attempts"])
        pods = inventory_snapshot.get("pods")
        if isinstance(pods, list):
            selected_pod = _select_best_existing_pod(
                pods=pods,
                preferred_gpu=gpu_type,
                anchor_pod_id=existing_pod_id,
                preferred_name_prefix=f"{pod_name}-",
            )
            if selected_pod is not None:
                selected_pod_running = _pod_is_running(selected_pod)
                pod_id = str(selected_pod.get("id") or pod_id or "")
                pod_name = str(selected_pod.get("name") or pod_name)
                pod_created = pod_id != existing_pod_id
                pod_resumed = pod_id == existing_pod_id
                machine = _normalize_pod_machine(selected_pod) or machine
                source["inventory_snapshot_path"] = str(Path(inventory_snapshot_path))
                source["account_balance_usd"] = inventory_snapshot.get("account_balance_usd")
                source["observed_via"] = inventory_snapshot.get("observed_via")

    _save_job_info(output_path, job_info)

    if inventory_snapshot is not None and bootstrap_only and pod_id:
        pods = inventory_snapshot.get("pods")
        selected_pod = _find_snapshot_pod(inventory_snapshot, pod_id)
        if selected_pod is not None:
            runtime, status, status_reason, ssh_metadata, ssh_probe = _snapshot_runtime_status(selected_pod)
            prior_started_at = None
            if (
                isinstance(prior_bootstrap_status, dict)
                and str(prior_bootstrap_status.get("pod_id") or "") == pod_id
                and isinstance(prior_bootstrap_status.get("started_at"), str)
            ):
                prior_started_at = str(prior_bootstrap_status["started_at"])
            started_at = prior_started_at or inventory_snapshot.get("observed_at")
            pod_resumed, pod_created = _resolve_bootstrap_origin_flags(
                prior_bootstrap_status=prior_bootstrap_status,
                selected_pod_id=pod_id,
                resumed=pod_resumed,
                newly_created=pod_created,
            )
            pending_runtime_age_seconds = None
            if status == "running_pending_runtime":
                pending_runtime_age_seconds = _seconds_since(
                    started_at if isinstance(started_at, str) else None,
                    now_utc=inventory_snapshot.get("observed_at") if isinstance(inventory_snapshot.get("observed_at"), str) else None,
                )
            ssh_unreachable_age_seconds = None
            if status == "running_ssh_unreachable":
                ssh_unreachable_age_seconds = _seconds_since(
                    started_at if isinstance(started_at, str) else None,
                    now_utc=inventory_snapshot.get("observed_at") if isinstance(inventory_snapshot.get("observed_at"), str) else None,
                )
            if (
                replace_pending_runtime
                and runtime is None
                and status == "running_pending_runtime"
                and os.environ.get("RUNPOD_API_KEY")
                and pending_runtime_age_seconds is not None
                and pending_runtime_age_seconds >= float(pending_runtime_threshold_seconds)
            ):
                replacement_name = f"{DEFAULT_POD_NAME}-arbos-r{int(time.time())}"
                capacity_attempts.append(
                    {
                        "attempted_at": _now_utc(),
                        "action": "terminate_stalled_pod",
                        "target_pod_id": pod_id,
                        "result": "requested",
                        "reason": "runtime_missing_beyond_threshold",
                        "threshold_seconds": pending_runtime_threshold_seconds,
                    }
                )
                terminate_pod(pod_id)
                new_pod_id = create_pod(
                    gpu_type=gpu_type,
                    name=replacement_name,
                    cloud_type=cloud_type,
                    volume_gb=volume_gb,
                )
                capacity_attempts.append(
                    {
                        "attempted_at": _now_utc(),
                        "action": "create_replacement_pod",
                        "target_pod_id": new_pod_id,
                        "target_pod_name": replacement_name,
                        "gpu_type": gpu_type,
                        "cloud_type": cloud_type,
                        "result": "requested",
                    }
                )
                inventory_snapshot["capacity_attempts"] = capacity_attempts
                if inventory_snapshot_path:
                    _save_json(Path(inventory_snapshot_path), inventory_snapshot)
                    inventory_snapshot = refresh_inventory_snapshot(Path(inventory_snapshot_path))
                    pods = inventory_snapshot.get("pods")
                    if isinstance(pods, list):
                        for item in pods:
                            if isinstance(item, dict) and str(item.get("id") or "") == new_pod_id:
                                selected_pod = item
                                break
                pod_id = new_pod_id
                pod_name = replacement_name
                pod_created = True
                pod_resumed = False
                started_at = inventory_snapshot.get("observed_at") if isinstance(inventory_snapshot.get("observed_at"), str) else _now_utc()
                if isinstance(selected_pod, dict):
                    runtime, status, status_reason, ssh_metadata, ssh_probe = _snapshot_runtime_status(selected_pod)
                else:
                    runtime = None
                    status = "provisioning"
                    status_reason = "Replacement pod was requested but has not appeared in the refreshed inventory yet."
                    ssh_metadata = None
                    ssh_probe = None
            elif (
                replace_unreachable_ssh
                and status == "running_ssh_unreachable"
                and os.environ.get("RUNPOD_API_KEY")
                and ssh_unreachable_age_seconds is not None
                and ssh_unreachable_age_seconds >= float(unreachable_ssh_threshold_seconds)
            ):
                replacement_name = f"{DEFAULT_POD_NAME}-arbos-r{int(time.time())}"
                stale_pod_id = pod_id
                capacity_attempts.append(
                    {
                        "attempted_at": _now_utc(),
                        "action": "terminate_unreachable_ssh_pod",
                        "target_pod_id": stale_pod_id,
                        "result": "requested",
                        "reason": "ssh_connection_refused_beyond_threshold",
                        "threshold_seconds": unreachable_ssh_threshold_seconds,
                    }
                )
                terminate_pod(stale_pod_id)
                new_pod_id = create_pod(
                    gpu_type=gpu_type,
                    name=replacement_name,
                    cloud_type=cloud_type,
                    volume_gb=volume_gb,
                )
                capacity_attempts.append(
                    {
                        "attempted_at": _now_utc(),
                        "action": "create_replacement_pod",
                        "target_pod_id": new_pod_id,
                        "target_pod_name": replacement_name,
                        "gpu_type": gpu_type,
                        "cloud_type": cloud_type,
                        "result": "requested",
                        "replaces_pod_id": stale_pod_id,
                    }
                )
                inventory_snapshot["capacity_attempts"] = capacity_attempts
                if inventory_snapshot_path:
                    _save_json(Path(inventory_snapshot_path), inventory_snapshot)
                    inventory_snapshot = refresh_inventory_snapshot(Path(inventory_snapshot_path))
                    pods = inventory_snapshot.get("pods")
                    if isinstance(pods, list):
                        selected_pod = None
                        for item in pods:
                            if isinstance(item, dict) and str(item.get("id") or "") == new_pod_id:
                                selected_pod = item
                                break
                pod_id = new_pod_id
                pod_name = replacement_name
                pod_created = True
                pod_resumed = False
                started_at = inventory_snapshot.get("observed_at") if isinstance(inventory_snapshot.get("observed_at"), str) else _now_utc()
                if isinstance(selected_pod, dict):
                    runtime, status, status_reason, ssh_metadata, ssh_probe = _snapshot_runtime_status(selected_pod)
                else:
                    runtime = None
                    status = "provisioning"
                    status_reason = "Replacement pod was requested after repeated SSH refusal, but it has not appeared in the refreshed inventory yet."
                    ssh_metadata = None
                    ssh_probe = None
            payload = _build_bootstrap_payload(
                base_model=base_model,
                preferred_gpu=gpu_type,
                pod_id=pod_id,
                pod_name=pod_name,
                status=status,
                status_reason=status_reason,
                started_at=started_at if isinstance(started_at, str) else None,
                resumed=pod_resumed,
                newly_created=pod_created,
                ssh=ssh_metadata,
                ssh_probe=ssh_probe,
                runtime=runtime,
                machine=machine,
                resume_attempt=inventory_snapshot.get("resume_attempt") if isinstance(inventory_snapshot.get("resume_attempt"), dict) else resume_attempt,
                source=source,
                credential_blockers=[],
                credential_warnings=credential_warnings,
                capacity_attempts=capacity_attempts,
            )
            payload["control_plane"] = {
                "provider": inventory_snapshot.get("provider") or "runpod",
                "observed_via": inventory_snapshot.get("observed_via") or "inventory snapshot",
                "account_balance_usd": inventory_snapshot.get("account_balance_usd"),
                "active_pod_count": len(pods) if isinstance(pods, list) else None,
                "pods_seen": [
                    {
                        "id": pod.get("id"),
                        "name": pod.get("name"),
                        "desired_status": pod.get("desiredStatus"),
                        "gpu_display_name": (pod.get("machine") or {}).get("gpuDisplayName") if isinstance(pod.get("machine"), dict) else None,
                        "public_ssh_port": (_extract_ssh_metadata(pod.get("runtime") if isinstance(pod.get("runtime"), dict) else None) or {}).get("port"),
                        "cost_per_hour_usd": pod.get("costPerHr"),
                    }
                    for pod in pods
                    if isinstance(pod, dict)
                ],
            }
            payload["anchor_pod_id"] = _resolve_anchor_pod_id(
                prior_bootstrap_status=prior_bootstrap_status,
                existing_pod_id=existing_pod_id,
                selected_pod_id=pod_id,
            )
            payload["active_candidate_pod_id"] = pod_id
            payload["active_candidate_pod_name"] = pod_name
            _save_bootstrap_status(bootstrap_path, payload)
            if status == "running":
                job_info["status"] = "bootstrap_ready_snapshot"
            elif status == "running_ssh_unreachable":
                job_info["status"] = "bootstrap_candidate_ssh_unreachable"
            else:
                job_info["status"] = "bootstrap_candidate_pending_runtime"
            job_info["pod_id"] = pod_id
            job_info["pod_name"] = pod_name
            _save_job_info(output_path, job_info)
            return job_info

    if credential_blockers:
        status = _build_bootstrap_payload(
            base_model=base_model,
            preferred_gpu=gpu_type,
            pod_id=pod_id,
            pod_name=pod_name,
            status="resume_blocked" if existing_pod_id else "provisioning_blocked",
            status_reason="Required local credentials are missing; refusing to fabricate a RunPod launch.",
            started_at=None,
            resumed=bool(existing_pod_id),
            newly_created=False,
            ssh=None,
            ssh_probe=None,
            runtime=None,
            machine=machine,
            resume_attempt=resume_attempt,
            source=source,
            credential_blockers=credential_blockers,
            credential_warnings=credential_warnings,
            capacity_attempts=capacity_attempts,
        )
        _save_bootstrap_status(bootstrap_path, status)
        raise RuntimeError("; ".join(credential_blockers))

    try:
        balance = get_balance()
        source["account_balance_usd"] = balance
        if 0 < balance < 5.0:
            raise RuntimeError(f"RunPod balance too low: ${balance:.2f}")

        if pod_id is None:
            job_info["status"] = "provisioning"
            _save_job_info(output_path, job_info)
            pod_id = create_pod(gpu_type=gpu_type, name=pod_name, cloud_type=cloud_type, volume_gb=volume_gb)
            pod_created = True
            pod_resumed = False
            started_at = _now_utc()
            capacity_attempts.append(
                {
                    "attempted_at": started_at,
                    "action": "create_new_pod",
                    "target_pod_id": pod_id,
                    "target_pod_name": pod_name,
                    "gpu_type": gpu_type,
                    "cloud_type": cloud_type,
                    "result": "requested",
                }
            )
        else:
            resume_attempt["attempted_at"] = _now_utc()
            if not selected_pod_running and pod_id == existing_pod_id:
                started_at = _now_utc()
                resume_pod(pod_id)
                resume_attempt["result"] = "requested"
                capacity_attempts.append(
                    {
                        "attempted_at": started_at,
                        "action": "resume_existing_pod",
                        "target_pod_id": pod_id,
                        "gpu_type": gpu_type,
                        "result": "requested",
                    }
                )
            else:
                if not isinstance(started_at, str) or not started_at:
                    started_at = _now_utc()
                pod_resumed = False
                resume_attempt["result"] = "not_needed"

        job_info["pod_id"] = pod_id
        job_info["pod_name"] = pod_name
        bootstrap_status_value = "provisioning"
        bootstrap_status_reason = None
        bootstrap_runtime = None
        bootstrap_ssh = None
        bootstrap_ssh_probe = None
        if download_only and str(pod_id or "") == str(existing_pod_id or ""):
            bootstrap_status_value = str(prior_status or "running")
            bootstrap_status_reason = prior_status_reason if isinstance(prior_status_reason, str) else None
            bootstrap_runtime = runtime if isinstance(runtime, dict) else prior_runtime if isinstance(prior_runtime, dict) else None
            bootstrap_ssh = ssh_metadata if isinstance(ssh_metadata, dict) else prior_ssh if isinstance(prior_ssh, dict) else None
            bootstrap_ssh_probe = ssh_probe if isinstance(ssh_probe, dict) else prior_ssh_probe if isinstance(prior_ssh_probe, dict) else None
        _save_bootstrap_status(
            bootstrap_path,
            _build_bootstrap_payload(
                base_model=base_model,
                preferred_gpu=gpu_type,
                pod_id=pod_id,
                pod_name=pod_name,
                status=bootstrap_status_value,
                status_reason=bootstrap_status_reason,
                started_at=started_at,
                resumed=pod_resumed,
                newly_created=pod_created,
                ssh=bootstrap_ssh,
                ssh_probe=bootstrap_ssh_probe,
                runtime=bootstrap_runtime,
                machine=machine,
                resume_attempt=resume_attempt,
                source=source,
                credential_blockers=credential_blockers,
                credential_warnings=credential_warnings,
                capacity_attempts=capacity_attempts,
            ),
        )

        if not wait_for_pod(pod_id):
            raise RuntimeError("Pod failed to reach RUNNING before timeout")

        snapshot_pod = None
        runtime = None
        status = "running"
        status_reason = None
        ssh_metadata = {"connection": "runpodctl / RunPod UI", "pod_id": pod_id}
        if inventory_snapshot_path and os.environ.get("RUNPOD_API_KEY"):
            inventory_snapshot = refresh_inventory_snapshot(Path(inventory_snapshot_path))
            snapshot_pod = _find_snapshot_pod(inventory_snapshot, pod_id)
            if snapshot_pod is not None:
                machine = _normalize_pod_machine(snapshot_pod) or machine
                runtime = snapshot_pod.get("runtime") if isinstance(snapshot_pod.get("runtime"), dict) else None
                ssh_metadata = _extract_ssh_metadata(runtime)
                status, status_reason = _classify_snapshot_pod_status(snapshot_pod)
                if runtime is not None and ssh_metadata is None:
                    ssh_metadata = {"connection": "runpodctl / RunPod UI", "pod_id": pod_id}
        runpodctl_started_at, runpodctl_ssh = _runpodctl_runtime_metadata(pod_id)
        if runpodctl_started_at:
            started_at = runpodctl_started_at
        if runtime is None or not (isinstance(ssh_metadata, dict) and ssh_metadata.get("host") and ssh_metadata.get("port")):
            if runpodctl_ssh is not None:
                ssh_metadata = runpodctl_ssh
        status, status_reason, ssh_probe = _resolve_status_with_ssh_probe(
            status=status,
            status_reason=status_reason,
            ssh_metadata=ssh_metadata,
        )

        _save_bootstrap_status(
            bootstrap_path,
            _build_bootstrap_payload(
                base_model=base_model,
                preferred_gpu=gpu_type,
                pod_id=pod_id,
                pod_name=pod_name,
                status=status,
                status_reason=status_reason,
                started_at=started_at,
                resumed=pod_resumed,
                newly_created=pod_created,
                ssh=ssh_metadata,
                ssh_probe=ssh_probe,
                runtime=runtime,
                machine=machine,
                resume_attempt=resume_attempt,
                source=source,
                credential_blockers=credential_blockers,
                credential_warnings=credential_warnings,
                capacity_attempts=capacity_attempts,
            ),
        )

        if bootstrap_only:
            if status == "running":
                job_info["status"] = "bootstrap_ready"
            elif status == "running_ssh_unreachable":
                job_info["status"] = "bootstrap_candidate_ssh_unreachable"
            else:
                job_info["status"] = "bootstrap_candidate_pending_runtime"
            _save_job_info(output_path, job_info)
            return job_info

        if download_only:
            logger.info("Downloading existing adapter bundle from pod %s", pod_id)
            try:
                _download_adapter_bundle(
                    pod_id=pod_id,
                    remote_adapter_dir=remote_adapter_dir,
                    output_dir=output_path,
                )
            except Exception as exc:
                if not _adapter_bundle_not_ready_error(exc):
                    raise
                remote_probe = _probe_remote_training_progress(pod_id, remote_adapter_dir)
                job_info["status"] = "adapter_not_ready"
                job_info["error"] = str(exc)
                if isinstance(remote_probe, dict):
                    job_info["remote_probe"] = remote_probe
                _save_job_info(output_path, job_info)
                _refresh_local_stub_train_metrics(
                    output_dir=output_path,
                    remote_probe=remote_probe,
                    base_model=base_model,
                    epochs=epochs,
                    lora_rank=lora_rank,
                )
                status_reason = f"Remote adapter bundle not ready yet: {exc}"
                if isinstance(remote_probe, dict):
                    highest_checkpoint = remote_probe.get("highest_checkpoint")
                    trainer_processes = remote_probe.get("trainer_processes")
                    if isinstance(highest_checkpoint, int):
                        status_reason += f" Latest remote checkpoint: {highest_checkpoint}."
                    if isinstance(trainer_processes, list) and trainer_processes:
                        status_reason += " Trainer process still active."
                _save_bootstrap_status(
                    bootstrap_path,
                    _build_bootstrap_payload(
                        base_model=base_model,
                        preferred_gpu=gpu_type,
                        pod_id=pod_id,
                        pod_name=pod_name,
                        status=status if status not in {"starting", "training"} else "running",
                        status_reason=status_reason,
                        started_at=started_at,
                        resumed=pod_resumed,
                        newly_created=pod_created,
                        ssh=ssh_metadata,
                        ssh_probe=ssh_probe,
                        runtime=runtime,
                        machine=machine,
                        resume_attempt=resume_attempt,
                        source=source,
                        credential_blockers=credential_blockers,
                        credential_warnings=credential_warnings,
                        capacity_attempts=capacity_attempts,
                    ),
                )
                return job_info
            job_info["status"] = "downloaded_existing_adapter"
            _save_job_info(output_path, job_info)
            return job_info

        _require_runpodctl()
        training_blockers = _credential_blockers(require_runpod=False, require_hf_token=require_hf_token)
        if training_blockers:
            raise RuntimeError("; ".join(training_blockers))

        logger.info("Uploading training inputs to pod %s", pod_id)
        upload_to_pod(pod_id, data_path, "/workspace/train.jsonl")
        has_val_data = bool(val_data_path) and Path(val_data_path).exists()
        if has_val_data:
            upload_to_pod(pod_id, val_data_path, "/workspace/val.jsonl")
        upload_to_pod(pod_id, str(REPO_ROOT / "training" / "finetune_bu.py"), "/workspace/finetune_bu.py")

        job_info["status"] = "training"
        _save_job_info(output_path, job_info)

        train_parts = [
            "cd /workspace &&",
            "python finetune_bu.py",
            "--base-model",
            shlex.quote(base_model),
            "--data train.jsonl",
        ]
        if has_val_data:
            train_parts.append("--val-data val.jsonl")
        train_parts.extend(
            [
                "--output-dir /workspace/lora_output",
                "--epochs",
                str(epochs),
                "--lora-rank",
                str(lora_rank),
            ]
        )
        train_cmd = " ".join(train_parts)
        logger.info("Training on pod %s", pod_id)
        training_output = run_on_pod(pod_id, train_cmd)
        job_info["training_output_tail"] = training_output[-4000:]
        _save_job_info(output_path, job_info)

        _download_adapter_bundle(
            pod_id=pod_id,
            remote_adapter_dir=remote_adapter_dir,
            output_dir=output_path,
        )
        job_info["status"] = "completed"
        _save_job_info(output_path, job_info)
        return job_info

    except Exception as exc:
        logger.error("RunPod job failed: %s", exc)
        job_info["status"] = "failed"
        job_info["error"] = str(exc)
        _save_job_info(output_path, job_info)
        _save_bootstrap_status(
            bootstrap_path,
            _build_bootstrap_payload(
                base_model=base_model,
                preferred_gpu=gpu_type,
                pod_id=pod_id,
                pod_name=pod_name,
                status=status if status not in {"starting", "training"} else "running",
                status_reason=str(exc),
                started_at=started_at,
                resumed=pod_resumed,
                newly_created=pod_created,
                ssh=ssh_metadata,
                ssh_probe=ssh_probe,
                runtime=runtime,
                machine=machine,
                resume_attempt=resume_attempt,
                source=source,
                credential_blockers=credential_blockers,
                credential_warnings=credential_warnings,
                capacity_attempts=capacity_attempts,
            ),
        )
        raise

    finally:
        if pod_id and not keep_pod and job_info.get("status") == "completed":
            try:
                terminate_pod(pod_id)
                job_info["pod_terminated"] = True
                _save_job_info(output_path, job_info)
            except Exception as exc:
                logger.error("FAILED to terminate pod %s: %s", pod_id, exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run LoRA fine-tune on RunPod")
    parser.add_argument("--data", default="data/autocinema_trajectory_harvest/sft/train.jsonl")
    parser.add_argument("--val-data", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gpu-type", default=DEFAULT_GPU)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--pod-name", default=DEFAULT_POD_NAME)
    parser.add_argument("--existing-pod-id")
    parser.add_argument("--bootstrap-status-path", default=str(DEFAULT_BOOTSTRAP_STATUS_PATH))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--require-hf-token", action="store_true")
    parser.add_argument(
        "--inventory-snapshot-path",
        help="Optional path to a machine-readable RunPod inventory snapshot captured out-of-band.",
    )
    parser.add_argument(
        "--replace-pending-runtime",
        action="store_true",
        help="If the selected bootstrap pod has been stuck without runtime metadata beyond the threshold, terminate it and create a replacement pod via the RunPod API.",
    )
    parser.add_argument(
        "--pending-runtime-threshold-seconds",
        type=int,
        default=1800,
        help="Minimum age for a running-without-runtime pod before --replace-pending-runtime can recycle it.",
    )
    parser.add_argument(
        "--replace-unreachable-ssh",
        action="store_true",
        help="If the selected bootstrap pod still refuses SSH beyond the threshold, terminate it and create a replacement pod via the RunPod API.",
    )
    parser.add_argument(
        "--unreachable-ssh-threshold-seconds",
        type=int,
        default=600,
        help="Minimum age for a running pod with refused SSH before --replace-unreachable-ssh can recycle it.",
    )
    parser.add_argument("--cloud-type", default="SECURE")
    parser.add_argument("--volume-gb", type=int, default=80)
    parser.add_argument("--remote-adapter-dir", default=DEFAULT_REMOTE_ADAPTER_DIR)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Skip launch and only download/validate an existing adapter bundle from --remote-adapter-dir.",
    )
    parser.add_argument(
        "--refresh-inventory-snapshot",
        action="store_true",
        help="Refresh --inventory-snapshot-path from the live RunPod GraphQL API and exit.",
    )
    args = parser.parse_args()

    if args.refresh_inventory_snapshot:
        if not args.inventory_snapshot_path:
            raise SystemExit("--refresh-inventory-snapshot requires --inventory-snapshot-path")
        snapshot = refresh_inventory_snapshot(Path(args.inventory_snapshot_path))
        print(json.dumps(snapshot, indent=2))
        return

    run_job(
        data_path=args.data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        gpu_type=args.gpu_type,
        base_model=args.base_model,
        pod_name=args.pod_name,
        existing_pod_id=args.existing_pod_id,
        bootstrap_status_path=args.bootstrap_status_path,
        epochs=args.epochs,
        lora_rank=args.lora_rank,
        keep_pod=args.keep_pod,
        bootstrap_only=args.bootstrap_only,
        require_hf_token=args.require_hf_token,
        inventory_snapshot_path=args.inventory_snapshot_path,
        replace_pending_runtime=args.replace_pending_runtime,
        pending_runtime_threshold_seconds=args.pending_runtime_threshold_seconds,
        replace_unreachable_ssh=args.replace_unreachable_ssh,
        unreachable_ssh_threshold_seconds=args.unreachable_ssh_threshold_seconds,
        cloud_type=args.cloud_type,
        volume_gb=args.volume_gb,
        remote_adapter_dir=args.remote_adapter_dir,
        download_only=args.download_only,
    )


if __name__ == "__main__":
    main()
