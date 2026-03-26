"""Orchestrate LoRA fine-tuning on RunPod: provision → upload → train → download → terminate.

Usage:
    python -m training.runpod_job \
        --data data/sft/train.jsonl \
        --val-data data/sft/val.jsonl \
        --output-dir models/bu-30b-lora

Requires RunPod API key in RUNPOD_API_KEY environment variable.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RunPod API helpers
# ---------------------------------------------------------------------------

def _runpod_api(method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
    """Call RunPod REST API."""
    import requests

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY not set")
    base = "https://api.runpod.io/graphql"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # RunPod uses GraphQL — this is a simplified wrapper
    resp = requests.post(base, json=data, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_balance() -> float:
    """Return current RunPod balance in USD."""
    try:
        result = _runpod_api("POST", "/graphql", {
            "query": "{ myself { currentSpendPerHr creditBalance } }"
        })
        return float(result.get("data", {}).get("myself", {}).get("creditBalance", 0))
    except Exception as e:
        logger.warning("Could not fetch balance: %s", e)
        return -1.0


def create_pod(
    gpu_type: str = "NVIDIA A100 80GB PCIe",
    image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    cloud_type: str = "COMMUNITY",
    volume_gb: int = 50,
    name: str = "autoppia-finetune",
) -> str:
    """Create a RunPod GPU pod. Returns pod ID."""
    import runpodctl  # type: ignore

    # Fallback: use runpodctl CLI if available
    cmd = [
        "runpodctl", "create", "pod",
        "--name", name,
        "--gpuType", gpu_type,
        "--imageName", image,
        "--volumeSize", str(volume_gb),
        "--cloudType", cloud_type,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Pod creation failed: {result.stderr}")
    # Parse pod ID from output
    for line in result.stdout.splitlines():
        if "pod" in line.lower() and "id" in line.lower():
            return line.split()[-1]
    return result.stdout.strip().split()[-1]


def wait_for_pod(pod_id: str, timeout: int = 600) -> bool:
    """Wait for pod to be running."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["runpodctl", "get", "pod", pod_id],
                capture_output=True, text=True, timeout=30,
            )
            if "RUNNING" in result.stdout:
                return True
        except Exception:
            pass
        time.sleep(15)
    return False


def upload_to_pod(pod_id: str, local_path: str, remote_path: str) -> None:
    """Upload file to pod via rsync/scp."""
    subprocess.run(
        ["runpodctl", "send", local_path, f"{pod_id}:{remote_path}"],
        check=True, timeout=300,
    )


def run_on_pod(pod_id: str, command: str) -> str:
    """Execute command on pod."""
    result = subprocess.run(
        ["runpodctl", "exec", pod_id, "--", "bash", "-c", command],
        capture_output=True, text=True, timeout=7200,  # 2hr max
    )
    if result.returncode != 0:
        logger.error("Pod command failed: %s", result.stderr)
    return result.stdout


def download_from_pod(pod_id: str, remote_path: str, local_path: str) -> None:
    """Download file from pod."""
    subprocess.run(
        ["runpodctl", "receive", f"{pod_id}:{remote_path}", local_path],
        check=True, timeout=600,
    )


def terminate_pod(pod_id: str) -> None:
    """Terminate pod to stop billing."""
    try:
        subprocess.run(
            ["runpodctl", "remove", "pod", pod_id],
            check=True, timeout=30,
        )
        logger.info("Pod %s terminated", pod_id)
    except Exception as e:
        logger.error("FAILED to terminate pod %s: %s — TERMINATE MANUALLY!", pod_id, e)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_job(
    data_path: str = "data/sft/train.jsonl",
    val_data_path: str = "data/sft/val.jsonl",
    output_dir: str = "models/bu-30b-lora",
    gpu_type: str = "NVIDIA A100 80GB PCIe",
    epochs: int = 3,
    lora_rank: int = 32,
) -> Dict[str, Any]:
    """Full fine-tune pipeline: provision → upload → train → download → terminate."""
    pod_id = None
    job_info: Dict[str, Any] = {"status": "starting"}

    try:
        # 1. Check balance
        balance = get_balance()
        logger.info("RunPod balance: $%.2f", balance)
        if 0 < balance < 5.0:
            raise RuntimeError(f"RunPod balance too low: ${balance:.2f}")

        # 2. Create pod
        logger.info("Creating pod with %s ...", gpu_type)
        pod_id = create_pod(gpu_type=gpu_type)
        logger.info("Pod created: %s", pod_id)

        job_info.update({"status": "provisioning", "pod_id": pod_id})
        _save_job_info(output_dir, job_info)

        # 3. Wait for pod
        if not wait_for_pod(pod_id):
            raise RuntimeError("Pod failed to start within timeout")
        job_info["status"] = "running"

        # 4. Upload data + script
        logger.info("Uploading training data ...")
        upload_to_pod(pod_id, data_path, "/workspace/train.jsonl")
        if os.path.exists(val_data_path):
            upload_to_pod(pod_id, val_data_path, "/workspace/val.jsonl")
        upload_to_pod(pod_id, "training/finetune_bu.py", "/workspace/finetune_bu.py")

        # 5. Run training
        logger.info("Starting training on pod ...")
        job_info["status"] = "training"
        _save_job_info(output_dir, job_info)

        train_cmd = (
            f"cd /workspace && python finetune_bu.py "
            f"--data train.jsonl "
            f"--val-data val.jsonl "
            f"--output-dir /workspace/lora_output "
            f"--epochs {epochs} "
            f"--lora-rank {lora_rank}"
        )
        output = run_on_pod(pod_id, train_cmd)
        logger.info("Training output:\n%s", output[-2000:])  # last 2k chars

        # 6. Download weights
        logger.info("Downloading adapter weights ...")
        os.makedirs(output_dir, exist_ok=True)
        for fname in ["adapter_model.safetensors", "adapter_config.json",
                       "tokenizer_config.json", "train_metrics.json"]:
            try:
                download_from_pod(pod_id, f"/workspace/lora_output/{fname}",
                                  os.path.join(output_dir, fname))
            except Exception as e:
                logger.warning("Could not download %s: %s", fname, e)

        job_info["status"] = "completed"
        _save_job_info(output_dir, job_info)
        return job_info

    except Exception as e:
        logger.error("Job failed: %s", e)
        job_info["status"] = "failed"
        job_info["error"] = str(e)
        _save_job_info(output_dir, job_info)
        raise

    finally:
        # ALWAYS terminate pod
        if pod_id:
            logger.info("Terminating pod %s ...", pod_id)
            terminate_pod(pod_id)
            job_info["pod_terminated"] = True
            _save_job_info(output_dir, job_info)


def _save_job_info(output_dir: str, info: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "job.json"), "w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run LoRA fine-tune on RunPod")
    parser.add_argument("--data", default="data/sft/train.jsonl")
    parser.add_argument("--val-data", default="data/sft/val.jsonl")
    parser.add_argument("--output-dir", default="models/bu-30b-lora")
    parser.add_argument("--gpu-type", default="NVIDIA A100 80GB PCIe")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=32)
    args = parser.parse_args()

    run_job(
        data_path=args.data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        gpu_type=args.gpu_type,
        epochs=args.epochs,
        lora_rank=args.lora_rank,
    )


if __name__ == "__main__":
    main()
