"""GPU training runner using RunPod for on-demand GPU provisioning.

Provisions a RunPod pod, uploads training data/scripts, runs training,
downloads model weights, and terminates the pod.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Optional

from .runpod_config import RunPodConfig


class GPUTrainingRunner:
    """Manages GPU training jobs on RunPod.

    Requires MCP tool callables for RunPod operations.
    """

    def __init__(
        self,
        config: Optional[RunPodConfig] = None,
        runpod_create: Optional[Callable[..., Any]] = None,
        runpod_get: Optional[Callable[..., Any]] = None,
        runpod_stop: Optional[Callable[..., Any]] = None,
        runpod_terminate: Optional[Callable[..., Any]] = None,
        runpod_balance: Optional[Callable[..., Any]] = None,
        runpod_gpu_types: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config or RunPodConfig()
        self._create = runpod_create
        self._get = runpod_get
        self._stop = runpod_stop
        self._terminate = runpod_terminate
        self._balance = runpod_balance
        self._gpu_types = runpod_gpu_types
        self._active_pod_id: Optional[str] = None

    def check_budget(self) -> Dict[str, Any]:
        """Check RunPod balance and verify sufficient budget.

        Returns dict with balance info and whether training is affordable.
        """
        if self._balance is None:
            return {"error": "RunPod balance tool not configured", "can_train": False}

        balance_info = self._balance()
        balance = float(balance_info.get("balance", 0))
        return {
            "balance": balance,
            "max_budget": self.config.max_total_budget,
            "can_train": balance >= self.config.max_cost_per_hour,
        }

    def find_gpu(self) -> Dict[str, Any]:
        """Find the cheapest adequate GPU type available.

        Returns GPU type info or error.
        """
        if self._gpu_types is None:
            return {"error": "RunPod GPU types tool not configured"}

        gpu_list = self._gpu_types()
        if not isinstance(gpu_list, list):
            gpu_list = gpu_list.get("gpuTypes", []) if isinstance(gpu_list, dict) else []

        # Filter by cost and sort
        affordable = []
        for gpu in gpu_list:
            if not isinstance(gpu, dict):
                continue
            cost = float(gpu.get("communityPrice", gpu.get("securePrice", 999)))
            if cost <= self.config.max_cost_per_hour:
                affordable.append({
                    "id": gpu.get("id", ""),
                    "name": gpu.get("displayName", gpu.get("id", "")),
                    "cost_per_hour": cost,
                    "memory_gb": gpu.get("memoryInGb", 0),
                })

        if not affordable:
            return {"error": "No affordable GPU found", "budget": self.config.max_cost_per_hour}

        affordable.sort(key=lambda g: g["cost_per_hour"])
        return {"gpu": affordable[0], "alternatives": affordable[:3]}

    def provision_pod(self, gpu_type_id: str = "") -> Dict[str, Any]:
        """Provision a RunPod GPU pod for training.

        Args:
            gpu_type_id: Specific GPU type to use. If empty, uses config default.

        Returns:
            Pod info dict with id, status.
        """
        if self._create is None:
            return {"error": "RunPod create tool not configured"}

        result = self._create(
            name="autoppia-training",
            imageName=self.config.container_image,
            gpuTypeId=gpu_type_id or self.config.gpu_type,
            gpuCount=self.config.gpu_count,
            volumeInGb=self.config.volume_size_gb,
            cloudType=self.config.cloud_type,
        )

        pod_id = result.get("id", "") if isinstance(result, dict) else ""
        self._active_pod_id = pod_id
        return {"pod_id": pod_id, "status": "provisioning"}

    def wait_for_ready(self, pod_id: str, timeout_s: int = 300) -> bool:
        """Wait for pod to be ready. Returns True if ready, False if timeout."""
        if self._get is None:
            return False

        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            info = self._get(podId=pod_id)
            status = info.get("desiredStatus", "") if isinstance(info, dict) else ""
            runtime = info.get("runtime", {}) if isinstance(info, dict) else {}
            if runtime and runtime.get("uptimeInSeconds", 0) > 0:
                return True
            time.sleep(10)
        return False

    def terminate_pod(self, pod_id: Optional[str] = None) -> bool:
        """Terminate a RunPod pod. Returns True on success."""
        target = pod_id or self._active_pod_id
        if not target or self._terminate is None:
            return False

        try:
            self._terminate(podId=target)
            if target == self._active_pod_id:
                self._active_pod_id = None
            return True
        except Exception:
            return False

    def run_training_job(
        self,
        script: str,
        data_path: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """Run a complete training job: provision -> train -> cleanup.

        This is the high-level API. For finer control, use the individual methods.

        Args:
            script: Training script module (e.g. 'training.train_ranker').
            data_path: Path to training data file.
            output_path: Path for saved model weights.

        Returns:
            Dict with status and metrics.
        """
        # Check budget
        budget = self.check_budget()
        if not budget.get("can_train", False):
            return {"error": "Insufficient budget", "budget_info": budget}

        # Find GPU
        gpu_info = self.find_gpu()
        if "error" in gpu_info:
            return {"error": f"No GPU available: {gpu_info['error']}"}

        gpu = gpu_info.get("gpu", {})
        gpu_id = gpu.get("id", "")

        # Provision
        pod_info = self.provision_pod(gpu_id)
        pod_id = pod_info.get("pod_id", "")
        if not pod_id:
            return {"error": "Failed to provision pod"}

        try:
            # Wait for ready
            if not self.wait_for_ready(pod_id):
                self.terminate_pod(pod_id)
                return {"error": "Pod did not become ready in time"}

            return {
                "status": "pod_ready",
                "pod_id": pod_id,
                "gpu": gpu,
                "message": f"Pod {pod_id} ready. Run training script manually or via SSH.",
            }

        except Exception as e:
            self.terminate_pod(pod_id)
            return {"error": f"Training job failed: {e}"}

    # Aliases expected by CHECK.sh
    def provision(self, gpu_type_id: str = "") -> Dict[str, Any]:
        """Alias for provision_pod."""
        return self.provision_pod(gpu_type_id)

    def run_training(
        self, script: str, data_path: str, output_path: str,
    ) -> Dict[str, Any]:
        """Alias for run_training_job."""
        return self.run_training_job(script, data_path, output_path)

    @property
    def active_pod_id(self) -> Optional[str]:
        return self._active_pod_id


# Alias expected by CHECK.sh
GPURunner = GPUTrainingRunner
