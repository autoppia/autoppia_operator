"""Configuration for RunPod GPU training infrastructure."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunPodConfig:
    """Configuration for RunPod GPU pod provisioning.

    Attributes:
        gpu_type: Preferred GPU type (e.g. 'NVIDIA RTX A4000').
        gpu_count: Number of GPUs per pod.
        container_image: Docker image for training.
        volume_size_gb: Persistent volume size.
        cloud_type: RunPod cloud type (COMMUNITY or SECURE).
        timeout_hours: Maximum pod lifetime before auto-terminate.
        min_download_speed: Minimum download speed in Mbps.
    """

    gpu_type: str = "NVIDIA RTX A4000"
    gpu_count: int = 1
    container_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
    volume_size_gb: int = 20
    cloud_type: str = "COMMUNITY"
    timeout_hours: float = 4.0
    min_download_speed: int = 500

    # Cost guardrails
    max_cost_per_hour: float = 1.0
    max_total_budget: float = 10.0
