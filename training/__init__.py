"""Training package exports with lazy loading.

This package is imported from lightweight helpers such as `training.format_for_sft`.
Avoid importing heavyweight ingestion/runtime modules at package import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "split_train_val": ("training.dataset", "split_train_val"),
    "write_jsonl": ("training.dataset", "write_jsonl"),
    "export_ppo_bootstrap": ("training.exporters", "export_ppo_bootstrap"),
    "export_sft": ("training.exporters", "export_sft"),
    "load_cleaned_trajectories": ("training.exporters", "load_cleaned_trajectories"),
    "IWAPClient": ("training.iwap_client", "IWAPClient"),
    "ActionRecord": ("training.models", "ActionRecord"),
    "StepRecord": ("training.models", "StepRecord"),
    "TaskInfo": ("training.models", "TaskInfo"),
    "TrajectoryRecord": ("training.models", "TrajectoryRecord"),
    "TrajectorySummary": ("training.models", "TrajectorySummary"),
    "build_sft_record": ("training.normalize", "build_sft_record"),
    "dedupe_trajectories": ("training.normalize", "dedupe_trajectories"),
    "extract_task_payload": ("training.normalize", "extract_task_payload"),
    "normalize_trajectory": ("training.normalize", "normalize_trajectory"),
    "DatasetArtifacts": ("training.pipeline", "DatasetArtifacts"),
    "IngestionStats": ("training.pipeline", "IngestionStats"),
    "TrajectoryBuildConfig": ("training.pipeline", "TrajectoryBuildConfig"),
    "export_training_bundle": ("training.pipeline", "export_training_bundle"),
    "ingest_from_iwap_api": ("training.pipeline", "ingest_from_iwap_api"),
    "ingest_from_s3": ("training.pipeline", "ingest_from_s3"),
    "S3ObjectRef": ("training.s3_source", "S3ObjectRef"),
    "S3TrajectorySource": ("training.s3_source", "S3TrajectorySource"),
    "decode_json_blob": ("training.s3_source", "decode_json_blob"),
    "parse_s3_uri": ("training.s3_source", "parse_s3_uri"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
