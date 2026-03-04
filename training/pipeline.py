from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import split_train_val, write_jsonl
from .iwap_client import IWAPClient
from .models import TrajectoryRecord
from .normalize import dedupe_trajectories, extract_task_payload, normalize_trajectory
from .s3_source import S3ObjectRef, S3TrajectorySource


@dataclass
class TrajectoryBuildConfig:
    min_eval_score: float = 0.5
    min_actions: int = 1
    max_actions: int = 80
    max_steps: int = 300
    keep_html: bool = False
    dedupe_actions: bool = True
    dedupe_trajectories: bool = True
    only_successful: bool = True


@dataclass
class IngestionStats:
    started_at: str
    source: str
    runs_seen: int = 0
    tasks_seen: int = 0
    tasks_with_payload_url: int = 0
    payload_downloaded: int = 0
    payload_parse_errors: int = 0
    trajectories_kept: int = 0
    trajectories_filtered: int = 0
    trajectories_deduped: int = 0
    finished_at: str | None = None


@dataclass
class DatasetArtifacts:
    output_dir: str
    cleaned_trajectories: str
    sft_all: str
    sft_train: str
    sft_val: str
    ppo_bootstrap: str
    manifest: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_map(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for t in tasks:
        task_id = t.get("taskId") or t.get("task_id")
        if task_id:
            by_id[str(task_id)] = t
    return by_id


def _task_payload_urls(logs: list[dict[str, Any]]) -> dict[str, str]:
    by_task: dict[str, str] = {}
    for entry in logs:
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        task_id = meta.get("taskId") or meta.get("task_id")
        payload_url = meta.get("payloadUrl") or meta.get("payload_url")
        if task_id and payload_url:
            by_task[str(task_id)] = str(payload_url)
    return by_task


def _keep_trajectory(trajectory: dict[str, Any], *, cfg: TrajectoryBuildConfig) -> bool:
    is_success = bool((trajectory.get("summary") or {}).get("success"))
    if cfg.only_successful and not is_success:
        return False
    actions_count = len(trajectory.get("actions") or [])
    return actions_count >= int(cfg.min_actions)


def _normalized_to_record(normalized: dict[str, Any]) -> TrajectoryRecord:
    return TrajectoryRecord.from_dict(normalized)


def build_ppo_bootstrap_rows(records: list[TrajectoryRecord], *, terminal_bonus: float = 1.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if not rec.steps:
            continue
        steps_total = max(1, len(rec.steps))
        dense_reward = float(rec.summary.eval_score) / float(steps_total)

        for idx, step in enumerate(rec.steps):
            done = idx == (len(rec.steps) - 1)
            reward = dense_reward
            if done and rec.summary.success:
                reward += float(terminal_bonus)

            action_payload = step.action.to_dict() if step.action else None
            if action_payload is None and idx < len(rec.actions):
                action_payload = rec.actions[idx].to_dict()

            obs_prompt = str(step.agent_input.get("prompt") or rec.task.prompt)
            obs_url = str(step.agent_input.get("current_url") or rec.task.url)

            rows.append(
                {
                    "trajectory_id": rec.trajectory_id,
                    "task_id": rec.task_id,
                    "step_index": idx,
                    "observation": {
                        "prompt": obs_prompt,
                        "url": obs_url,
                        "step_index": idx,
                    },
                    "action": action_payload,
                    "reward": float(reward),
                    "done": bool(done),
                    "success": bool(rec.summary.success if done else False),
                    "eval_score": float(rec.summary.eval_score if done else 0.0),
                }
            )
    return rows


def ingest_from_iwap_api(
    *,
    base_url: str,
    token: str | None,
    max_runs: int | None,
    page_limit: int,
    include_unfinished: bool,
    cfg: TrajectoryBuildConfig,
) -> tuple[list[TrajectoryRecord], IngestionStats]:
    stats = IngestionStats(started_at=_now_iso(), source="iwap_api")
    kept: list[dict[str, Any]] = []

    with IWAPClient(base_url=base_url, token=token) as client:
        run_ids = list(
            client.iter_run_ids(
                max_runs=max_runs,
                page_limit=page_limit,
                include_unfinished=include_unfinished,
            )
        )
        stats.runs_seen = len(run_ids)

        for run_id in run_ids:
            try:
                tasks = client.get_run_tasks(run_id)
                logs = client.get_run_logs(run_id)
            except Exception:
                continue

            by_task = _task_map(tasks)
            payload_urls = _task_payload_urls(logs)
            stats.tasks_seen += len(by_task)
            stats.tasks_with_payload_url += len(payload_urls)

            for task_id, payload_url in payload_urls.items():
                task_meta = by_task.get(task_id, {})
                try:
                    raw_payload = client.fetch_task_log_payload(payload_url)
                    stats.payload_downloaded += 1
                except Exception:
                    stats.payload_parse_errors += 1
                    continue

                payload = extract_task_payload(raw_payload)
                if not payload:
                    stats.payload_parse_errors += 1
                    continue

                normalized = normalize_trajectory(
                    payload=payload,
                    run_id=run_id,
                    source_url=payload_url,
                    task_meta=task_meta,
                    keep_html=bool(cfg.keep_html),
                    min_eval_score=float(cfg.min_eval_score),
                    max_steps=int(cfg.max_steps),
                    max_actions=int(cfg.max_actions),
                    dedupe_actions=bool(cfg.dedupe_actions),
                )

                if _keep_trajectory(normalized, cfg=cfg):
                    kept.append(normalized)
                    stats.trajectories_kept += 1
                else:
                    stats.trajectories_filtered += 1

    if cfg.dedupe_trajectories:
        before = len(kept)
        kept = dedupe_trajectories(kept)
        stats.trajectories_deduped = before - len(kept)

    stats.finished_at = _now_iso()
    return [_normalized_to_record(x) for x in kept], stats


def ingest_from_s3(
    *,
    source: S3TrajectorySource,
    cfg: TrajectoryBuildConfig,
    max_objects: int | None = None,
) -> tuple[list[TrajectoryRecord], IngestionStats]:
    stats = IngestionStats(started_at=_now_iso(), source="s3")
    kept: list[dict[str, Any]] = []

    object_refs = list(source.iter_objects(max_objects=max_objects))
    stats.runs_seen = len(object_refs)

    for idx, ref in enumerate(object_refs, start=1):
        stats.tasks_seen += 1
        stats.tasks_with_payload_url += 1

        try:
            raw = source.fetch_json(ref)
            stats.payload_downloaded += 1
        except Exception:
            stats.payload_parse_errors += 1
            continue

        payload = extract_task_payload(raw)
        if not payload:
            stats.payload_parse_errors += 1
            continue

        run_id = str(
            raw.get("run_id")
            or raw.get("runId")
            or payload.get("run_id")
            or payload.get("agent_run_id")
            or f"s3-{idx:06d}"
        )
        task_meta = raw.get("task") if isinstance(raw.get("task"), dict) else None

        normalized = normalize_trajectory(
            payload=payload,
            run_id=run_id,
            source_url=ref.uri,
            task_meta=task_meta if isinstance(task_meta, dict) else None,
            keep_html=bool(cfg.keep_html),
            min_eval_score=float(cfg.min_eval_score),
            max_steps=int(cfg.max_steps),
            max_actions=int(cfg.max_actions),
            dedupe_actions=bool(cfg.dedupe_actions),
        )

        if _keep_trajectory(normalized, cfg=cfg):
            kept.append(normalized)
            stats.trajectories_kept += 1
        else:
            stats.trajectories_filtered += 1

    if cfg.dedupe_trajectories:
        before = len(kept)
        kept = dedupe_trajectories(kept)
        stats.trajectories_deduped = before - len(kept)

    stats.finished_at = _now_iso()
    return [_normalized_to_record(x) for x in kept], stats


def export_training_bundle(
    *,
    out_dir: Path,
    records: list[TrajectoryRecord],
    stats: IngestionStats,
    val_ratio: float,
    seed: int,
    system_prompt: str | None,
) -> DatasetArtifacts:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_rows = [r.to_dict() for r in records]
    sft_rows = [r.to_sft_record(system_prompt=system_prompt) for r in records]
    train_rows, val_rows = split_train_val(sft_rows, val_ratio=val_ratio, seed=seed)
    ppo_rows = build_ppo_bootstrap_rows(records)

    cleaned_path = out_dir / "cleaned_trajectories.jsonl"
    sft_all_path = out_dir / "sft_all.jsonl"
    sft_train_path = out_dir / "sft_train.jsonl"
    sft_val_path = out_dir / "sft_val.jsonl"
    ppo_path = out_dir / "ppo_bootstrap.jsonl"

    write_jsonl(cleaned_path, cleaned_rows)
    write_jsonl(sft_all_path, sft_rows)
    write_jsonl(sft_train_path, train_rows)
    write_jsonl(sft_val_path, val_rows)
    write_jsonl(ppo_path, ppo_rows)

    manifest = {
        "stats": {
            **asdict(stats),
            "records_total": len(records),
            "sft_total": len(sft_rows),
            "sft_train": len(train_rows),
            "sft_val": len(val_rows),
            "ppo_steps": len(ppo_rows),
        },
        "files": {
            "cleaned_trajectories": str(cleaned_path),
            "sft_all": str(sft_all_path),
            "sft_train": str(sft_train_path),
            "sft_val": str(sft_val_path),
            "ppo_bootstrap": str(ppo_path),
        },
        "exported_at": _now_iso(),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return DatasetArtifacts(
        output_dir=str(out_dir),
        cleaned_trajectories=str(cleaned_path),
        sft_all=str(sft_all_path),
        sft_train=str(sft_train_path),
        sft_val=str(sft_val_path),
        ppo_bootstrap=str(ppo_path),
        manifest=str(manifest_path),
    )


__all__ = [
    "DatasetArtifacts",
    "IngestionStats",
    "TrajectoryBuildConfig",
    "build_ppo_bootstrap_rows",
    "export_training_bundle",
    "ingest_from_iwap_api",
    "ingest_from_s3",
]
