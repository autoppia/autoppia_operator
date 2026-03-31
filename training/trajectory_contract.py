from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_EPISODE_FIELDS = (
    "episode_task_id",
    "seed",
    "use_case",
    "success",
    "score",
    "result_path",
    "trace_dir",
    "trace_file",
    "attempt_name",
)


@dataclass(frozen=True)
class CanonicalEpisodePaths:
    result_path: Path
    trace_dir: Path
    trace_file: Path


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_provenance_fields(
    *,
    task_cache_path: str | Path | None = None,
    prompt_override: str | None = None,
    operator_version: str = "step_engine",
    policy_mode: str = "direct",
) -> dict[str, str]:
    task_cache_text = ""
    if task_cache_path:
        path = Path(task_cache_path)
        if path.exists():
            task_cache_text = path.read_text(encoding="utf-8")
    return {
        "operator_version": str(operator_version).strip() or "step_engine",
        "policy_mode": str(policy_mode).strip() or "direct",
        "task_cache_hash": hashlib.sha256(task_cache_text.encode("utf-8")).hexdigest(),
        "prompt_hash": hashlib.sha256(str(prompt_override or "").encode("utf-8")).hexdigest(),
    }


def row_identity(row: dict[str, Any]) -> tuple[int, str]:
    return int(row.get("seed") or 0), str(row.get("episode_task_id") or "")


def canonical_paths_for_row(*, gold_root: Path, row: dict[str, Any]) -> CanonicalEpisodePaths:
    seed = int(row.get("seed") or 0)
    attempt_name = str(row.get("attempt_name") or "").strip()
    episode_task_id = str(row.get("episode_task_id") or "").strip()
    trace_dir = gold_root / "traces" / f"seed_{seed:04d}_{attempt_name}"
    result_path = gold_root / "runs" / f"seed_{seed:04d}_{attempt_name}.json"
    trace_file = trace_dir / "episodes" / f"{episode_task_id}.json"
    return CanonicalEpisodePaths(
        result_path=result_path,
        trace_dir=trace_dir,
        trace_file=trace_file,
    )


def repair_episode_row_paths(*, gold_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(row)
    canonical = canonical_paths_for_row(gold_root=gold_root, row=repaired)
    repaired["result_path"] = str(canonical.result_path)
    repaired["trace_dir"] = str(canonical.trace_dir)
    repaired["trace_root"] = str(canonical.trace_dir)
    repaired["trace_file"] = str(canonical.trace_file)
    return repaired


def validate_episode_row(
    row: dict[str, Any],
    *,
    require_trace_files: bool = True,
    require_result_paths: bool = True,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(row, dict):
        return ["row_not_object"]

    for field in REQUIRED_EPISODE_FIELDS:
        value = row.get(field)
        if value in (None, "", []):
            issues.append(f"missing_field:{field}")

    episode_task_id = str(row.get("episode_task_id") or "")
    score = float(row.get("score") or 0.0)
    success = bool(row.get("success"))
    if episode_task_id and (" " in episode_task_id or "/" in episode_task_id):
        issues.append(f"episode_id_invalid:{episode_task_id}")
    if success is False or score < 1.0:
        issues.append(f"episode_not_gold:{episode_task_id or 'unknown'}")

    trace_dir = str(row.get("trace_dir") or "").strip()
    if trace_dir and not Path(trace_dir).exists():
        issues.append(f"trace_dir_missing:{episode_task_id or 'unknown'}")

    if require_trace_files:
        trace_file = str(row.get("trace_file") or "").strip()
        if not trace_file:
            issues.append(f"episode_missing_trace:{episode_task_id or 'unknown'}")
        elif not Path(trace_file).exists():
            issues.append(f"trace_missing:{episode_task_id or 'unknown'}")

    if require_result_paths:
        result_path = str(row.get("result_path") or "").strip()
        if not result_path:
            issues.append(f"result_missing:{episode_task_id or 'unknown'}")
        elif not Path(result_path).exists():
            issues.append(f"result_path_missing:{episode_task_id or 'unknown'}")

    for field in ("operator_version", "policy_mode", "task_cache_hash", "prompt_hash"):
        if str(row.get(field) or "").strip() == "":
            issues.append(f"provenance_missing:{field}")

    return issues


def canonical_row_fingerprint(row: dict[str, Any]) -> str:
    normalized = {
        key: row.get(key)
        for key in (
            "episode_task_id",
            "seed",
            "use_case",
            "success",
            "score",
            "attempt_name",
            "result_path",
            "trace_dir",
            "trace_file",
            "operator_version",
            "policy_mode",
            "task_cache_hash",
            "prompt_hash",
        )
    }
    return _json_digest(normalized)
