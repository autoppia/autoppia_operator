from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRACE_SECTIONS = ("before", "after", "act_request")
LARGE_INLINE_KEYS = ("screenshot", "snapshot_html")


@dataclass(frozen=True)
class TraceCompactionStats:
    files_rewritten: int = 0
    fields_removed: int = 0


def _compact_step_payload(step: dict[str, Any]) -> int:
    removed = 0
    for section_name in TRACE_SECTIONS:
        section = step.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in LARGE_INLINE_KEYS:
            if key in section:
                section.pop(key, None)
                removed += 1
    return removed


def compact_trace_dir(trace_dir: Path) -> TraceCompactionStats:
    files_rewritten = 0
    fields_removed = 0
    for path in trace_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = payload.get("steps") if isinstance(payload, dict) else None
        if not isinstance(steps, list):
            continue
        removed_here = 0
        for step in steps:
            if isinstance(step, dict):
                removed_here += _compact_step_payload(step)
        if removed_here <= 0:
            continue
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        files_rewritten += 1
        fields_removed += removed_here
    return TraceCompactionStats(files_rewritten=files_rewritten, fields_removed=fields_removed)
