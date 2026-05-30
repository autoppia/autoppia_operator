from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrajectoryCandidate:
    use_case: str
    seed: int
    attempt_name: str
    teacher_model: str
    generation_mode: str
    brief_path: str
    prompt_lines: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_case": self.use_case,
            "seed": int(self.seed),
            "attempt_name": self.attempt_name,
            "teacher_model": self.teacher_model,
            "generation_mode": self.generation_mode,
            "brief_path": self.brief_path,
            "prompt_lines": list(self.prompt_lines),
            "actions": [dict(action) for action in self.actions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrajectoryCandidate:
        if not isinstance(payload, dict):
            raise TypeError("candidate payload must be a dict")
        actions = payload.get("actions")
        prompt_lines = payload.get("prompt_lines")
        return cls(
            use_case=str(payload.get("use_case") or "").strip().upper(),
            seed=int(payload.get("seed") or 0),
            attempt_name=str(payload.get("attempt_name") or "").strip(),
            teacher_model=str(payload.get("teacher_model") or "").strip(),
            generation_mode=str(payload.get("generation_mode") or "").strip(),
            brief_path=str(payload.get("brief_path") or "").strip(),
            prompt_lines=tuple(str(item).strip() for item in (prompt_lines or []) if str(item).strip()) if isinstance(prompt_lines, list) else (),
            actions=tuple(dict(item) for item in (actions or []) if isinstance(item, dict)) if isinstance(actions, list) else (),
            metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
        )


def candidate_path(*, output_root: Path, seed: int, attempt_name: str) -> Path:
    out_dir = output_root / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_attempt = str(attempt_name or "candidate").strip().replace("/", "_")
    return out_dir / f"seed_{int(seed):04d}_{safe_attempt}.json"


def write_candidate(path: Path, candidate: TrajectoryCandidate) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_candidate(path: Path) -> TrajectoryCandidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TrajectoryCandidate.from_dict(payload)
