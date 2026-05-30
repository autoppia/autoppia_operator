from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import Skill, SkillRegistry, SkillStep


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _load_skill(package_dir: Path, task: dict[str, Any]) -> Skill:
    trajectory_path = package_dir / str(task.get("trajectory") or "")
    trajectory = _read_json(trajectory_path) if trajectory_path.exists() else {}
    actions = trajectory.get("actions") if isinstance(trajectory.get("actions"), list) else []
    return Skill(
        id=str(task.get("id") or trajectory.get("task_id") or "").strip(),
        name=str(task.get("name") or task.get("id") or "").strip(),
        description=str(task.get("prompt") or "").strip(),
        success_criteria=str(task.get("success_criteria") or "").strip(),
        keywords=_tuple_of_strings(task.get("keywords")),
        prompts=_tuple_of_strings(task.get("prompts")),
        steps=tuple(SkillStep(action=dict(action)) for action in actions if isinstance(action, dict)),
    )


def load_skill_registry(package_dir: str | os.PathLike[str] | None = None) -> SkillRegistry:
    raw_path = str(package_dir or os.getenv("AUTOPPIA_CUSTOM_OPERATOR_PACKAGE") or "").strip()
    if not raw_path:
        return SkillRegistry()
    package_path = Path(raw_path).expanduser().resolve()
    operator_path = package_path / "operator.json"
    tasks_path = package_path / "tasks.json"
    if not operator_path.exists() or not tasks_path.exists():
        return SkillRegistry()

    operator = _read_json(operator_path)
    tasks = _read_json(tasks_path)
    skills = tuple(_load_skill(package_path, task) for task in tasks if isinstance(task, dict))
    defaults = operator.get("defaults") if isinstance(operator.get("defaults"), dict) else {}
    return SkillRegistry(
        operator_id=str(operator.get("id") or package_path.name),
        operator_name=str(operator.get("name") or package_path.name),
        base_url=str(operator.get("base_url") or ""),
        instructions=str(operator.get("instructions") or ""),
        defaults={str(k): str(v) for k, v in defaults.items()},
        skills=skills,
    )
