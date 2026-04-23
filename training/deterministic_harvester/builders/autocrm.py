"""Deterministic plans for `autocrm` sourced from IWA `p05_autocrm/trajectories.py`."""

from __future__ import annotations

from autoppia_iwa.src.demo_webs.trajectory_registry import get_trajectory_map

import training._iwa_path  # noqa: F401
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.iwa_planned_actions import (
    frontend_url_for_project,
    iwa_actions_to_planned_actions,
)
from training.deterministic_harvester.normalizer import DeterministicTaskObjective

_TRAJ_MAP: dict[str, object] | None = None


def _trajectory_map() -> dict[str, object]:
    global _TRAJ_MAP
    if _TRAJ_MAP is None:
        loaded = get_trajectory_map("autocrm")
        if not loaded:
            raise RuntimeError("IWA returned no trajectory map for autocrm; check p05_autocrm/trajectories.py")
        _TRAJ_MAP = dict(loaded)
    return _TRAJ_MAP  # type: ignore[return-value]


def list_autocrm_iwa_use_cases() -> frozenset[str]:
    return frozenset(_trajectory_map().keys())


def build_autocrm_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    if str(objective.web_project_id or "").strip().lower() != "autocrm":
        raise ValueError("build_autocrm_plan requires web_project_id=autocrm")
    use_case = str(objective.use_case or "").strip().upper()
    traj_map = _trajectory_map()
    traj = traj_map.get(use_case)
    if traj is None:
        raise ValueError(f"Unsupported deterministic use case: {use_case}")
    fe = frontend_url_for_project("autocrm")
    actions_raw = list(getattr(traj, "actions", None) or [])
    actions = iwa_actions_to_planned_actions(actions_raw, frontend_url=fe)
    if actions and str(actions[0].get("type") or "").strip() == "NavigateAction" and str(objective.task_url or "").strip():
        first = dict(actions[0])
        first["url"] = str(objective.task_url).strip()
        actions[0] = first
    prompt = str(getattr(traj, "prompt", "") or "")
    prompt_lines = (f"IWA p05 autocrm trajectory for {use_case}", f"seed={objective.seed}", prompt)
    return DeterministicPlan(
        prompt_lines=prompt_lines,
        actions=tuple(actions),
        metadata={
            "source": "iwa_p05_autocrm",
            "web_project_id": "autocrm",
            "trajectory_prompt": prompt,
            "iwa_use_case": use_case,
        },
    )


__all__ = [
    "build_autocrm_plan",
    "list_autocrm_iwa_use_cases",
]
