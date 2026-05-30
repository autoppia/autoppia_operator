"""
IWA-based deterministic plans with the same *shape* as :mod:`autocinema`

Each web project exports:

- ``<PROJECT>_PLAN_BUILDERS`` — ``use_case`` → callable(objective) → list of action dicts
- ``build_<project>_plan`` — wraps those actions in :class:`DeterministicPlan` (like ``build_autocinema_plan``)

Actions are built from IWA ``trajectories.py`` and passed through
:func:`training.deterministic_harvester.trajectory_selectors.planned_actions_for_iwa_use_case_enriched`
so ``selector_candidates`` get the same id/class/text expansion as
:mod:`training.deterministic_harvester.selectors` for that ``web_project_id``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from autoppia_iwa.src.demo_webs.trajectory_registry import get_trajectory_map

import training._iwa_path  # noqa: F401
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.normalizer import DeterministicTaskObjective
from training.deterministic_harvester.trajectory_selectors import (
    list_iwa_use_cases,
    planned_actions_for_iwa_use_case_enriched,
)

PlanActionBuilder = Callable[[DeterministicTaskObjective], list[dict[str, Any]]]


def build_iwa_enriched_action_list(
    project_id: str,
    use_case: str,
    objective: DeterministicTaskObjective,
) -> list[dict[str, Any]]:
    """Trajectory actions for one use case with variant-expanded ``selector_candidates``."""
    u = str(use_case or "").strip().upper()
    actions: list[dict[str, Any]] = [
        dict(s)
        for s in planned_actions_for_iwa_use_case_enriched(
            str(project_id or "").strip(),
            u,
            task_url=None,
            seed=objective.seed,
        )
    ]
    if actions and str(actions[0].get("type") or "").strip() == "NavigateAction" and str(objective.task_url or "").strip():
        first = dict(actions[0])
        first["url"] = str(objective.task_url).strip()
        actions[0] = first
    return actions


def _make_action_builder_for_use_case(project_id: str, use_case: str) -> PlanActionBuilder:
    def _build(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
        return build_iwa_enriched_action_list(project_id, use_case, objective)

    return _build


@lru_cache(maxsize=32)
def iwa_enriched_action_builders(project_id: str) -> dict[str, PlanActionBuilder]:
    """``AUTOCINEMA_PLAN_BUILDERS``-style map: IWA use case name → action list builder."""
    pid = str(project_id or "").strip().lower()
    return {uc: _make_action_builder_for_use_case(pid, uc) for uc in sorted(list_iwa_use_cases(pid))}


def iwa_enriched_action_builder(project_id: str, use_case: str) -> PlanActionBuilder | None:
    return iwa_enriched_action_builders(project_id).get(str(use_case or "").strip().upper())


def _trajectory_prompt(project_id: str, use_case: str) -> str:
    m = get_trajectory_map(str(project_id or "").strip()) or {}
    t = m.get(str(use_case or "").strip().upper())
    if t is None:
        return ""
    return str(getattr(t, "prompt", "") or "")


def build_iwa_enriched_deterministic_plan(
    project_id: str,
    objective: DeterministicTaskObjective,
    *,
    iwa_source_label: str,
    prompt_preamble: str,
) -> DeterministicPlan:
    """
    :param iwa_source_label: Stored in ``metadata['source']`` (e.g. ``\"iwa_p04_autodining\"``).
    :param prompt_preamble: First prompt line is ``f\"{prompt_preamble} {use_case}\"`` (matches legacy IWA builders).
    """
    pid = str(project_id or "").strip().lower()
    oid = str(objective.web_project_id or "").strip().lower()
    if oid != pid:
        raise ValueError(f"Expected web_project_id={pid!r}, got {objective.web_project_id!r}")
    use_case = str(objective.use_case or "").strip().upper()
    builder = iwa_enriched_action_builder(pid, use_case)
    if builder is None:
        raise ValueError(f"Unsupported deterministic use case: {use_case}")
    actions = builder(objective)
    prompt = _trajectory_prompt(pid, use_case)
    first_line = f"{str(prompt_preamble or '').rstrip()} {use_case}".strip()
    return DeterministicPlan(
        prompt_lines=(first_line, f"seed={objective.seed}", prompt),
        actions=tuple(actions),
        metadata={
            "source": str(iwa_source_label or "").strip(),
            "web_project_id": pid,
            "trajectory_prompt": prompt,
            "iwa_use_case": use_case,
            "selectors": "iwa_enriched",
            "selector_strategy": "semantic_first_with_xpath_fallback",
            "auth_required": bool(objective.auth_required),
            "field_values": dict(objective.field_values),
            "entity_filters": dict(objective.entity_filters),
            "route_target": objective.route_target,
        },
    )


__all__ = [
    "PlanActionBuilder",
    "build_iwa_enriched_action_list",
    "build_iwa_enriched_deterministic_plan",
    "iwa_enriched_action_builder",
    "iwa_enriched_action_builders",
]
