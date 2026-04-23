from __future__ import annotations

from collections.abc import Callable

from training.deterministic_harvester.builders.autobooks import build_autobooks_plan, list_autobooks_iwa_use_cases
from training.deterministic_harvester.builders.autocinema import AUTOCINEMA_PLAN_BUILDERS, build_autocinema_plan
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.normalizer import DeterministicTaskObjective

DeterministicPlanBuilder = Callable[[DeterministicTaskObjective], DeterministicPlan]

DETERMINISTIC_PLAN_BUILDERS: dict[tuple[str, str], DeterministicPlanBuilder] = {
    **{("autocinema", use_case): build_autocinema_plan for use_case in AUTOCINEMA_PLAN_BUILDERS},
    **{("autobooks", use_case): build_autobooks_plan for use_case in list_autobooks_iwa_use_cases()},
}


def build_registered_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    project_id = str(objective.web_project_id or "").strip().lower() or "autocinema"
    use_case = str(objective.use_case or "").strip().upper()
    builder = DETERMINISTIC_PLAN_BUILDERS.get((project_id, use_case))
    if builder is None:
        raise ValueError(f"Unsupported deterministic use case: {use_case}")
    return builder(objective)


__all__ = ["DETERMINISTIC_PLAN_BUILDERS", "DeterministicPlanBuilder", "build_registered_plan"]
