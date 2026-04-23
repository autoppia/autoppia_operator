from __future__ import annotations

from collections.abc import Callable

from training.deterministic_harvester.builders.autobooks import AUTOBOOKS_PLAN_BUILDERS, build_autobooks_plan
from training.deterministic_harvester.builders.autocalendar import AUTOCALENDAR_PLAN_BUILDERS, build_autocalendar_plan
from training.deterministic_harvester.builders.autocinema import AUTOCINEMA_PLAN_BUILDERS, build_autocinema_plan
from training.deterministic_harvester.builders.autoconnect import AUTOCONNECT_PLAN_BUILDERS, build_autoconnect_plan
from training.deterministic_harvester.builders.autocrm import AUTOCRM_PLAN_BUILDERS, build_autocrm_plan
from training.deterministic_harvester.builders.autodelivery import AUTODELIVERY_PLAN_BUILDERS, build_autodelivery_plan
from training.deterministic_harvester.builders.autodining import AUTODINING_PLAN_BUILDERS, build_autodining_plan
from training.deterministic_harvester.builders.autolist import AUTOLIST_PLAN_BUILDERS, build_autolist_plan
from training.deterministic_harvester.builders.autolodge import AUTOLODGE_PLAN_BUILDERS, build_autolodge_plan
from training.deterministic_harvester.builders.automail import AUTOMAIL_PLAN_BUILDERS, build_automail_plan
from training.deterministic_harvester.builders.autowork import AUTOWORK_PLAN_BUILDERS, build_autowork_plan
from training.deterministic_harvester.builders.autozone import AUTOZONE_PLAN_BUILDERS, build_autozone_plan
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.normalizer import DeterministicTaskObjective

DeterministicPlanBuilder = Callable[[DeterministicTaskObjective], DeterministicPlan]

DETERMINISTIC_PLAN_BUILDERS: dict[tuple[str, str], DeterministicPlanBuilder] = {
    **{("autocinema", use_case): build_autocinema_plan for use_case in AUTOCINEMA_PLAN_BUILDERS},
    **{("autobooks", use_case): build_autobooks_plan for use_case in AUTOBOOKS_PLAN_BUILDERS},
    **{("autodining", use_case): build_autodining_plan for use_case in AUTODINING_PLAN_BUILDERS},
    **{("autocalendar", use_case): build_autocalendar_plan for use_case in AUTOCALENDAR_PLAN_BUILDERS},
    **{("autocrm", use_case): build_autocrm_plan for use_case in AUTOCRM_PLAN_BUILDERS},
    **{("automail", use_case): build_automail_plan for use_case in AUTOMAIL_PLAN_BUILDERS},
    **{("autolist", use_case): build_autolist_plan for use_case in AUTOLIST_PLAN_BUILDERS},
    **{("autolodge", use_case): build_autolodge_plan for use_case in AUTOLODGE_PLAN_BUILDERS},
    **{("autodelivery", use_case): build_autodelivery_plan for use_case in AUTODELIVERY_PLAN_BUILDERS},
    **{("autowork", use_case): build_autowork_plan for use_case in AUTOWORK_PLAN_BUILDERS},
    **{("autoconnect", use_case): build_autoconnect_plan for use_case in AUTOCONNECT_PLAN_BUILDERS},
    **{("autozone", use_case): build_autozone_plan for use_case in AUTOZONE_PLAN_BUILDERS},
}


def build_registered_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    project_id = str(objective.web_project_id or "").strip().lower() or "autocinema"
    use_case = str(objective.use_case or "").strip().upper()
    builder = DETERMINISTIC_PLAN_BUILDERS.get((project_id, use_case))
    if builder is None:
        raise ValueError(f"Unsupported deterministic use case: {use_case}")
    return builder(objective)


__all__ = ["DETERMINISTIC_PLAN_BUILDERS", "DeterministicPlanBuilder", "build_registered_plan"]
