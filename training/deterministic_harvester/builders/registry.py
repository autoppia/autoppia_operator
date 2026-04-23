from __future__ import annotations

from collections.abc import Callable

from training.deterministic_harvester.builders.autobooks import build_autobooks_plan, list_autobooks_iwa_use_cases
from training.deterministic_harvester.builders.autocalendar import build_autocalendar_plan, list_autocalendar_iwa_use_cases
from training.deterministic_harvester.builders.autocinema import AUTOCINEMA_PLAN_BUILDERS, build_autocinema_plan
from training.deterministic_harvester.builders.autoconnect import build_autoconnect_plan, list_autoconnect_iwa_use_cases
from training.deterministic_harvester.builders.autocrm import build_autocrm_plan, list_autocrm_iwa_use_cases
from training.deterministic_harvester.builders.autodelivery import build_autodelivery_plan, list_autodelivery_iwa_use_cases
from training.deterministic_harvester.builders.autodining import build_autodining_plan, list_autodining_iwa_use_cases
from training.deterministic_harvester.builders.autolist import build_autolist_plan, list_autolist_iwa_use_cases
from training.deterministic_harvester.builders.autolodge import build_autolodge_plan, list_autolodge_iwa_use_cases
from training.deterministic_harvester.builders.automail import build_automail_plan, list_automail_iwa_use_cases
from training.deterministic_harvester.builders.autowork import build_autowork_plan, list_autowork_iwa_use_cases
from training.deterministic_harvester.builders.autozone import build_autozone_plan, list_autozone_iwa_use_cases
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.normalizer import DeterministicTaskObjective

DeterministicPlanBuilder = Callable[[DeterministicTaskObjective], DeterministicPlan]

DETERMINISTIC_PLAN_BUILDERS: dict[tuple[str, str], DeterministicPlanBuilder] = {
    **{("autocinema", use_case): build_autocinema_plan for use_case in AUTOCINEMA_PLAN_BUILDERS},
    **{("autobooks", use_case): build_autobooks_plan for use_case in list_autobooks_iwa_use_cases()},
    **{("autodining", use_case): build_autodining_plan for use_case in list_autodining_iwa_use_cases()},
    **{("autocalendar", use_case): build_autocalendar_plan for use_case in list_autocalendar_iwa_use_cases()},
    **{("autocrm", use_case): build_autocrm_plan for use_case in list_autocrm_iwa_use_cases()},
    **{("automail", use_case): build_automail_plan for use_case in list_automail_iwa_use_cases()},
    **{("autolist", use_case): build_autolist_plan for use_case in list_autolist_iwa_use_cases()},
    **{("autolodge", use_case): build_autolodge_plan for use_case in list_autolodge_iwa_use_cases()},
    **{("autodelivery", use_case): build_autodelivery_plan for use_case in list_autodelivery_iwa_use_cases()},
    **{("autowork", use_case): build_autowork_plan for use_case in list_autowork_iwa_use_cases()},
    **{("autoconnect", use_case): build_autoconnect_plan for use_case in list_autoconnect_iwa_use_cases()},
    **{("autozone", use_case): build_autozone_plan for use_case in list_autozone_iwa_use_cases()},
}


def build_registered_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    project_id = str(objective.web_project_id or "").strip().lower() or "autocinema"
    use_case = str(objective.use_case or "").strip().upper()
    builder = DETERMINISTIC_PLAN_BUILDERS.get((project_id, use_case))
    if builder is None:
        raise ValueError(f"Unsupported deterministic use case: {use_case}")
    return builder(objective)


__all__ = ["DETERMINISTIC_PLAN_BUILDERS", "DeterministicPlanBuilder", "build_registered_plan"]
