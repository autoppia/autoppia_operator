from __future__ import annotations

from autoppia_iwa.src.demo_webs.projects.p12_autolist.trajectories import load_autolist_use_case_completion_flows

import training._iwa_path  # noqa: F401  # must run before autoppia_iwa imports
from training.deterministic_harvester.builders.autolist import build_autolist_plan, list_autolist_iwa_use_cases
from training.deterministic_harvester.builders.registry import DETERMINISTIC_PLAN_BUILDERS, build_registered_plan
from training.deterministic_harvester.normalizer import DeterministicTaskObjective, normalize_task_row


def _add_task_clicked_task_row() -> dict:
    return {
        "id": "add-task-clicked-sample",
        "web_project_id": "autolist",
        "url": "http://localhost:8011/?seed=682",
        "prompt": "Add a task per task constraints",
        "specifications": {},
        "tests": [
            {
                "type": "CheckEventTest",
                "event_name": "AUTOLIST_ADD_TASK_CLICKED",
                "event_criteria": {},
            }
        ],
        "relevant_data": {},
        "use_case": {
            "name": "AUTOLIST_ADD_TASK_CLICKED",
            "constraints": [],
        },
    }


def test_list_autolist_iwa_use_cases_matches_iwa_loader() -> None:
    from_iwa = frozenset(load_autolist_use_case_completion_flows().keys())
    from_op = list_autolist_iwa_use_cases()
    assert from_iwa == from_op


def test_deterministic_registry_covers_all_iwa_autolist_use_cases() -> None:
    expected = frozenset(load_autolist_use_case_completion_flows().keys())
    registered = {uc for project_id, uc in DETERMINISTIC_PLAN_BUILDERS if project_id == "autolist"}
    assert registered == expected


def test_build_autolist_plan_aligns_first_navigate_to_task_url() -> None:
    row = _add_task_clicked_task_row()
    objective = normalize_task_row(row)
    assert isinstance(objective, DeterministicTaskObjective)
    plan = build_autolist_plan(objective)
    assert plan.metadata.get("source") == "iwa_p12_autolist"
    assert plan.actions[0]["type"] == "NavigateAction"
    assert plan.actions[0]["url"] == objective.task_url


def test_build_registered_plan_routes_autolist() -> None:
    row = _add_task_clicked_task_row()
    objective = normalize_task_row(row)
    plan = build_registered_plan(objective)
    assert plan.metadata.get("source") == "iwa_p12_autolist"
    assert any(a.get("type") in {"TypeAction", "ClickAction", "SelectAction", "SendKeysIWAAction", "WaitAction"} for a in plan.actions)
