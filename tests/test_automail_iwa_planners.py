from __future__ import annotations

from autoppia_iwa.src.demo_webs.projects.p06_automail.trajectories import load_automail_use_case_completion_flows

import training._iwa_path  # noqa: F401  # must run before autoppia_iwa imports
from training.deterministic_harvester.builders.automail import build_automail_plan, list_automail_iwa_use_cases
from training.deterministic_harvester.builders.registry import DETERMINISTIC_PLAN_BUILDERS, build_registered_plan
from training.deterministic_harvester.normalizer import DeterministicTaskObjective, normalize_task_row


def _search_email_task_row() -> dict:
    return {
        "id": "search-email-sample",
        "web_project_id": "automail",
        "url": "http://localhost:8005/?seed=53",
        "prompt": "Search email per task constraints",
        "specifications": {},
        "tests": [
            {
                "type": "CheckEventTest",
                "event_name": "SEARCH_EMAIL",
                "event_criteria": {},
            }
        ],
        "relevant_data": {},
        "use_case": {
            "name": "SEARCH_EMAIL",
            "constraints": [],
        },
    }


def test_list_automail_iwa_use_cases_matches_iwa_loader() -> None:
    from_iwa = frozenset(load_automail_use_case_completion_flows().keys())
    from_op = list_automail_iwa_use_cases()
    assert from_iwa == from_op


def test_deterministic_registry_covers_all_iwa_automail_use_cases() -> None:
    expected = frozenset(load_automail_use_case_completion_flows().keys())
    registered = {uc for project_id, uc in DETERMINISTIC_PLAN_BUILDERS if project_id == "automail"}
    assert registered == expected


def test_build_automail_plan_aligns_first_navigate_to_task_url() -> None:
    row = _search_email_task_row()
    objective = normalize_task_row(row)
    assert isinstance(objective, DeterministicTaskObjective)
    plan = build_automail_plan(objective)
    assert plan.metadata.get("source") == "iwa_p06_automail"
    assert plan.actions[0]["type"] == "NavigateAction"
    assert plan.actions[0]["url"] == objective.task_url


def test_build_registered_plan_routes_automail() -> None:
    row = _search_email_task_row()
    objective = normalize_task_row(row)
    plan = build_registered_plan(objective)
    assert plan.metadata.get("source") == "iwa_p06_automail"
    assert any(a.get("type") in {"TypeAction", "ClickAction", "SelectAction", "SendKeysIWAAction"} for a in plan.actions)
