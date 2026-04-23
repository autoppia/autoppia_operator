from __future__ import annotations

from autoppia_iwa.src.demo_webs.projects.p02_autobooks.trajectories import load_autobooks_use_case_completion_flows

import training._iwa_path  # noqa: F401  # must run before autoppia_iwa imports
from training.deterministic_harvester.builders.autobooks import build_autobooks_plan, list_autobooks_iwa_use_cases
from training.deterministic_harvester.builders.registry import DETERMINISTIC_PLAN_BUILDERS, build_registered_plan
from training.deterministic_harvester.normalizer import DeterministicTaskObjective, normalize_task_row


def _search_book_task_row() -> dict:
    return {
        "id": "search-book-sample",
        "web_project_id": "autobooks",
        "url": "http://localhost:8001/?seed=569",
        "prompt": "Search for the book 'The Silent Patient' in the database",
        "specifications": {},
        "tests": [
            {
                "type": "CheckEventTest",
                "event_name": "SEARCH_BOOK",
                "event_criteria": {},
            }
        ],
        "relevant_data": {},
        "use_case": {
            "name": "SEARCH_BOOK",
            "constraints": [],
        },
    }


def test_list_autobooks_iwa_use_cases_matches_iwa_loader() -> None:
    from_iwa = frozenset(load_autobooks_use_case_completion_flows().keys())
    from_op = list_autobooks_iwa_use_cases()
    assert from_iwa == from_op


def test_deterministic_registry_covers_all_iwa_autobooks_use_cases() -> None:
    expected = frozenset(load_autobooks_use_case_completion_flows().keys())
    registered = {uc for project_id, uc in DETERMINISTIC_PLAN_BUILDERS if project_id == "autobooks"}
    assert registered == expected


def test_build_autobooks_plan_aligns_first_navigate_to_task_url() -> None:
    row = _search_book_task_row()
    objective = normalize_task_row(row)
    assert isinstance(objective, DeterministicTaskObjective)
    plan = build_autobooks_plan(objective)
    assert plan.metadata.get("source") == "iwa_p02_autobooks"
    assert plan.actions[0]["type"] == "NavigateAction"
    assert plan.actions[0]["url"] == objective.task_url


def test_build_registered_plan_routes_autobooks() -> None:
    row = _search_book_task_row()
    objective = normalize_task_row(row)
    plan = build_registered_plan(objective)
    assert plan.metadata.get("source") == "iwa_p02_autobooks"
    assert any(a.get("type") == "TypeAction" for a in plan.actions)
