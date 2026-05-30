from __future__ import annotations

import pytest

from training.deterministic_harvester.builders.registry import build_registered_plan
from training.deterministic_harvester.normalizer import normalize_task_row
from training.deterministic_harvester.trajectory_selectors import (
    list_iwa_use_cases,
    planned_actions_for_iwa_use_case_enriched,
)
from training.deterministic_harvester.use_case_selectors import has_semantic_selector_coverage

PROJECTS = (
    "autobooks",
    "autozone",
    "autodining",
    "autocrm",
    "automail",
    "autodelivery",
    "autolodge",
    "autoconnect",
    "autowork",
    "autocalendar",
    "autolist",
)

PORTS = {
    "autobooks": 8001,
    "autozone": 8002,
    "autodining": 8003,
    "autocrm": 8004,
    "automail": 8005,
    "autodelivery": 8006,
    "autolodge": 8007,
    "autoconnect": 8008,
    "autowork": 8009,
    "autocalendar": 8010,
    "autolist": 8011,
}


@pytest.mark.parametrize("project_id", PROJECTS)
def test_semantic_selector_coverage_for_all_use_cases(project_id: str) -> None:
    missing = [use_case for use_case in sorted(list_iwa_use_cases(project_id)) if not has_semantic_selector_coverage(project_id, use_case, seed=2)]
    assert not missing, f"Missing semantic selector coverage for {project_id}: {missing}"


@pytest.mark.parametrize("project_id", PROJECTS)
def test_xpath_candidates_are_tail_fallbacks_when_present(project_id: str) -> None:
    for use_case in sorted(list_iwa_use_cases(project_id)):
        actions = planned_actions_for_iwa_use_case_enriched(project_id, use_case, seed=2)
        for step in actions:
            candidates = step.get("selector_candidates")
            if not isinstance(candidates, list) or not candidates:
                continue
            types = [str(item.get("type") or "").strip() for item in candidates if isinstance(item, dict)]
            if "xpathSelector" not in types:
                continue
            if all(t == "xpathSelector" for t in types):
                # Some steps remain pure fallback XPath; coverage is validated
                # at use-case level by `test_semantic_selector_coverage_for_all_use_cases`.
                continue
            first_xpath = types.index("xpathSelector")
            assert all(t == "xpathSelector" for t in types[first_xpath:]), f"{project_id}/{use_case} has non-xpath candidate after xpath fallback: {types}"


@pytest.mark.parametrize("project_id", PROJECTS)
def test_registered_plan_metadata_declares_semantic_first_strategy(project_id: str) -> None:
    use_case = sorted(list_iwa_use_cases(project_id))[0]
    row = {
        "id": f"{project_id}-{use_case.lower()}-sample",
        "web_project_id": project_id,
        "url": f"http://localhost:{PORTS[project_id]}/?seed=2",
        "prompt": f"Execute {use_case}",
        "specifications": {},
        "tests": [
            {
                "type": "CheckEventTest",
                "event_name": use_case,
                "event_criteria": {},
            }
        ],
        "relevant_data": {},
        "use_case": {
            "name": use_case,
            "constraints": [],
        },
    }
    objective = normalize_task_row(row)
    plan = build_registered_plan(objective)
    assert plan.metadata.get("selector_strategy") == "semantic_first_with_xpath_fallback"
