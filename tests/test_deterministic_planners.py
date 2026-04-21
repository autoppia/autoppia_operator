from __future__ import annotations

from training.deterministic_harvester.normalizer import normalize_task_row
from training.deterministic_harvester.planners import build_deterministic_plan


def _task_row(*, use_case: str, prompt: str, url: str, event_criteria: dict, relevant_data: dict | None = None) -> dict:
    return {
        "id": f"{use_case.lower()}-task",
        "url": url,
        "prompt": prompt,
        "specifications": {},
        "tests": [
            {
                "type": "CheckEventTest",
                "event_name": use_case,
                "event_criteria": event_criteria,
            }
        ],
        "relevant_data": relevant_data or {},
        "use_case": {
            "name": use_case,
            "constraints": [],
        },
    }


def test_build_deterministic_plan_for_add_to_watchlist_uses_login_and_movie_resolution() -> None:
    task_row = _task_row(
        use_case="ADD_TO_WATCHLIST",
        prompt="movie_name equals 'Inception'",
        url="http://example.test/movies?seed=23",
        event_criteria={"movie": {"name": "Inception"}},
        relevant_data={"user_for_login": {"username": "bob", "password": "letmein"}},
    )
    objective = normalize_task_row(task_row)
    plan = build_deterministic_plan(objective)
    action_types = [action["type"] for action in plan.actions]
    assert action_types[:4] == ["NavigateAction", "TypeAction", "TypeAction", "ClickAction"]
    assert "OpenMovieDetailAction" not in action_types
    assert action_types[-2:] == ["ClickAction", "ClickAction"]
    assert action_types[-1] == "ClickAction"
    watchlist_selectors = plan.actions[-1].get("selector_candidates") or []
    assert watchlist_selectors
    assert watchlist_selectors[0]["type"] == "attributeValueSelector"


def test_build_deterministic_plan_for_edit_user_targets_profile_form() -> None:
    task_row = _task_row(
        use_case="EDIT_USER",
        prompt="first_name equals 'Ben', website equals 'https://example.org'",
        url="http://example.test/profile?seed=9",
        event_criteria={"first_name": "Ben", "website": "https://example.org"},
        relevant_data={"user_for_login": {"username": "ben", "password": "secret123"}},
    )
    objective = normalize_task_row(task_row)
    plan = build_deterministic_plan(objective)
    assert plan.actions[0]["type"] == "NavigateAction"
    first_name_actions = [action for action in plan.actions if action["type"] == "TypeAction" and action.get("field_name") == "first name"]
    assert first_name_actions
    assert first_name_actions[0]["selector_candidates"][0]["type"] == "attributeValueSelector"
    assert plan.actions[-1]["selector_candidates"][0]["type"] in {
        "attributeValueSelector",
        "idSelector",
        "tagContainsSelector",
        "textSelector",
    }


def test_build_deterministic_plan_preserves_localhost_origin() -> None:
    task_row = _task_row(
        use_case="CONTACT",
        prompt="subject equals 'Local test'",
        url="http://localhost:3000/?seed=41",
        event_criteria={"subject": "Local test"},
    )
    task_row["web_project_id"] = "autocinema"

    objective = normalize_task_row(task_row)
    plan = build_deterministic_plan(objective)

    assert plan.actions[0]["url"] == "http://localhost:3000/contact?seed=41"
